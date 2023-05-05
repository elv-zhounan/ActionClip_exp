import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import json

import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
from utils import *
from data import text_prompt
from loss import KLLoss
from lr import WarmupCosineAnnealingLR
from model import CLIP
from inference import validate

from torch.utils.tensorboard import SummaryWriter

def norm_features(image_features:torch.Tensor, text_features:torch.Tensor, vision_fusion=None):
    if vision_fusion is not None:
        image_features = vision_fusion(image_features)
    else:
        image_features = image_features.mean(dim=1, keepdim=False)

    image_features = norm(image_features)
    text_features = norm(text_features)

    return image_features, text_features

def main():
    """------------------------------------------define config------------------------------------------"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./cfg/exp/k400/k400_momentum.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if "subset" in config["data"]:
        working_dir = os.path.join('./exp', config['network']['arch'], config['data']['dataset'], f"{config['data']['num_segments']}_frames_subset{config['data']['subset']}")
        subset = list(json.load(open(f"./cfg/exp/k400/subset{config['data']['subset']}.json")).keys())
    else:
        working_dir = os.path.join('./exp', config['network']['arch'], config['data']['dataset'], f"{config['data']['num_segments']}_frames")
        subset = None

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    logger.info(f"using {device}")
    """------------------------------------------define config finish------------------------------------------"""


    """------------------------------------------define model------------------------------------------"""
    logger.info("building model")
    model = CLIP(**yaml.safe_load(open(f"./cfg/model/{config.network.arch}.yaml")), dropout=config.network.dropout, emb_dropout=config.network.emb_dropout)
    model_m = CLIP(**yaml.safe_load(open(f"./cfg/model/{config.network.arch}.yaml")), dropout=0, emb_dropout=0)
    model_m.eval()

    model.load_state_dict(torch.load(config.network.pretrained))
    model.load_state_dict(torch.load(config.network.pretrained))
    logger.info("converting to fp16")
    convert_weights(model)
    convert_weights(model_m)


    # construct the params group
    vision_params = list(map(id, model.visual.parameters()))
    text_params = filter(lambda p: id(p) not in vision_params, model.parameters())
    param_group = [{'params': text_params}, {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.lr_ratio}]
    model = torch.nn.DataParallel(model).to(device)
    model_m = torch.nn.DataParallel(model).to(device)
    """------------------------------------------define model finish------------------------------------------"""


    """------------------------------------------define dataset------------------------------------------"""
    # plan is save dataset to a pickle first, because it takes too long to build
    transform_train = transforms.Compose([
            transforms.Resize(config.data.input_size),
            transforms.CenterCrop(config.data.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])
    transform_test = transforms.Compose([
            transforms.Resize(config.data.input_size),
            transforms.CenterCrop(config.data.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    ds_cls = getattr(dataset, config.data.dataset+"_DATASETS")

    # train set, try load pick first
    train_info = json.load(open("./cfg/data_k400/train_info.json"))
    train_dataset = ds_cls(config.data.root, config.data.num_segments, transform_train, False, config.data.cut_freq, subset=subset, info=train_info)
    test_info = json.load(open("./cfg/data_k400/test_info.json"))
    test_dataset = ds_cls(config.data.root, config.data.num_segments, transform_test, True, config.data.cut_freq, subset=subset, info=test_info)

    logger.info(f"train dataset length: {len(train_dataset)}, test dataset length: {len(test_dataset)}")

    # construct loader
    train_loader = DataLoader(train_dataset,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    """------------------------------------------define dataset finish------------------------------------------"""
    
    
    """------------------------------------------define solver------------------------------------------"""
    optimizer = optim.AdamW(param_group, betas=(0.9, 0.98), eps=1e-8, weight_decay=config.solver.weight_decay)

    image_queue = nn.functional.normalize(torch.randn(model.module.embed_dim, config.solver.queue_size), dim=0)
    text_queue = nn.functional.normalize(torch.randn(model.module.embed_dim, config.solver.queue_size), dim=0)
    ptr_queue = torch.zeros(1, dtype=torch.long)

    #TODO need to check the details
    # lr_scheduler = WarmupCosineAnnealingLR(
    #         optimizer,
    #         config.solver.epochs,
    #         warmup_epochs=config.solver.lr_warmup_step
    #     )
    """------------------------------------------define solver finish------------------------------------------"""


    """------------------------------------------training loop start------------------------------------------"""
    best_prec1 = 0.0
    # for eval use, 
    # classes var is not using in training loop
    # num_text_aug and text_dict are for generating text description in training process
    logger.info("building classes descriptions ...... ")
    classes, num_text_aug, text_dict = text_prompt(train_dataset.class_names)
    _iter = 0
    logger.info("start training loop")
    writer = SummaryWriter(working_dir)
    
    for epoch in range(config.solver.epochs):
        """train"""
        model.train()
        bar = tqdm(train_loader)
        for batch_idx,(images, list_id) in enumerate(bar):
            
            # TODO add scheduler
            # if (batch_idx+1) == 1 or (batch_idx+1) % 10 == 0:
            #     lr_scheduler.step(epoch + batch_idx / len(train_loader))
            _iter += 1
            optimizer.zero_grad()

            # prepare
            text_id = np.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            # send to device
            images = images.to(device)
            texts = texts.to(device)

            # forward
            image_feat, text_feat = model(images, texts)
            image_feat, text_feat = norm_features(image_feat, text_feat, None)

            with torch.no_grad():
                momentum_update(model.module, model_m.module. config.network.momentum)

                image_feat_m, text_feat_m = model_m(images, texts)
                image_feat_m, text_feat_m = norm_features(image_feat_m, text_feat_m, None)
                image_feat_m_all = torch.cat([image_feat_m.t(), image_queue.clone().detach()],dim=1)   
                text_feat_m_all = torch.cat([text_feat_m.t(),text_queue.clone().detach()],dim=1)

                # TODO is it the correct way to deal with the logit_scale ?
                logit_scale = model.module.logit_scale.exp()
                sim_i2t_m = logit_scale * image_feat_m @ text_feat_m_all
                sim_t2i_m = logit_scale * text_feat_m @ image_feat_m_all   

                sim_targets = torch.zeros(sim_i2t_m.size()).to(device)
                sim_targets.fill_diagonal_(1)   

                sim_i2t_targets = config.solver.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - config.solver.alpha) * sim_targets
                sim_t2i_targets = config.solver.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - config.solver.alpha) * sim_targets        

            logit_scale = model.module.logit_scale.exp()
            sim_i2t = logit_scale * image_feat @ text_feat_m_all
            sim_t2i = logit_scale *  text_feat @ image_feat_m_all

            loss_i2t = F.kl_div(input=F.log_softmax(sim_i2t, dim=1), target=sim_i2t_targets, log_target=False)
            loss_t2i = F.kl_div(input=F.log_softmax(sim_t2i, dim=1), target=sim_t2i_targets, log_target=False)
            loss_itc = (loss_i2t+loss_t2i)/2

            dequeue_and_enqueue(image_queue, text_queue, ptr_queue, image_feat_m, text_feat_m, config.solver.queue_size) 

            # back propagate
            loss_itc.backward()

            # update parmas
            convert_models_to_fp32(model)
            optimizer.step()
            convert_weights(model)

            # logging
            bar.set_postfix_str(f"Total loss: {round(float(loss_itc.data), 3)}")
            writer.add_scalar('train/total_loss', loss_itc.data, _iter)


        """test"""
        prec1, prec5 = validate(epoch, test_loader, classes, num_text_aug, device, model, fusion_model=None)
        writer.add_scalar('test/acc1', prec1, epoch)
        writer.add_scalar('test/acc5', prec5, epoch)
        # convert to fp32
        convert_models_to_fp32(model)
        # saving the best checkpoint so far
        ckpt = { "state_dict": model.module.state_dict()} 
        # saving the best 
        if prec1 > best_prec1:
            filename = "{}/best_model.pt".format(working_dir)
            torch.save(ckpt, filename)
        # convert back to fp16
        convert_weights(model)
        # savinn the latest
        filename = "{}/last_model.pt".format(working_dir)
        torch.save(ckpt, filename)
        # ploting 
        best_prec1 = max(prec1, best_prec1)
        logger.info('Testing: {}/{}'.format(prec1,best_prec1))


if __name__ == '__main__':
    main()