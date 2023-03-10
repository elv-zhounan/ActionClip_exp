import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"

import warnings
warnings.filterwarnings("ignore")

import torch
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
    parser.add_argument('--config', '-cfg', default='./cfg/exp/k400/k400.yaml')
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

    # add teacher if distilling
    if config.network.teacher is not None and config.network.teacher != "":
        logger.info("preparing teacher")
        teacher_model = CLIP(**yaml.safe_load(open(f"./cfg/model/{config.network.arch}.yaml")))
        teacher_ckpt = torch.load(config.network.teacher)
        if "model_state_dict" in teacher_ckpt:
            # meaning this is the official released model weights
            teacher_model.load_state_dict(teacher_ckpt["model_state_dict"])
            teacher_model.eval()
            teacher_fusion_state_dict = {}
            for k, v in teacher_ckpt["fusion_model_state_dict"].items():
                teacher_fusion_state_dict[k[7:]] = v
            from fusion_vision import Fusion
            teacher_fusion = Fusion(77, 512)
            teacher_fusion.load_state_dict(teacher_fusion_state_dict)
            teacher_fusion.eval()
        elif "state_dict" in teacher_ckpt:
            teacher_model.load_state_dict(teacher_ckpt["state_dict"])
            teacher_model.eval()
            teacher_fusion = None
        else:
            teacher_model.load_state_dict(teacher_ckpt)
            teacher_model.eval()
            teacher_fusion = None

        teacher_logit_scale = teacher_model.logit_scale.exp()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model = torch.nn.DataParallel(teacher_model).to(device)

        if teacher_fusion:
            for param in teacher_fusion.parameters():
                param.requires_grad = False
            teacher_fusion = torch.nn.DataParallel(teacher_fusion).to(device)

    else:
        teacher_model = None
        teacher_fusion = None


    logger.info("building model")
    model = CLIP(**yaml.safe_load(open(f"./cfg/model/{config.network.arch}.yaml")))

    if config.network.continue_from != "":
        logger.info("loading weights to continue training")
        assert os.path.exists(config.network.continue_from), "CLIP pretrained weights does not exist"
        model.load_state_dict(torch.load(config.network.continue_from)["state_dict"])
    elif config.network.pretrained != "":
        logger.info("loading pretrained weights for transfer learning")
        assert os.path.exists(config.network.pretrained), "CLIP pretrained weights does not exist"
        model.load_state_dict(torch.load(config.network.pretrained))
    logger.info("converting to fp16")
    convert_weights(model)

    # construct the params group
    vision_params = list(map(id, model.visual.parameters()))
    text_params = filter(lambda p: id(p) not in vision_params, model.parameters())
    param_group = [{'params': text_params}, {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio}]
    model = torch.nn.DataParallel(model).to(device)

    # TODO according to the paper, the post vision fusion brings very few help. so I would not use that for now
    if config.network.sim_header == "meanP":
        fusion_model = None
    else:
        raise NotImplementedError()
        param_group.append({'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}) 
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
    loss_img = KLLoss()
    loss_txt = KLLoss()

    soft_loss_img = torch.nn.KLDivLoss(size_average=True, reduce=True, reduction='sum', log_target=False)
    soft_loss_text = torch.nn.KLDivLoss(size_average=True, reduce=True, reduction='sum', log_target=False)


    optimizer = optim.AdamW(param_group, betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8, weight_decay=config.solver.weight_decay)

    
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
        if fusion_model is not None:
            fusion_model.train()
        bar = tqdm(train_loader)
        for batch_idx,(images, list_id) in enumerate(bar):
            
            # TODO add scheduler
            # if config.solver.type != 'monitor':
            #     if (batch_idx+1) == 1 or (batch_idx+1) % 10 == 0:
            #         lr_scheduler.step(epoch + batch_idx / len(train_loader))
            _iter += 1
            optimizer.zero_grad()

            # prepare
            text_id = np.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            # send to device
            images = images.to(device)
            texts = texts.to(device)

            # forward
            image_features, text_features = model(images, texts)
            image_features, text_features = norm_features(image_features, text_features, fusion_model)

            # get logits in one batch
            logit_scale = model.module.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            ground_truth = torch.tensor(gen_label(list_id), dtype=image_features.dtype)
            ground_truth = ground_truth.to(device)

            # get hard loss using KL
            hard_loss_imgs = loss_img(logits_per_image, ground_truth)
            hard_loss_texts = loss_txt(logits_per_text, ground_truth)
            hard_loss = (hard_loss_imgs + hard_loss_texts) / 2

            # getting soft label is distilling
            if teacher_model is not None:
                with torch.no_grad():
                    # we do not want to compute the logits inside the model, becuase the input will be splitted,
                    # then the logits given by the model will not be a complete logits on the whole batch
                    teacher_image_features, teacher_text_features = teacher_model(images, texts)
                    teacher_image_features, teacher_text_features = norm_features(teacher_image_features, teacher_text_features, teacher_fusion)

                    teacher_pred_per_image = torch.softmax(teacher_logit_scale * teacher_image_features @ teacher_text_features.t(), dim=1)
                    teacher_pred_per_text = torch.softmax(teacher_logit_scale * teacher_text_features @ teacher_image_features.t(), dim=1)

                    soft_label_per_image = teacher_pred_per_image.to(dtype=image_features.dtype).detach()
                    soft_label_per_text = teacher_pred_per_text.to(dtype=image_features.dtype).detach()


                # get soft label loss
                log_prob_img = torch.log_softmax(logits_per_image, dim=1)
                log_prob_text = torch.log_softmax(logits_per_text, dim=1)
                soft_loss_imgs = soft_loss_img(log_prob_img, soft_label_per_image)
                soft_loss_texts = soft_loss_text(log_prob_text, soft_label_per_text)
                soft_loss = (soft_loss_imgs + soft_loss_texts) / 2

                # total loss include both of soft label loss and hard label loss
                total_loss =  config.solver.soft_label_ratio * soft_loss + (1-config.solver.soft_label_ratio) * hard_loss

            else:
                total_loss = hard_loss

            # back propagate
            total_loss.backward()

            # update parmas
            convert_models_to_fp32(model)
            optimizer.step()
            convert_weights(model)

            # logging
            bar.set_postfix_str(f"Total loss: {round(float(total_loss), 3)}")
            writer.add_scalar('train/hard_img_loss', hard_loss_imgs.data, _iter)
            writer.add_scalar('train/hard_text_loss', hard_loss_texts.data, _iter)
            if teacher_model is not None:
                writer.add_scalar('train/soft_img_loss', soft_loss_imgs.data, _iter)
                writer.add_scalar('train/soft_text_loss', soft_loss_texts.data, _iter)
            writer.add_scalar('train/total_loss', total_loss.data, _iter)


        """test"""
        prec1, prec5 = validate(epoch, test_loader, classes, num_text_aug, device, model, fusion_model=fusion_model)
        writer.add_scalar('test/acc1', prec1, epoch)
        writer.add_scalar('test/acc5', prec5, epoch)
        # convert to fp32
        convert_models_to_fp32(model)
        # saving the best checkpoint so far
        ckpt = { "state_dict": model.module.state_dict()} 
        if fusion_model:
            ckpt["vision_fusion"] = fusion_model.state_dict()
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