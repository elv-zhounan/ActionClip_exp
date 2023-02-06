import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3, 4"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import json

import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle
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

def main():
    """------------------------------------------define config------------------------------------------"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./cfg/exp/k400/k400.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['arch'], config['data']['dataset'], f"{config['data']['num_segments']}_frames")
    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    writer = SummaryWriter(working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    print(device)
    """------------------------------------------define config finish------------------------------------------"""


    """------------------------------------------define model------------------------------------------"""
    model = CLIP(**yaml.safe_load(open(f"./cfg/model/{config.network.arch}.yaml")))
    if config.network.pretrained.clip != "":
        assert os.path.exists(config.network.pretrained.clip), "CLIP pretrained weights does not exist"
        model.load_state_dict(torch.load(config.network.pretrained.clip))

    # construct the params group
    vision_params = list(map(id, model.visual.parameters()))
    text_params = filter(lambda p: id(p) not in vision_params, model.parameters())
    param_group = [{'params': text_params}, {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio}]

    model = torch.nn.DataParallel(model).to(device)

    # TODO
    if config.network.sim_header == "meanP":
        fusion_model = None
    else:
        raise NotImplementedError()
        param_group.append({'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}) 
    """------------------------------------------define model finish------------------------------------------"""


    """------------------------------------------define dataset------------------------------------------"""
    # plan is save dataset to a pickle first, because it takes too long to build
    transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])
    transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    ds_cls = getattr(dataset, config.data.dataset+"_DATASETS")
    # train set, try load pick first
    train_dataset_path = os.path.join(working_dir, "train_datasset.obj")
    if os.path.exists(train_dataset_path):
        print("Loading training set from pickle file")
        train_dataset = pickle.load(open(train_dataset_path, "rb"))
    else:
        train_dataset = ds_cls(config.data.root, config.data.num_segments, transform_train, False, config.data.cut_freq)
        pickle.dump(train_dataset, open(train_dataset_path, "wb"))
    # test set, try load pick first
    test_dataset_path = os.path.join(working_dir, "test_datasset.obj")
    if os.path.exists(test_dataset_path):
        print("Loading testing set from pickle file")
        test_dataset = pickle.load(open(test_dataset_path, "rb"))
    else:
        test_dataset = ds_cls(config.data.root, config.data.num_segments, transform_test, True, config.data.cut_freq)
        pickle.dump(test_dataset, open(test_dataset_path, "wb"))
    # construct loader
    train_loader = DataLoader(train_dataset,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    """------------------------------------------define dataset finish------------------------------------------"""
    
    
    """------------------------------------------define solver------------------------------------------"""
    loss_img = KLLoss()
    loss_txt = KLLoss()

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
    print("building classes descriptions ...... ")
    classes, num_text_aug, text_dict = text_prompt(train_dataset.class_names)
    _iter = 0
    for epoch in range(config.solver.epochs):
        """train"""
        model.train()
        if fusion_model:
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
            ground_truth = torch.tensor(gen_label(list_id), dtype=images.dtype)

            # send to device
            images= images.to(device)
            texts = texts.to(device)
            ground_truth = ground_truth.to(device)

            # forward
            image_features, text_features = model(images, texts, fusion_model, ground_truth, loss_img, loss_txt)
           
            # get logits in one batch
            logit_scale = model.module.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            # get loss using KL 
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts)/2

            # update parmas
            total_loss.backward()
            optimizer.step()

            # logging
            bar.set_postfix_str(f"Total loss: {round(float(total_loss), 3)}")
            writer.add_scalar('train/img_loss', loss_imgs.data, _iter)
            writer.add_scalar('train/text_loss', loss_texts.data, _iter)
            writer.add_scalar('train/total_loss', total_loss.data, _iter)


        """test"""
        prec1, prec5 = validate(epoch, test_loader, classes, device, model, num_text_aug, fusion_model=fusion_model)
        writer.add_scalar('test/acc1', prec1, epoch)
        writer.add_scalar('test/acc5', prec5, epoch)
        # saving the best checkpoint so far
        print('Saving:')
        ckpt = { "state_dict": model.module.state_dict()} 
        if fusion_model:
            ckpt["vision_fusion"] = fusion_model.state_dict()
        # saving the best 
        if prec1 > best_prec1:
            filename = "{}/best_model.pt".format(working_dir)
            torch.save(ckpt, filename)
        # savinn the latest
        filename = "{}/last_model.pt".format(working_dir)
        torch.save(ckpt, filename)
        # ploting 
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1,best_prec1))


if __name__ == '__main__':
    main()