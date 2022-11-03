import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2, 4"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import json

from dataset import UCF101_DATASETS
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

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./cfg/ucf101.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'])
    config = DotMap(config)

    print(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    print(device)
    model = CLIP(**yaml.safe_load(open("./cfg/ViT-B-16.yaml")))
    if config.network.pretrained.clip != "":
        assert os.path.exists(config.network.pretrained.clip), "CLIP pretrained weights does not exist"
        model.load_state_dict(torch.load(config.network.pretrained.clip))
    
    model = torch.nn.DataParallel(model).to(device)

    if config.network.sim_header == "meanP":
        fusion_model = None
    else:
        raise NotImplementedError()
        # need to change cfg file as well to specify which header we are using and the pretrained weights

    transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.GaussianBlur(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])
    transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    train_data = UCF101_DATASETS(config.data.root, 16, 1, transform_train, False)
    test_data = UCF101_DATASETS(config.data.root, 16, 1, transform_test, True)

    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)

    loss_img = KLLoss()
    loss_txt = KLLoss()

    param_group = [ {'params': model.parameters()}] 
    if fusion_model:
        param_group.append({'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio})  
                            # maybe we need to use a fucion model later
    optimizer = optim.SGD(param_group, config.solver.lr, momentum=config.solver.momentum, weight_decay=config.solver.weight_decay)
    #TOREAD: need to check the details
    lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )

    best_prec1 = 0.0

    classes_names = list(json.load(open("./cfg/ucf101_labels.json")).keys())
    classes, num_text_aug, text_dict = text_prompt(classes_names)

    prec1, prec5 = validate(-1, test_loader, classes, device, model, num_text_aug, fusion_model=fusion_model)
    print(f"[before training, acc_top1 = {prec1}, acc_top5 = {prec5}]")
    for epoch in range(config.solver.epochs):
        model.train()
        # fusion_model.train()
        bar = tqdm(train_loader)
        for batch_idx,(images,list_id) in enumerate(bar):
            if config.solver.type != 'monitor':
                if (batch_idx+1) == 1 or (batch_idx+1) % 10 == 0:
                    lr_scheduler.step(epoch + batch_idx / len(train_loader))

            optimizer.zero_grad()

            text_id = np.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            images= images.to(device) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)
            image_features, text_features = model(images, texts, fusion_model)
            # cosine similarity as logits
            logit_scale = model.module.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            ground_truth = torch.tensor(gen_label(list_id), dtype=image_features.dtype, device=device)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)
            total_loss = (loss_imgs + loss_texts)/2

            total_loss.backward()
            optimizer.step()

            bar.set_postfix_str(f"Total loss: {round(float(total_loss), 3)}")


        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1, prec5 = validate(epoch, test_loader, classes, device, model, num_text_aug, fusion_model=fusion_model)


        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1,best_prec1))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)
        ckpt = { "state_dict": model.module.state_dict()} 
        if fusion_model:
           ckpt["vision_fusion"] = fusion_model.state_dict()
        torch.save(ckpt, filename)

if __name__ == '__main__':
    main()