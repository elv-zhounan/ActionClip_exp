"""
    declaration:
        for simplification: I won't add visionn fusion in this script, just using mean and norm
"""
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

from dataset import DATASETS
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

# mp specific import
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

class loss_module(nn.Module):
    def __init__(self,):
        super(loss_module, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_feat, text_feat, labels):
        # gather the tensors from all processes
        image_feat = all_gather_with_grad(image_feat)
        text_feat = all_gather_with_grad(text_feat)
        labels = all_gather_with_grad(labels)


        # we can get the in-batch contrastive loss
        ground_truth = torch.eq(labels, labels.t()).to(image_feat.dtype)
        ground_truth = ground_truth / ground_truth.sum(1,keepdim=True)     

        # computing loss
        sim_i2t = self.logit_scale.exp() * image_feat @ text_feat.t()
        sim_t2i = self.logit_scale.exp() * text_feat @ image_feat.t()  
        loss_i2t_gt = -torch.sum(F.log_softmax(sim_i2t, dim=1)*ground_truth,dim=1).mean()
        loss_t2i_gt = -torch.sum(F.log_softmax(sim_t2i, dim=1)*ground_truth,dim=1).mean()
        loss_gt = (loss_i2t_gt+loss_t2i_gt)/2

        return loss_gt


def train(model, train_loader, criterion, optimizer, device, num_text_aug, text_dict, _iter=0, writer=None):
    model.train()
    bar = tqdm(train_loader) if writer is not None else train_loader
    for batch_idx,(images, list_id) in enumerate(bar):
        _iter += 1
        optimizer.zero_grad()

        # prepare
        text_id = np.random.randint(num_text_aug, size=len(list_id))
        texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

        # send to device
        labels = list_id.view(-1, 1).cuda(device)
        images = images.cuda(device)
        texts = texts.cuda(device)
        
        # forward
        image_feat, text_feat = model(images, texts)
        image_feat = image_feat.mean(dim=1, keepdim=False)
        image_feat = F.normalize(image_feat)
        text_feat = F.normalize(text_feat)

        # get loss
        loss = criterion(image_feat, text_feat, labels)

        # back propagate
        loss.backward()

        # update parmas
        optimizer.step()

        # logging
        if writer is not None:
            bar.set_postfix_str(f"Total loss: {round(float(loss.data), 3)}")
            writer.add_scalar('train/total_loss', loss.data, _iter)

    return _iter

def main(gpu, args):
    args.gpu = gpu # aka local_rank
    args.rank = gpu # aka global rank, since we only use one device, so local rank = global rank

    """------------------------------------------define config------------------------------------------"""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    saveRoot = config["data"]["saveRoot"]
    if "subset" in config["data"]:
        working_dir = os.path.join(saveRoot, config['network']['arch'], f"{config['data']['num_segments']}_frames_subset{config['data']['subset']}_ddp")
        subset = list(json.load(open(f"./cfg/exp/k400/subset{config['data']['subset']}.json")).keys())
    else:
        working_dir = os.path.join(saveRoot, config['network']['arch'], f"{config['data']['num_segments']}_frames_ddp")
        subset = None
    
    if gpu == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)

    config = DotMap(config)

    ############### Init env for distributed training ###############
    dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()
    torch.cuda.set_device(args.gpu)
    """------------------------------------------define config finish------------------------------------------"""


    """------------------------------------------define model------------------------------------------"""
    logger.info(f"gpu:{gpu} building model")
    model = CLIP(**yaml.safe_load(open(f"./cfg/model/{config.network.arch}.yaml")), dropout=config.network.dropout, emb_dropout=config.network.emb_dropout)
    if config.network.pretrained != "":
        model.load_state_dict(torch.load(config.network.pretrained))

    # construct the params group
    logger.info(f"gpu:{gpu} constructing param groups")
    vision_params = list(map(id, model.visual.parameters()))
    text_params = filter(lambda p: id(p) not in vision_params, model.parameters())
    param_group = [{'params': text_params}, {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.lr_ratio}]

    # convert to ddp mode
    model.cuda(args.gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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

    if config.data.dataset_info != "":
        train_info = json.load(open(os.path.join(config.data.dataset_info, "train_info.json")))
    else:
        train_info = None
    train_dataset = DATASETS(config.data.root, config.data.num_segments, transform_train, False, config.data.cut_freq, subset=subset, info=train_info)
    logger.info(f"gpu:{gpu} train dataset length: {len(train_dataset)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    train_loader = DataLoader(train_dataset,batch_size=config.data.batch_size//args.num_gpus,num_workers=config.data.workers//args.num_gpus, shuffle=False, drop_last=True, sampler=train_sampler)
    """------------------------------------------define dataset finish------------------------------------------"""


    """------------------------------------------define solver------------------------------------------"""
    criterion = loss_module().cuda(gpu)
    optimizer = optim.AdamW(param_group, betas=(0.9, 0.98), eps=1e-8, weight_decay=config.solver.weight_decay)
   
    """------------------------------------------define solver finish------------------------------------------"""


    """------------------------------------------training loop start------------------------------------------"""
    classes, num_text_aug, text_dict = text_prompt(train_dataset.class_names)
    _iter = 0
    if gpu == 0:
        writer = SummaryWriter(working_dir)
    else:
        writer = None
   
    for epoch in range(config.solver.epochs):
        """train"""
        _iter = train(model, train_loader, criterion, optimizer, gpu, num_text_aug, text_dict, _iter, writer=writer)

        """save the model it is on gpu 0"""
        if gpu == 0:
            # saving the best checkpoint so far
            ckpt = { "state_dict": model.module.state_dict()} 
            # saving the best 
            if epoch % config.solver.save_freq == 0:
                filename = f"{working_dir}/{epoch+1}.pt"
                torch.save(ckpt, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./cfg/exp/ucf101/ucf101_ddp.yaml')
    args = parser.parse_args()

    args.dist_url = 'tcp://127.0.0.1:10001'
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.num_gpus * 1 # we only use one device

    mp.spawn(main, nprocs=args.num_gpus, args=(args,), join=True)