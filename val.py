import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import json
from loguru import logger

import torch
import torch.nn.functional as F
from torchvision import transforms
import yaml
from model import CLIP
from fusion_vision import Fusion
from typing import List, Union
from tokenizer import SimpleTokenizer
from data import *
from utils import *
from dataset import DATASETS
# from utils import get_video_frames, norm
import time
from tqdm import tqdm
from train_test_split import get_frames

def encode_image(model:CLIP, input:Union[List[torch.Tensor], torch.Tensor]):
    return model.encode_image(input)

def encode_text(model:CLIP, tokenizer:SimpleTokenizer, input:str):
    input = tokenize(tokenizer, input)
    return model.encode_text(input)

def norm_features(image_features:torch.Tensor, text_features:torch.Tensor, vision_fusion=None):
    if vision_fusion is not None:
        image_features = vision_fusion(image_features)
    else:
        image_features = image_features.mean(dim=1, keepdim=False)

    image_features = norm(image_features)
    text_features = norm(text_features)

    return image_features, text_features

@torch.no_grad()
def validate(epoch, test_loader, classes, num_text_aug, device, model, fusion_model=None):
    model.eval()
    if fusion_model:
        fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():
        text_inputs = classes.to(device)
        bar = tqdm(test_loader)
        for _, (image, class_id) in enumerate(bar):
            b, t, c, h, w = image.shape
            image = image.to(device)
            class_id = class_id.to(device)
            image_features, text_features = model(image, text_inputs)
            image_features, text_features = norm_features(image_features, text_features, fusion_model)

            logits =  image_features @ text_features.t()
            pred = logits.view(b, num_text_aug, -1).mean(dim=1, keepdim=False).softmax(dim=-1)

            values_1, indices_1 = pred.topk(1, dim=-1)
            values_5, indices_5 = pred.topk(5, dim=-1)

            num += b
            for i in range(b):
                if class_id[i] == indices_1[i%len(classes)]:
                    corr_1 += 1
                if class_id[i] in indices_5[i%len(classes)]:
                    corr_5 += 1

            bar.set_postfix_str(f" up to now in avg:  Top1: {round(float(corr_1) / num, 3)}, Top5: {round(float(corr_5) / num, 3)}")


    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    print('Epoch: [{}]: Top1: {}, Top5: {}'.format(epoch, top1, top5))
    return top1, top5

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", type=str, default="/home/elv-zhounan/ActionClip_exp/cfg/model/ViT-B-16.yaml")
    parser.add_argument("--ckpt_path", type=str, default="/pool0/ml/elv-zhounan/action/ucf101/exp/actionclip/ViT-B-16/8_frames_ddp/31.pt")
    parser.add_argument("--test_root", type=str, default="/pool0/ml/elv-zhounan/action/ucf101/images")
    parser.add_argument("--test_info", type=str, default="")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--freq", type=int, default=1)
    parser.add_argument("--subset", type=str, default="") #/home/elv-zhounan/ActionClip_exp/cfg/exp/k400/subset50.json
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(args.model_cfg))
    ckpt = torch.load(args.ckpt_path)
    net = CLIP(**cfg)

    if "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
        fusion_state_dict = {}
        for k, v in ckpt["fusion_model_state_dict"].items():
            fusion_state_dict[k[7:]] = v
        fusion = Fusion(77, 512) 
        fusion.load_state_dict(fusion_state_dict)
        fusion.eval()
        fusion = torch.nn.DataParallel(fusion).to(device)
    else:
        net.load_state_dict(ckpt["state_dict"])
        fusion = None
    
    net.eval()
    net = net.to(device)

    
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    test_info = json.load(open(args.test_info)) if args.test_info != "" else None
    subset = None if args.subset == "" else list(json.load(open(args.subset)).keys())

    test_dataset = DATASETS(args.test_root, args.frames, transform_test, True, args.freq, subset=subset, info=test_info)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=16,num_workers=16,shuffle=False,pin_memory=False,drop_last=True)

    if subset is None:
        classes, num_text_aug, text_dict = text_prompt(test_dataset.class_names)
    else:
        classes, num_text_aug, text_dict = text_prompt(sorted(subset))

    prec1, prec5 = validate(0, test_loader, classes, num_text_aug, device, net, fusion_model=fusion)

    print(prec1, prec5)