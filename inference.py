import warnings
warnings.filterwarnings("ignore")

import os

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

            logit_scale = model.module.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
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
    parser.add_argument("--ckpt_path", type=str, default="/home/elv-zhounan/ActionClip_exp/weights/K400_pretrained/vit-b-16-32f.pt")
    parser.add_argument("--classes_name", type=str, default="./classes_name.json")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--test_file", type=str, default="./test/test.mp4")
    parser.add_argument("--inf_as_one", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--subset", type=str, default="") #/home/elv-zhounan/ActionClip_exp/cfg/exp/k400/subset50.json
    parser.add_argument("--data_root", type=str, default="/pool0/ml/elv-zhounan/action/kinetics/k400/images/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    FRAMES = args.frames
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
    logit_scale = net.logit_scale.exp()
    logger.info(f"logit_scale: {logit_scale}")
    if args.val:
        net = torch.nn.DataParallel(net).to(device)
        subset = None if args.subset == "" else list(json.load(open(args.subset)).keys())
        if subset is None:
            classes, num_text_aug, text_dict = text_prompt(json.load(open(args.classes_name)))
        else:
            classes, num_text_aug, text_dict = text_prompt(sorted(subset))
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_info = json.load(open("./cfg/data_k400/test_info.json"))
        test_dataset = DATASETS(args.data_root, FRAMES, transform_test, True, 2, subset=subset, info=test_info)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=16,num_workers=16,shuffle=False,pin_memory=False,drop_last=True)
        prec1, prec5 = validate(0, test_loader, classes, num_text_aug, device, net, fusion_model=fusion)
    else:
        net = net.to(device)
        classes_names = json.load(open(args.classes_name))
        if isinstance(classes_names, dict):
            classes_names = list(classes_names.keys())
        classes, num_text_aug, text_dict = text_prompt(classes_names, train=False)
        with torch.no_grad():
            text_features = net.encode_text(classes.to(device))
            print(text_features.shape)
            inf_as_one = args.inf_as_one
            video_path = args.test_file
            start = time.time()
            frames = get_frames(video_path)
            get_frames_timestamp = time.time()
            print("# of frames: ", len(frames), f"  cost {get_frames_timestamp-start} sec")
            
            if inf_as_one:
                interval = len(frames) // FRAMES
                frames = frames[::interval][:FRAMES]
                frames = tranform(frames, net.visual.input_resolution)
                print("stack images input shape: ", frames.shape)
                image_features = net.encode_image(frames.view(-1, 3, 224, 224).to(device)).view(1, FRAMES, -1)
                get_image_features_timestamp = time.time()
                print("encoded image features shape: ", image_features.shape, f"  cost {get_image_features_timestamp - get_frames_timestamp} sec")

                if fusion is not None:
                    image_features = fusion(image_features)
                else:
                    image_features = torch.mean(image_features, dim=1, keepdim=False)
                # compute similarity
                image_features = F.normalize(image_features, dim=1).cpu()
                text_features = F.normalize(text_features, dim=1).cpu()

                logits = logit_scale * image_features @ text_features.T
                pred = torch.softmax(logits, dim=1)
                get_sim_mat_timestamp = time.time()
                print("similarity met shape is: ", pred.shape, f"  cost {get_sim_mat_timestamp - get_image_features_timestamp} sec")
                sims, topK = torch.topk(pred, k=5, dim=1)
                print("topK indices: ", topK)
                for c, s in zip(topK.flatten().tolist(), sims.flatten().tolist()):
                    print(s, classes_names[c%len(classes_names)])

                print(f"cost {time.time() - start} sec in total")
            else:
                while len(frames) > FRAMES:
                    clip_frames = frames[:FRAMES]
                    frames = frames[FRAMES:]
                    clip_frames = tranform(clip_frames, net.visual.input_resolution)
                    image_features = net.encode_image(clip_frames.view(-1, 3, 224, 224).to(device)).view(1, FRAMES, -1)
                    if fusion is not None:
                        image_features = fusion(image_features)
                    else:
                        image_features = torch.mean(image_features, dim=1, keepdim=False)
                    # compute similarity
                    image_features = F.normalize(image_features, dim=1).cpu()
                    text_features = F.normalize(text_features, dim=1).cpu()
                    logits = logit_scale * image_features @ text_features.T
                    pred = torch.softmax(logits, dim=1)
                    sims, topK = torch.topk(pred, k=3, dim=1)
                    for c, s in zip(topK.flatten().tolist(), sims.flatten().tolist()):
                        print(s, classes_names[c%len(classes_names)])
                    print() 