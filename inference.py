import warnings
warnings.filterwarnings("ignore")

import os

import json
from loguru import logger

import torch
import torch.nn.functional as F
import yaml
from model import CLIP
from fusion_vision import Fusion
from typing import List, Union
from tokenizer import SimpleTokenizer
from data import *
# from utils import get_video_frames, norm
import time
from tqdm import tqdm
from train_test_split import get_frames

def encode_image(model:CLIP, input:Union[List[torch.Tensor], torch.Tensor]):
    return model.encode_image(input)

def encode_text(model:CLIP, tokenizer:SimpleTokenizer, input:str):
    input = tokenize(tokenizer, input)
    return model.encode_text(input)

@torch.no_grad()
def validate(epoch, val_loader, classes, device, model, num_text_aug, fusion_model=None):
    model.eval()
    if fusion_model:
        fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():
        text_inputs = classes.to(device)
        bar = tqdm(val_loader)
        for _, (image, class_id) in enumerate(bar):
            b, t, c, h, w = image.shape
            class_id = class_id.to(device)
            image_features, text_features = model(image, text_inputs)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).mean(dim=1, keepdim=False).softmax(dim=-1)

            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)

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
    with torch.no_grad():
        FRAMES = 32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inf_as_one = False
        cfg = yaml.safe_load(open("/home/elv-zhounan/ActionClip_exp/cfg/model/ViT-B-16.yaml"))
        # ckpt = torch.load("/home/elv-zhounan/ActionClip_exp/exp/ViT-B-16/K400/16_frames/best_model.pt")
        ckpt = torch.load("/home/elv-zhounan/ActionClip_exp/weights/K400_pretrained/vit-b-16-32f.pt")
        video_path = "/home/elv-zhounan/ActionClip_exp/test/test.mp4"
        classes_names = sorted(os.listdir("/pool0/ml/elv-zhounan/action/kinetics/k400/train"))

        if "model_state_dict" in ckpt:

            net = CLIP(**cfg)
            net.load_state_dict(ckpt["model_state_dict"])
            net.eval()
            logit_scale = net.logit_scale.exp()
            net = net.to(device)
            logger.info(f"logit_scale: {logit_scale}")
            fusion_state_dict = {}
            for k, v in ckpt["fusion_model_state_dict"].items():
                fusion_state_dict[k[7:]] = v
            fusion = Fusion(77, 512) 
            fusion.load_state_dict(fusion_state_dict)
            fusion.eval()
            fusion = fusion.to(device)
        else:
            net = CLIP(**cfg)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            logit_scale = net.logit_scale.exp()
            logger.info(f"logit_scale: {logit_scale}")
            net = net.to(device)

            fusion = None

        classes_encodes, num_text_aug, text_dict = text_prompt(classes_names, train=False)
        text_features = net.encode_text(classes_encodes.to(device))
        print(text_features.shape)
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
                sims, topK = torch.topk(pred, k=1, dim=1)
                for c, s in zip(topK.flatten().tolist(), sims.flatten().tolist()):
                    print(s, classes_names[c%len(classes_names)])