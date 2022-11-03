import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import warnings
warnings.filterwarnings("ignore")

import torch
import yaml
from model import CLIP
from fusion_vision import Fusion
from typing import List, Union
from tokenizer import SimpleTokenizer
from data import *
from utils import get_video_frames, norm
import random
from tqdm import tqdm
from train_test_split import get_frames

def load():
    cfg = yaml.safe_load(open("./cfg/ViT-B-16.yaml"))

    net = CLIP(**cfg)
    net.load_state_dict(torch.load(f"./exp/clip_ucf/ViT-B-16/ucf101/last_model.pt")["state_dict"])
    net.eval()

    fusion = Fusion(77, 512) 
    fusion.load_state_dict(torch.load("./weights/fusion-model-state-dict-16f.pt"))
    fusion.eval()

    return net, fusion

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
                if class_id[i] == indices_1[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1

            bar.set_postfix_str(f" up to now in avg:  Top1: {round(float(corr_1) / num, 3)}, Top5: {round(float(corr_5) / num, 3)}")


    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    print('Epoch: [{}]: Top1: {}, Top5: {}'.format(epoch, top1, top5))
    return top1, top5

if __name__ == "__main__":
    with torch.no_grad():
        net, fusion = load()
        
        text = "this is a video about rock climbing indoor"

        video_path = "/pool0/ml/elv-zhounan/ucf101/UCF101/v_RockClimbingIndoor_g11_c02.avi"

        text_feature = encode_text(net, SimpleTokenizer(), text)
        print(text_feature.shape)

        # *_, frames = get_video_frames(video_path=video_path, freq=4)
        frames = get_frames(video_path)
        # print(len(frames))
        # interval = len(frames) // 16
        # frames = frames[::interval][:16]
        indexes = sorted(random.sample(list(range(len(frames))), k=16))
        frames = [frames[i] for i in indexes]
        frames = tranform(frames, net.visual.input_resolution)
        print(frames.shape)
        b, t, c, h, w = frames.size()
        image_input = frames.view(-1, c, h, w)
        image_feature = encode_image(net, image_input).view(b, t, -1)
        print("CLIP image encoder output shape: ", image_feature.shape)
        
        # using fusion model to do fuse
        # image_feature = fusion(image_feature)
        #  use meanP for now
        image_feature = image_feature.mean(dim=1, keepdim=False)
        print("Merged images feature shape: ", image_feature.shape)

        # compute similarity
        image_feature = norm(image_feature)
        text_feature = norm(text_feature)
        cos_sim = torch.mm(image_feature, torch.transpose(text_feature, 0, 1))
        print(cos_sim)


        import json
        classes = list(json.load(open("/home/elv-zhounan/ActionCLIP_inference_only/cfg/ucf101_labels.json")).keys())
        texts = [f"this is a video about {c}" for c in classes]

        text_features = torch.concat([encode_text(net, SimpleTokenizer(), text) for text in texts], dim=0)

        sim = image_feature @ text_features.T

        sim = torch.softmax(sim, dim=-1)

        print(sim)

        v, i = torch.max(sim, dim=1)
        print(i, v)
        print(classes[i[0]])