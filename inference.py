import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")

import json

import torch
import torch.nn.functional as F
import yaml
from model import CLIP
from fusion_vision import Fusion
from typing import List, Union
from tokenizer import SimpleTokenizer
from data import *
from utils import get_video_frames, norm
import time
from tqdm import tqdm
from train_test_split import get_frames

FRAMES = 16

def load():
    cfg = yaml.safe_load(open("./cfg/ViT-B-16.yaml"))

    net = CLIP(**cfg)
    net.load_state_dict(torch.load(f"./exp/clip_ucf/ViT-B-16/ucf101/last_model_without_attnmask.pt")["state_dict"])
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, fusion = load()
        net = net.to(device)

        classes_names = list(json.load(open("./cfg/ucf101_labels.json")).keys())
        classes_encodes, num_text_aug, text_dict = text_prompt(classes_names)
        text_features = net.encode_text(classes_encodes.to(device))
        print(text_features.shape)

        start = time.time()

        video_path = "/pool0/ml/elv-zhounan/action/ucf101/UCF101/v_TaiChi_g10_c02.avi"
        frames = get_frames(video_path)
        get_frames_timestamp = time.time()
        print("# of frames: ", len(frames), f"  cost {get_frames_timestamp-start} sec")
        
        interval = len(frames) // FRAMES
        frames = frames[::interval][:FRAMES]
        frames = tranform(frames, net.visual.input_resolution)
        print("stack images input shape: ", frames.shape)
        image_features = net.encode_image(frames.view(-1, 3, 224, 224).to(device)).view(1, FRAMES, -1)
        get_image_features_timestamp = time.time()
        print("encoded image features shape: ", image_features.shape, f"  cost {get_image_features_timestamp - get_frames_timestamp} sec")

        image_features = torch.mean(image_features, dim=1, keepdim=False)
        print("Using mean to aggregate the seq features, shape is: ", image_features.shape)
        # compute similarity
        image_features = F.normalize(image_features, dim=1).cpu()
        text_features = F.normalize(text_features, dim=1).cpu()
        cos_sim = image_features @ text_features.T
        get_sim_mat_timestamp = time.time()
        print("similarity met shape is: ", cos_sim.shape, f"  cost {get_sim_mat_timestamp - get_image_features_timestamp} sec")

        sims, topK = torch.topk(cos_sim, k=5, dim=1)
        print("topK indices: ", topK)
        for c, s in zip(topK.flatten().tolist(), sims.flatten().tolist()):
            print(s, classes_names[c % 101])

        print(f"cost {time.time() - start} sec in total")