import torch
import yaml
from model import CLIP
from fusion_vision import Fusion
from typing import List, Union
from tokenizer import SimpleTokenizer
from data import *
from utils import get_video_frames, norm
import random


def load():
    cfg = yaml.safe_load(open("./cfg/ViT-B-16.yaml"))

    net = CLIP(**cfg)
    net.load_state_dict(torch.load(f"/Users/zhounanli/ActionClipWeights/ViT-B-16-state-dict.pt"))
    net.eval()

    fusion = Fusion(77, 512) 
    fusion.load_state_dict(torch.load("/Users/zhounanli/ActionClipWeights/fision-model-state-dict-16f.pt"))
    fusion.eval()

    return net, fusion

def encode_image(model:CLIP, input:Union[List[torch.Tensor], torch.Tensor]):
    return model.encode_image(input)

def encode_text(model:CLIP, tokenizer:SimpleTokenizer, input:str):
    input = tokenize(tokenizer, input)
    print(input.shape)
    return model.encode_text(input)


if __name__ == "__main__":
    with torch.no_grad():
        net, fusion = load()
        
        text = "this is a video about rock climbing indoor"
        video_path = "/Users/zhounanli/Desktop/ML_codes/ActionCLIP_inference_only/v_RockClimbingIndoor_g11_c02.avi"

        text_feature = encode_text(net, SimpleTokenizer(), text)
        print(text_feature.shape)

        *_, frames = get_video_frames(video_path=video_path, freq=0)
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
        image_feature = fusion(image_feature)
        #  use meanP for now
        # image_feature = image_feature.mean(dim=1, keepdim=False)
        print("Merged images feature shape: ", image_feature.shape)

        # compute similarity
        image_feature = norm(image_feature)
        text_feature = norm(text_feature)
        cos_sim = torch.mm(image_feature, torch.transpose(text_feature, 0, 1))
        print(cos_sim)

