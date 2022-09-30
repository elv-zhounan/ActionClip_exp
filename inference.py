import torch
import yaml
from model import CLIP
from utils import convert_weights
from typing import List, Union
from tokenizer import SimpleTokenizer
from data import *
from utils import get_video_frames, norm
model_name = "ViT-B-16"



def load(cfg=f"./cfg/{model_name}.yaml"):
    if type(cfg) == str:
        cfg = yaml.safe_load(open(cfg))

    assert type(cfg) == dict

    net = CLIP(**cfg)

    net.load_state_dict(torch.load(f"/Users/zhounanli/ClipWeights/{model_name}-state-dict.pt"))

    return net

def encode_image(model:CLIP, input:Union[List[torch.Tensor], torch.Tensor]):
    return model.encode_image(input)

def encode_text(model:CLIP, tokenizer:SimpleTokenizer, input:str):
    input = tokenize(tokenizer, input)
    return model.encode_text(input)


if __name__ == "__main__":
    # retrive_state_dict()
    # get_cfg()
    net = load()
    # convert_weights(net)
    
    text = "men talking and a train is driving through"
    video_path = "/Users/zhounanli/proj-eve/iq__2uCxJK4dw1BC7ZeKFCY8b3dg1ngZ/video/iq__2uCxJK4dw1BC7ZeKFCY8b3dg1ngZ.00006.mp4"

    text_feature = encode_text(net, SimpleTokenizer(), text)
    print(text_feature.shape)
    *_, frames = get_video_frames(video_path=video_path, freq=0)
    frames = frames[::3]
    frames = tranform(frames, net.visual.input_resolution)
    print(frames.shape)
    b, t, c, h, w = frames.size()
    image_input = frames.view(-1, c, h, w)
    image_feature = encode_image(net, image_input).view(b, t, -1)
    print(image_feature.shape)
    
    #  use meanP for now
    image_feature = image_feature.mean(dim=1, keepdim=False)
    print(image_feature.shape)

    # compute similarity
    image_feature = norm(image_feature)
    text_feature = norm(text_feature)
    cos_sim = torch.mm(image_feature, torch.transpose(text_feature, 0, 1))
    print(cos_sim)

