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
from data import *
# from utils import get_video_frames, norm
from tqdm import tqdm
from train_test_split import get_frames
from inference import encode_image, encode_text

if __name__ == "__main__":
    with torch.no_grad():
        FRAMES = 32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg = yaml.safe_load(open("/home/elv-zhounan/ActionClip_exp/cfg/model/ViT-B-16.yaml"))
        # ckpt = torch.load("/home/elv-zhounan/ActionClip_exp/exp/ViT-B-16/K400/16_frames/best_model.pt")
        ckpt = torch.load("/home/elv-zhounan/ActionClip_exp/weights/K400_pretrained/vit-b-16-32f.pt")
        classes_names = json.load(open("classes_name.json"))
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

        total = 0
        total_tags = 0
        for root, _, files in os.walk("/pool0/ml/elv-zhounan/mgm_clips/mgm_clip_shots"):
            for file in files:
                if not file.endswith("mp4"):
                    continue
                video_path = os.path.join(root, file)
                res_path = os.path.join(root, file.replace("mp4", "json"))
                frames = get_frames(video_path)
                # we would like to utilize all of the frames given in the frames
                start = 0
                res = []
                while start + FRAMES < len(frames):
                    clip_frames = frames[start:start+FRAMES]
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
                    values, indices = torch.max(pred, dim=1)
                    s = values.flatten().tolist()[0]
                    c = indices.flatten().tolist()[0]
                    # print(s, classes_names[c%len(classes_names)])
                    if s > 0.5:
                        res.append({"start_frame": start, "end_frame": start+FRAMES-1, "label": classes_names[c%len(classes_names)], "score": round(s, 3)})
                        total_tags += 1
                    start += FRAMES
                if start > len(frames):
                    # we do not want to waste the last few frames
                    start = len(frames) - FRAMES
                    end = len(frames)
                    clip_frames = frames[start:end]
                    clip_frames = tranform(clip_frames, net.visual.input_resolution)
                    image_features = net.encode_image(clip_frames.view(-1, 3, 224, 224).to(device)).view(1, FRAMES, -1)
                    if fusion is not None:
                        image_features = fusion(image_features)
                    else:
                        image_features = torch.mean(image_features, dim=1, keepdim=False)
                    image_features = F.normalize(image_features, dim=1).cpu()
                    text_features = F.normalize(text_features, dim=1).cpu()
                    logits = logit_scale * image_features @ text_features.T
                    pred = torch.softmax(logits, dim=1)
                    values, indices = torch.max(pred, dim=1)
                    s = values.flatten().tolist()[0]
                    c = indices.flatten().tolist()[0]
                    # print(s, classes_names[c%len(classes_names)])
                    if s > 0.5:
                        res.append({"start_frame": start, "end_frame": end-1, "label": classes_names[c%len(classes_names)], "score": round(s, 3)})
                        total_tags += 1
                json.dump(res, open(res_path, "w"))
                total += 1
                logger.info(f"just done {root}, {file} total completed video {total} total tags we got {total_tags}")