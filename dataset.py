import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
import json
from torchvision import transforms
import tqdm

from loguru import logger

class DATASETS(data.Dataset):
    def __init__(self, root, num_segments=16, transform=None, test_mode=False, freq=2, subset=None, info=None):
        assert transform is not None
        self.num_segments = num_segments
        self.transform = transform
        self.test_mode = test_mode
        self.freq = freq
        root = os.path.join(root, "train" if not test_mode else "test")
        if subset:
            self.class_names = sorted(subset)
        else:
            self.class_names = sorted(os.listdir(root))
        logger.info("loading video list")
        self.videos = []
        self.frame_cnt = []
        self.labels = []

        bar = tqdm.tqdm(self.class_names)
        for i, c in enumerate(bar):
            c_root = os.path.join(root, c)
            c_videos = sorted(os.listdir(c_root))
            bar.set_postfix_str(f"{len(c_videos)} videos for cls {c}")
            for v in c_videos:
                v = os.path.join(c_root, v)
                if info is not None:
                    if v in info:
                        cnt = info[v]
                        self.videos.append(v)
                        self.labels.append(i)
                        self.frame_cnt.append(cnt)
                else:
                    cnt = len(os.listdir(v))
                    if  cnt > 0:
                        self.videos.append(v)
                        self.labels.append(i)
                        self.frame_cnt.append(cnt)
        

    def _get_indices(self, video_frame_num):
        
        if self.num_segments == 1:
            return np.array([video_frame_num //2], dtype=np.int)
        
        if video_frame_num <= self.num_segments:
            return np.arange(self.num_segments) % video_frame_num

        if self.test_mode:
            distance = video_frame_num // self.num_segments
            return np.array([j * distance for j in range(self.num_segments)], dtype=int)
        else:
            # random select a random start, and them choose the continuous num_segments frames 
            offset = random.randint(0, video_frame_num-self.num_segments)
            return np.array([offset+j for j in range(self.num_segments)], dtype=int)

    def _get_frames(self, video_path, indices):
        images = []
        for i in indices:
            images.append(self.transform(Image.open(os.path.join(video_path, f"{i*self.freq}.jpg")).convert('RGB')))
        return torch.stack(images)

    def __getitem__(self, index):
        video_path = self.videos[index]
        video_frame_num = self.frame_cnt[index]
        frame_indices = self._get_indices(video_frame_num)
        images = self._get_frames(video_path, frame_indices)
        label = self.labels[index]
        return images, label

    def __len__(self):
        return len(self.videos)


if __name__ == "__main__":
    t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    dataset = DATASETS("/pool0/ml/elv-zhounan/action/kinetics/k400/images", 16, t, False)
    logger.info(len(dataset))
