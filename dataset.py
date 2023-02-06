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

class UCF101_DATASETS(data.Dataset):
    def __init__(self, root,
                 num_segments=16, 
                 new_length=1,
                 transform=None,
                 test_mode=False):

        self.num_segments = num_segments
        self.seg_length = new_length
        self.transform = transform
        self.test_mode = test_mode

        root = os.path.join(root, "train" if not test_mode else "test")
        print("loading video list")
        self.video_list = [os.path.join(root, f) for f in os.listdir(root)]
        print("counting video frames")
        self.frames_cnt = [len(os.listdir(f)) for f in tqdm.tqdm(self.video_list)]
        self.total_length = len(self.video_list)
        self.labels = json.load(open("./cfg/ucf101_labels.json"))


    # TODO
    def _get_random_indices(self, record):
        raise NotImplementedError()

    # if using this, each video will always give out the same set of frames
    # lets use this first, simlify the original version
    # TOREAD
    def _get_indices(self, video_frame_num):
        
        if self.num_segments == 1:
            return np.array([video_frame_num //2], dtype=np.int)
        
        if video_frame_num <= self.num_segments:
            return np.mod(np.arange(self.num_segments), video_frame_num)

        offset = (video_frame_num / self.num_segments - self.seg_length) / 2.0
        return np.array([i * video_frame_num / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=int)

    def _get_frames(self, video_path, indices):
        images = []
        for i in indices:
            images.append(self.transform(Image.open(os.path.join(video_path, f"{i}.jpg")).convert('RGB')))
        return torch.stack(images)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        video_frame_num = self.frames_cnt[index]
        segment_indices = self._get_indices(video_frame_num)
        images = self._get_frames(video_path, segment_indices)
        label = self.labels[video_path.split("/")[-1].split("_")[1]]
        return images, label

    def __len__(self):
        return len(self.video_list)

class K400_DATASETS(data.Dataset):
    def __init__(self, root, num_segments=16, transform=None, test_mode=False, freq=2):
        assert transform is not None
        self.num_segments = num_segments
        self.transform = transform
        self.test_mode = test_mode
        self.freq = freq
        print("loading video list")
        root = os.path.join(root, "train" if not test_mode else "test")
        self.videos = []
        self.frame_cnt = []
        self.labels = []
        self.class_names = sorted(os.listdir(root))
        bar = tqdm.tqdm(self.class_names)
        for i, c in enumerate(bar):
            c_root = os.path.join(root, c)
            c_videos = sorted(os.listdir(c_root))
            bar.set_postfix_str(f"{len(c_videos)} videos for cls {c}")
            for v in c_videos:
                cnt = len(os.listdir(os.path.join(c_root, v)))
                if  cnt > 0:
                    self.videos.append(os.path.join(c_root, v))
                    self.labels.append(i)
                    self.frame_cnt.append(cnt)
        
    def _get_indices(self, video_frame_num):
        
        if self.num_segments == 1:
            return np.array([video_frame_num //2], dtype=np.int)
        
        if video_frame_num <= self.num_segments:
            return np.arange(self.num_segments) % video_frame_num

        if self.test_mode:
            # a test set
            # I would like it to return the same set of frames
            # so I would evenly choose num_segments across the whole range 
            distance = video_frame_num // self.num_segments
            return np.array([j * distance for j in range(self.num_segments)], dtype=int)
        else:
            # training mode, similar idea to test mode
            # but the start will not be the first frame, will be random, reasonable one
            offset = random.randint(0, video_frame_num-self.num_segments)
            distance = (video_frame_num - offset) // self.num_segments
            return np.array([j * distance for j in range(self.num_segments)], dtype=int)

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

    dataset = K400_DATASETS("/pool0/ml/elv-zhounan/action/kinetics/k400/images", 16, t, False)
    print(len(dataset))
    
