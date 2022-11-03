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
    def _get_indices(self, record_num_frames):
        
        if self.num_segments == 1:
            return np.array([record_num_frames //2], dtype=np.int)
        
        if record_num_frames <= self.num_segments:
            return np.mod(np.arange(self.num_segments), record_num_frames)

        offset = (record_num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record_num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=int)

    def _get_frames(self, video_path, indices):
        images = []
        for i in indices:
            images.append(self.transform(Image.open(os.path.join(video_path, f"{i}.jpg")).convert('RGB')))
        return torch.stack(images)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        record_num_frames = self.frames_cnt[index]
        segment_indices = self._get_indices(record_num_frames)
        images = self._get_frames(video_path, segment_indices)
        label = self.labels[video_path.split("/")[-1].split("_")[1]]
        return images, label

    def __len__(self):
        return len(self.video_list)


if __name__ == "__main__":
    t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    dataset = UCF101_DATASETS("/pool0/ml/elv-zhounan/ucf101/images", 16, 1, t, False)
    print(len(dataset))
    for i in tqdm.tqdm(range(len(dataset))):
        images, label = dataset[i]
        assert images.shape == torch.Size([16, 3, 224, 224]), f"{images.shape}"
