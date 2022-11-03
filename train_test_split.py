# including split train and test, extracting frames from videos and saving them in a folder with the same name as the video

import cv2
import os
import random
import tqdm
import json
import shutil

SPLIT = 0.8
IMAGE_ROOT = "/pool0/ml/elv-zhounan/ucf101/images"

def get_frames(vid_path):
    video = cv2.VideoCapture(vid_path)
    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(fcount):
        ret, frame = video.read()
        assert ret
        frames.append(frame)
    return frames

if __name__ == "__main__":
    root = "/pool0/ml/elv-zhounan/ucf101/UCF101"
    filenames = os.listdir(root)

    # frames = get_frames(os.path.join(root, filenames[0]))
    # print(len(frames))

    dic = {}
    for file in filenames:
        c = file.split(".")[0].split("_")[1]
        if c not in dic:
            dic[c] = []
        dic[c].append(file)

    
    print(len(dic))
    print([(x, len(dic[x])) for x in dic])

    dic_labels = {k:i for i, k in enumerate(list(dic.keys()))}
    json.dump(dic_labels, open("./cfg/ucf101_labels.json", "w"))

    for c in dic:
        files = dic[c]
        random.shuffle(files)
        train_l = int(len(files) * SPLIT)
        train_files = files[:train_l]
        test_files = files[train_l:]

        for file in tqdm.tqdm(train_files):
            train_dir = os.path.join(IMAGE_ROOT, "train", file.split(".")[0])
            os.mkdir(train_dir)
            try:
                frames = get_frames(os.path.join(root, file))
            except:
                shutil.rmtree(train_dir)
                continue
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(train_dir, f"{i}.jpg"), frame)
        
        for file in tqdm.tqdm(test_files):
            test_dir = os.path.join(IMAGE_ROOT, "test", file.split(".")[0])
            os.mkdir(test_dir)
            try:
                frames = get_frames(os.path.join(root, file))
            except:
                shutil.rmtree(test_dir)
                continue
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(test_dir, f"{i}.jpg"), frame)
        

    