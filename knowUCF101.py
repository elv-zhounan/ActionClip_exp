import cv2
import datetime
import os
import tqdm
import matplotlib.pyplot as plt


def getInfo(path):
    data = cv2.VideoCapture(path)
    
    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)
    
    # calculate duration of the video
    seconds = round(frames / fps)
    # video_time = datetime.timedelta(seconds=seconds)
    # print(f"duration in seconds: {seconds}")

    return frames, fps, seconds




if __name__ == "__main__":
    root = "/pool0/ml/elv-zhounan/ucf101/UCF101"

    filenames = os.listdir(root)

    print("Num of videos: ", len(filenames))
    # demo = filenames[0]
    # frames, fps, seconds = getInfo(os.path.join(root, demo))

    frames = []

    for file in tqdm.tqdm(filenames):
        frame, fps, _ = getInfo(os.path.join(root, file))
        frames.append(frame)


    plt.figure()
    plt.hist(frames)
    plt.savefig("UCF101_frames.jpg")

    print(min(frames))
    print(max(frames))
    print(sum(frames) / len(frames))

    
    # I would like to train a model using 16 frames, random sampling from all the key frames
    29.0
    1776.0
    186.50142642642643
    # but if using ffmpeg, to extract the key frames, I don't know if all of them will still have enough frames left