from torch import nn
import torch
import cv2
import os
from loguru import logger
import random

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def norm(t, eps=1e-8):
    return t / (torch.linalg.norm(t, dim=1, keepdim=True) + eps)

def get_video_frames(video_path, freq):
    if not os.path.exists(video_path):
        logger.info('Video file not exists, check path %s' % (video_path))
        raise SystemExit

    cap = cv2.VideoCapture(video_path)
    frames_per_sec = cap.get(cv2.CAP_PROP_FPS)

    '''
        key_frames sampling with 'freq' frames sampled in between
        e.g.
            key frames:{0, 23}, freq:1 --> frames sampled: {0, 12, 24}
            param: video_path, freq
            ret: f_num
    '''
    logger.info('Getting i frames')
    tmp_path = f'{os.path.basename(video_path)}_iframe.txt'
    os.system(
        f'ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv \
        {video_path} | grep frame | grep -n I | cut -d : -f 1 > {tmp_path} 2> /dev/null'
    )

    def _get_fnum(freq, fps_min=4, fps_max=8, mezz_dur=30.03):
        with open(tmp_path, 'r') as f:
            f_num = [int(n.strip())-1 for n in f.readlines()]
        n = len(f_num)
        if n < mezz_dur*fps_min:
            freq = int((mezz_dur*fps_min-n)//(n-1) + 1 + freq)
            logger.info(f"Sampling frequency modified to {freq}")
        else:
            random.seed(1)
            f_num = sorted(random.sample(f_num, min(n, int(mezz_dur*fps_max))))
        tmp = []
        for i in range(1, len(f_num)):
            tmp.extend([f_num[i-1]+int((f_num[i]-f_num[i-1])*(j+1)/(freq+1))
                       for j in range(freq)])
        return set(tmp).union(set(f_num))

    f_num = _get_fnum(freq)
    logger.info(f'Frame IDs to tag: {sorted(f_num)}')
    os.remove(tmp_path)

    n_frame = 0
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            '''uniform frame sampling, e.g. 0,2,4,6,etc.
            if n_frame%freq==0:
                images.append(frame)
            '''
            if n_frame in f_num:
                images.append(frame)
        else:
            break
        if n_frame % 1000 == 0:
            logger.info(f'Capturing frame # {n_frame}')
        n_frame += 1

    cap.release()
    frame_num = len(images)
    assert frame_num == len(f_num)
    logger.info("Total # of frames %s" % frame_num)
    return frames_per_sec, f_num, images

if __name__ == "__main__":
    frames_per_sec, f_num, images = get_video_frames("/Users/zhounanli/proj-eve/iq__2uCxJK4dw1BC7ZeKFCY8b3dg1ngZ/video/iq__2uCxJK4dw1BC7ZeKFCY8b3dg1ngZ.00006.mp4", 0)
    print(images[0].shape)
    print(type(images[0]))