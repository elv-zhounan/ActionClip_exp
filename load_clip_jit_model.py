import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
import torch
from model import CLIP

if __name__ == "__main__":
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    with open("/home/elv-zhounan/ActionClip_exp/weights/ViT-B-16.pt", "rb") as f:
        model = torch.jit.load(f, map_location=device).eval()
    
    # print(model)
    # print(model.state_dict().keys())
    
    cfg = yaml.safe_load(open("/home/elv-zhounan/ActionClip_exp/cfg/model/ViT-B-16.yaml"))
    a_clip = CLIP(**cfg)

    a_clip_state_dict = a_clip.state_dict()
    clip_state_dict = model.state_dict()

    save = True
    for k in a_clip_state_dict:
        if k not in clip_state_dict:
            print(k, "not in pretrained clip model state dict")
            save = False
            continue
        if a_clip_state_dict[k].shape != clip_state_dict[k].shape:
            print(k, "shape not matched")
            save = False
            continue
    
    
    _del = []
    for k in clip_state_dict:
        if k not in a_clip_state_dict:
            print(k, clip_state_dict[k])
            assert clip_state_dict[k] == cfg[k]
            _del.append(k)
    
    for k in _del:
        del clip_state_dict[k]
    

    if save:
        torch.save(clip_state_dict, "/home/elv-zhounan/ActionClip_exp/weights/vit-b-16-state-dict.pt")
    

    # test again
    a_clip.load_state_dict(torch.load("/home/elv-zhounan/ActionClip_exp/weights/vit-b-16-state-dict.pt"))