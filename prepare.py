import re
from matplotlib.pyplot import get
import torch
import yaml
from model import CLIP
from utils import convert_weights
model_name = "ViT-B-16"
def retrive_state_dict():
    ckpt = torch.jit.load(f"/Users/zhounanli/ClipWeights/{model_name}.pt", map_location="cpu")
    # print(ckpt)
    state_dict = ckpt.state_dict()
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    torch.save(state_dict, f"/Users/zhounanli/ClipWeights/{model_name}-state-dict.pt")

def get_cfg():

    ckpt = torch.jit.load(f"/Users/zhounanli/ClipWeights/{model_name}.pt", map_location="cpu")
    state_dict = ckpt.state_dict()

    assert "visual.proj" in state_dict


    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
   

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    cfg = {
        "vision_width":vision_width,
        "vision_layers":vision_layers,
        "vision_patch_size":vision_patch_size,
        "image_resolution":image_resolution,
        "embed_dim":embed_dim,
        "context_length":context_length,
        "vocab_size":vocab_size,
        "transformer_width":transformer_width,
        "transformer_heads":transformer_heads,
        "transformer_layers":transformer_layers,
    }

    print(cfg)

    yaml.safe_dump(cfg, open(f"./cfg/{model_name}.yaml", "w"))

def load(cfg=f"./cfg/{model_name}.yaml"):
    if type(cfg) == str:
        cfg = yaml.safe_load(open(cfg))

    assert type(cfg) == dict

    net = CLIP(**cfg)

    net.load_state_dict(torch.load(f"/Users/zhounanli/ClipWeights/{model_name}-state-dict.pt"))

    return net

if __name__ == "__main__":
    # retrive_state_dict()
    # get_cfg()
    net = load()
    net = convert_weights(net)

    


