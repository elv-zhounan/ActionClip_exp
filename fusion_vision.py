import torch
from torch import nn 
from collections import OrderedDict

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class Fusion(nn.Module):
    def __init__(self, T, emb_dim):
        super().__init__()
        self.T = T

        self.embed_dim = emb_dim
        self.heads = emb_dim // 64

        self.frame_position_embeddings = nn.Embedding(T, emb_dim)
        self.transformer = TemporalTransformer(width=emb_dim, layers=6, heads=self.heads)

    def forward(self, x):
        b, t, c = x.size()
        assert c == self.embed_dim

        x = x.contiguous()
        x_original = x
        position_ids = torch.arange(t, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(b, -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + frame_position_embeddings

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original
        return x.mean(dim=1, keepdim=False)


if __name__ == "__main__":
    # i'm really not sure why they set the context_length to be 77, but according to the weight's shape, it is 
    # I assume this fusion model will work for both text and vision
    # so for compatible with a text seq, the pos emb shape 0 is set to 77
    fusion = Fusion(77, 512) 
    pretrained_state_dict = torch.load("/Users/zhounanli/ActionClipWeights/fision-model-state-dict-16f.pt")
    model_state_dict = fusion.state_dict()
    for k in model_state_dict:
        if model_state_dict[k].shape != pretrained_state_dict[k].shape:
            print(k, model_state_dict[k].shape, pretrained_state_dict[k].shape)
    fusion.load_state_dict(torch.load("/Users/zhounanli/ActionClipWeights/fusion-model-state-dict-16f.pt"))
