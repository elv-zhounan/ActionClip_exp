import torch

ckpt = torch.load("/Users/zhounanli/ActionClipWeights/vit-b-16-16f.pt", map_location="cpu")
state_dict = ckpt["model_state_dict"]
fusion_model_state_dict = ckpt["fusion_model_state_dict"]

# print(state_dict.keys())

# torch.save(state_dict, "/Users/zhounanli/ActionClipWeights/vit-b-16-16f-state-dict.pt")



print(fusion_model_state_dict.keys())
print(set([name.split(".")[1] for name in fusion_model_state_dict.keys()]))
print(set([name.split(".")[2] for name in fusion_model_state_dict.keys()]))
# dic = {}
# for k in fusion_model_state_dict:
#     dic[k[7:]] = fusion_model_state_dict[k]
# torch.save(dic, "/Users/zhounanli/ActionClipWeights/fision-model-state-dict-16f.pt")


embed_dim = state_dict["text_projection"].shape[1]
context_length = state_dict["positional_embedding"].shape[0]
vocab_size = state_dict["token_embedding.weight"].shape[0]
transformer_width = state_dict["ln_final.weight"].shape[0]
transformer_heads = transformer_width // 64

print(embed_dim, context_length, vocab_size, transformer_width, transformer_heads)