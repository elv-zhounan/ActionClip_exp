# test the one distilled on a subset Epoch: [0]: Top1: 91.24586092715232, Top5: 99.04801324503312
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3,4 python inference.py --ckpt_path ./exp/ViT-B-16/K400/32_frames_subset50/best_model.pt --val --subset ./cfg/exp/k400/subset50.json

# test the one trained on the whole dataset from CLIP pretrained weights
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3,4 python inference.py --ckpt_path ./exp/ViT-B-16/K400/16_frames_fp16_training_using_clip/best_model.pt --val
