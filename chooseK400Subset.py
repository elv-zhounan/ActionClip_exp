import os
import json
import tqdm

train_root = "/pool0/ml/elv-zhounan/action/kinetics/k400/images/train"
subset_num = 50

classes = os.listdir(train_root)

cnt = {}

for cls in tqdm.tqdm(classes):
    cls_root = os.path.join(train_root, cls)
    c = len(os.listdir(cls_root))
    cnt[cls] = c

_sorted = sorted(cnt.items(), key=lambda x:x[1], reverse=True)

print(_sorted)

subset = {k:v for (k,  v) in _sorted[:subset_num]}

json.dump(subset, open(f"/home/elv-zhounan/ActionClip_exp/cfg/exp/k400/subset{subset_num}.json", "w"))