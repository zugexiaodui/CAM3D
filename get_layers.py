import torch
import sys
f_name=sys.argv[1]
ckpt = torch.load(f_name,map_location='cpu')
for name in ckpt['state_dict']:
    print(name,ckpt['state_dict'][name].shape)
