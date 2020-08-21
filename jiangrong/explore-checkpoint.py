#%%
import os, sys
import torch

# %%
init = torch.load("/workspace/once-for-all/exp/normal2kernel/checkpoint/checkpoint.pth.tar", map_location='cpu')['state_dict']
# %%
init.keys()
# %%
