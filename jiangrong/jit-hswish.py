#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) * 0.16666666666666666
        # return x * F.relu6(x + 3., inplace=self.inplace) / 6.0

# %%
net = Hswish()
_ = net(torch.Tensor(1, 3, 224, 224))

# %%
trace_model = torch.jit.trace(net, (torch.Tensor(1, 3, 224, 224), ))
trace_model.save('./assets/hswish.jit')
