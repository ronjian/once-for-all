#%%
import numpy as np
import torch
import torch.nn as nn


class testOP(nn.Module):
    def __init__(self):
        super(testOP, self).__init__()

    def forward(self, x):
        x = nn.AvgPool2d((224, 224))(x)
        return x

# %%
net = testOP()
_ = net(torch.Tensor(1, 3, 224, 224))

# %%
trace_model = torch.jit.trace(net, (torch.Tensor(1, 3, 224, 224), ))
trace_model.save('./assets/avgpool.jit')

# %%
