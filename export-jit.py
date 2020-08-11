#%%
from ofa.model_zoo import ofa_specialized
import torch

# %%
netid = 'flops@595M_top1@80.0_finetune@75'
net, image_size = ofa_specialized(netid
                                , pretrained=True)


# %%
net.eval()

# %%
_ = net(torch.Tensor(1,3,236,236))

# %%
trace_model = torch.jit.trace(net, (torch.Tensor(1,3,236,236), ))


# %%
trace_model.save(netid.replace('@', '-').replace('-','_') + '.jit')

# %%
