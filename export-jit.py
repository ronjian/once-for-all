#%%
from ofa.model_zoo import ofa_specialized
import torch

# %%
# for netid in ['flops@595M_top1@80.0_finetune@75',
#                 'pixel1_lat@143ms_top1@80.1_finetune@75'
#                 ,'note10_lat@64ms_top1@80.2_finetune@75']:
for netid in ['cpu_lat@17ms_top1@75.7_finetune@25']:
    net, image_size = ofa_specialized(netid
                                    , pretrained=True)
    
    tgtid = netid.replace('@', '-').replace('-','_') + '.jit'
    print(netid, image_size, tgtid)
    net.eval()
    _ = net(torch.Tensor(1,3,image_size,image_size))
    trace_model = torch.jit.trace(net, (torch.Tensor(1,3,image_size,image_size), ))
    trace_model.save(tgtid)

