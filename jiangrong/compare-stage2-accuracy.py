# '.torch/ofa_nets/ofa_D234_E6_K357'
#   - kernel 7, depth 4, top1=77.9
#   - kernel 5, depth 4, top1=77.7
#   - kernel 5, depth 3, top1=77.1
#   - kernel 5, depth 2, top1=75.1
#   - kernel 3, depth 2, top1=74.1

# '/workspace/once-for-all/exp/kernel2kernel_depth/phase2/checkpoint/checkpoint.pth.tar'
#   - kernel 7, depth 4, top1=76.3
#   - kernel 5, depth 4, top1=76.3
#   - kernel 5, depth 3, top1=76.2
#   - kernel 5, depth 2, top1=74.0
#   - kernel 3, depth 2, top1=73.2

# %%
import sys;sys.path.append('/workspace/once-for-all')
import json
import copy
import math
import random
import time
import numpy as np
import torch.nn as nn
import torch
import os;os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from ofa.elastic_nn.networks import OFAMobileNetV3
from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.imagenet_codebase.run_manager import ImagenetRunConfig
from ofa.imagenet_codebase.run_manager import RunManager

# %%
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
ofa_network = OFAMobileNetV3(dropout_rate=0
                    , width_mult_list=1.0
                    , ks_list=[3, 5, 7]
                    , expand_ratio_list=[6]
                    , depth_list=[2,3,4],)
ofa_network.to('cuda:0')
# init = torch.load('/workspace/once-for-all/exp/kernel2kernel_depth/phase2/checkpoint/checkpoint.pth.tar',
#                 map_location='cuda:0')['state_dict']
init = torch.load('.torch/ofa_nets/ofa_D234_E6_K357',
                map_location='cuda:0')['state_dict']
ofa_network.load_state_dict(init)
# %%
ofa_network.set_active_subnet(ks=[3] * 20, d=[2] * 5, e=[6] * 20)
subnet = ofa_network.get_active_subnet(preserve_weight=True)
print(subnet.module_str)

# %%
run_config = ImagenetRunConfig(test_batch_size=64, n_worker=4)
run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
run_config.data_provider.assign_active_img_size(224)
run_manager.reset_running_statistics(net=subnet)

#%%
loss, top1, top5 = run_manager.validate(net=subnet)
print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

