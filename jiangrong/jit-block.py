import torch

import sys; sys.path.append("/workspace/once-for-all")
from ofa.model_zoo import MobileInvertedResidualBlock
from ofa.layers import ConvLayer, PoolingLayer, LinearLayer, 

block_config = {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 6,
                "mid_channels": 288,
                "act_func": "h_swish",
                "use_se": True
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": [
                    3
                ],
                "out_channels": [
                    3
                ],
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act"
            }
        }
block = MobileInvertedResidualBlock.build_from_config(block_config)
_ = block(torch.Tensor(1, 3, 224, 224))
trace_model = torch.jit.trace(block, (torch.Tensor(1, 3, 224, 224), ))
trace_model.save('./assets/MobileInvertedResidualBlock.jit')