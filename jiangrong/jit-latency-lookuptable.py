import sys; sys.path.append("/workspace/once-for-all")
import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''

from ofa.layers import ConvLayer, LinearLayer
from ofa.model_zoo import MobileInvertedResidualBlock

import yaml
import torch.nn as nn
import torch

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def parse_op(op_str):
    terms = op_str.split('-')
    op_name = terms[0]
    op_dict = {}
    assert op_name in ["AvgPool2D", "expanded_conv",
                       "Logits", "Conv", "Conv_1", "Conv_2"]

    for term in terms[1:]:
        [key, value] = term.split(':')
        if key == "input" or key == "output":
            op_dict[key] = value
        elif key == "expand" or key == "kernel" or key == "stride":
            op_dict[key] = int(value)
        elif key == "idskip" or key == "se" or key == "hs":
            op_dict[key] = bool(int(value))
        else:
            raise Exception("error key: {}".format(key))

    input_channel = int(op_dict["input"].split('x')[-1])
    output_channel = int(op_dict["output"].split('x')[-1])
    if op_name == "Logits":
        op = LinearLayer(in_features=input_channel, out_features=output_channel)
    elif op_name == "AvgPool2D":
        op = nn.AdaptiveAvgPool2d((output_channel, int(op_dict["output"].split('x')[-2])))
    elif op_name == "expanded_conv":
        block_config = {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": input_channel,
                "out_channels": output_channel,
                "kernel_size": op_dict["kernel"],
                "stride": op_dict["stride"],
                "expand_ratio": op_dict["expand"] // input_channel,
                "mid_channels": op_dict["expand"],
                "act_func": "h_swish" if op_dict["hs"] else "relu",
                "use_se": op_dict["se"]
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": [
                    input_channel
                ],
                "out_channels": [
                    input_channel
                ],
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act"
            } if op_dict["idskip"] else None
        }
        op = MobileInvertedResidualBlock.build_from_config(block_config)
    elif op_name == ("Conv"):
        assert input_channel == 3
        op = ConvLayer(in_channels = 3, out_channels = output_channel, kernel_size=3, stride=2, dilation=1, groups=1, bias=False,
                       has_shuffle=False, use_bn=True, act_func='h_swish', dropout_rate=0, ops_order='weight_bn_act')
    elif op_name == ("Conv_1"):
        op = ConvLayer(in_channels = input_channel, out_channels = output_channel, kernel_size=1, stride=1, dilation=1, groups=1, bias=False,
                       has_shuffle=False, use_bn=True, act_func='h_swish', dropout_rate=0, ops_order='weight_bn_act')
    elif op_name == ("Conv_2"):
        op = ConvLayer(in_channels = input_channel, out_channels = output_channel, kernel_size=1, stride=1, dilation=1, groups=1, bias=False,
                       has_shuffle=False, use_bn=False, act_func='h_swish', dropout_rate=0, ops_order='weight_bn_act')

    if op_name == "Logits":
        input_tensor = torch.Tensor(1, 3, 7, 7)
        fake_op = nn.Sequential(nn.Conv2d(3, input_channel, 1)
                                ,nn.AdaptiveAvgPool2d((1,1))
                                ,View((1, input_channel)))
    else:
        input_tensor = torch.Tensor(1, 3, int(op_dict["input"].split('x')[-2]), int(op_dict["input"].split('x')[-3]))
        fake_op = nn.Conv2d(3, input_channel, 1)
    op = nn.Sequential(fake_op, op)

    _ = fake_op(input_tensor)
    _ = op(input_tensor)
    
    return op, fake_op, input_tensor

if __name__ == "__main__":
    test_op, prepare_op, input_t = parse_op("expanded_conv-input:20x20x48-output:20x20x48-expand:192-kernel:3-stride:1-idskip:1-se:1-hs:0")
    # RESOLUTION_LIST = [160, 176, 192, 208, 224]
    # idx = 1
    # with open('./jit-latency-lookuptable.meta', 'w') as wf:
    #     for resolution in RESOLUTION_LIST:
    #         fname = "./assets/{}_lookup_table.yaml".format(resolution)
    #         with open(fname, 'r') as fp:
    #             lut = yaml.load(fp)

    #         for opstr in lut.keys():
    #             print(opstr)
    #             test_op, prepare_op, input_t = parse_op(opstr)
    #             traced = torch.jit.trace(test_op, (input_t, ))
    #             traced.save('./assets/jits/{}_test.jit'.format(idx))
    #             fake_traced = torch.jit.trace(prepare_op, (input_t, ))
    #             fake_traced.save('./assets/jits/{}_fake.jit'.format(idx))
    #             input_size = 'x'.join(map(str, list(input_t.size())))
    #             wf.write("{},{},{},{}\n".format(idx, resolution, opstr, input_size))
    #             idx += 1

