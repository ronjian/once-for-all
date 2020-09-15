# #%%
# import copy
# import random

# import torch
# import torch.nn as nn

# # from ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicKernelConvLayer, DynamicBottleneck
# from ofa.elastic_nn.modules.dynamic_layers import DynamicKernelConvLayer, DynamicBottleneck
# # from ofa.layers import ConvLayer, IdentityLayer, MBInvertedConvLayer
# from ofa.layers import ConvLayer
# # from ofa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3, MobileInvertedResidualBlock
# # from ofa.imagenet_codebase.utils import make_divisible, int2list
# from ofa.imagenet_codebase.utils import int2list
# from ofa.elastic_nn.networks.yolov5_common import SPP, Concat

# class OFAYolov5s(Yolov5s):
#     def __init__(self, n_classes=29, bn_param=(0.1, 1e-5), ks_list=[1, 3, 5, 7]):

#         self.ks_list = int2list(ks_list, 1) # int2list([3,5,7], 1) => [3, 5, 7]
#         self.ks_list.sort()
#         self.backbone = []
#         self.dynamic_kernal_ops = []
#         self.backbone.append(ConvLayer(in_channels = 3, out_channels = 32
#                             , kernel_size=5, stride=2, dilation=1, groups=1
#                             , bias=False, has_shuffle=False,use_bn=True
#                             , act_func='relu', dropout_rate=0, ops_order='weight_bn_act'))
#         self.base_stage_width = [64, 128, 256, 512]
#         self.base_stage_depth = [1, 3, 3, 1]
#         self.downsample_8_idx = 1 + self.base_stage_depth[0] + 1 + self.base_stage_depth[1] - 1
#         self.downsample_16_idx = 1 + self.base_stage_depth[0] + 1 + self.base_stage_depth[1] + 1 + self.base_stage_depth[2] - 1
#         in_channel = 32
#         for idx,(width, depth) in enumerate(zip(self.base_stage_width, self.base_stage_depth)):
#             output_channel = width
#             self.backbone.append(DynamicKernelConvLayer(
#                     in_channel, output_channel
#                     , kernel_size_list=self.ks_list
#                     , stride=2, act_func='relu'
#                 ))
#             self.dynamic_kernal_ops.append(self.backbone[-1].kernelConv.conv)
#             if idx == len(self.base_stage_width) - 1:
#                 self.backbone.append(SPP(output_channel, output_channel, k=(5, 9, 13)))
#             for i in range(depth):
#                 self.backbone.append(DynamicBottleneck(
#                     output_channel, output_channel
#                     , kernel_size_list=self.ks_list, shortcut=True
#                     , stride=1, act_func='relu'
#                 ))
#                 self.dynamic_kernal_ops.append(self.backbone[-1].cv1.conv)
#                 self.dynamic_kernal_ops.append(self.backbone[-1].cv2.conv)

#             in_channel = output_channel
        
#         self.neck = DynamicBottleneck(
#                     output_channel, output_channel
#                     , kernel_size_list=self.ks_list, shortcut=False
#                     , stride=1, act_func='relu'
#                 )
#         self.dynamic_kernal_ops.append(self.neck.cv1.conv)
#         self.dynamic_kernal_ops.append(self.neck.cv2.conv)

#         self.head0_1 = nn.Conv2d(256, 29, 1,1,0)

#         self.head1_1 = nn.Upsample((512//16, 672//16), None, 'nearest')
#         self.head1_2 = Concat(1)
#         self.head1_3 = DynamicKernelConvLayer(
#                             512 + 256, 256
#                             , kernel_size_list=self.ks_list
#                             , stride=1, act_func='relu'
#                         )
#         self.dynamic_kernal_ops.append(self.head1_3.kernelConv.conv)
#         self.head1_4 = DynamicBottleneck(
#                             256, 256
#                             , kernel_size_list=self.ks_list, shortcut=False
#                             , stride=1, act_func='relu')
#         self.dynamic_kernal_ops.append(self.head1_4.cv1.conv)
#         self.dynamic_kernal_ops.append(self.head1_4.cv2.conv)
#         self.head1_5 = nn.Conv2d(256, 29, 1,1,0)

#         self.head2_1 = nn.Upsample((512//8, 672//8), None, 'nearest')
#         self.head2_2 = Concat(1)
#         self.head2_3 = DynamicKernelConvLayer(
#                             256 + 128, 256
#                             , kernel_size_list=self.ks_list
#                             , stride=1, act_func='relu'
#                         )
#         self.dynamic_kernal_ops.append(self.head2_3.kernelConv.conv)
#         self.head2_4 = DynamicBottleneck(
#                             256, 256
#                             , kernel_size_list=self.ks_list, shortcut=False
#                             , stride=1, act_func='relu')
#         self.dynamic_kernal_ops.append(self.head2_4.cv1.conv)
#         self.dynamic_kernal_ops.append(self.head2_4.cv2.conv)
#         self.head2_5 = nn.Conv2d(256, 29, 1,1,0)

#     @staticmethod
#     def name():
#         return 'OFAYolov5s'

#     def forward(self, x):
#         for idx, layer in enumerate(self.backbone):
#             x = layer(x)
#             if idx == self.downsample_8_idx:
#                 down8tensor = x
#             elif idx == self.downsample_16_idx:
#                 down16tensor = x
#         x = self.neck(x)

#         h1 = self.head1_1(x)
#         h1 = self.head1_2([h1, down16tensor])
#         h1 = self.head1_3(h1)
#         h1 = self.head1_4(h1)

#         h2 = self.head2_1(h1)
#         h2 = self.head2_2([h2, down8tensor])
#         h2 = self.head2_3(h2)
#         h2 = self.head2_4(h2)

#         h0 = self.head0_1(x)
#         h1 = self.head1_5(h1)
#         h2 = self.head2_5(h2)

#         return (h0,h1,h2)

#     @property
#     def module_str(self):
#         _str = "TODO:jiangrong"
#         # _str = self.first_conv.module_str + '\n'
#         # _str += self.blocks[0].module_str + '\n'

#         # for stage_id, block_idx in enumerate(self.block_group_info):
#         #     depth = self.runtime_depth[stage_id]
#         #     active_idx = block_idx[:depth]
#         #     for idx in active_idx:
#         #         _str += self.blocks[idx].module_str + '\n'

#         # _str += self.final_expand_layer.module_str + '\n'
#         # _str += self.feature_mix_layer.module_str + '\n'
#         # _str += self.classifier.module_str + '\n'
#         return _str

#     @property
#     def config(self):
#         return {
#             # 'name': OFAYolov5s.__name__,
#             # 'bn': self.get_bn_param(),
#             # 'first_conv': self.first_conv.config,
#             # 'blocks': [
#             #     block.config for block in self.blocks
#             # ],
#             # 'final_expand_layer': self.final_expand_layer.config,
#             # 'feature_mix_layer': self.feature_mix_layer.config,
#             # 'classifier': self.classifier.config,
#         }

#     @staticmethod
#     def build_from_config(config):
#         raise ValueError('do not support this function')

#     """ Adding methods """

#     def load_weights_from_net(self, src_model_dict):
#         model_dict = self.state_dict()
#         for key in src_model_dict:
#             if key in model_dict:
#                 new_key = key
#             elif '.bn.bn.' in key:
#                 new_key = key.replace('.bn.bn.', '.bn.')
#             elif '.conv.conv.weight' in key:
#                 new_key = key.replace('.conv.conv.weight', '.conv.weight')
#             elif '.linear.linear.' in key:
#                 new_key = key.replace('.linear.linear.', '.linear.')
#             ##############################################################################
#             elif '.linear.' in key:
#                 new_key = key.replace('.linear.', '.linear.linear.')
#             elif 'bn.' in key:
#                 new_key = key.replace('bn.', 'bn.bn.')
#             elif 'conv.weight' in key:
#                 new_key = key.replace('conv.weight', 'conv.conv.weight')
#             else:
#                 raise ValueError(key)
#             assert new_key in model_dict, '%s' % new_key
#             model_dict[new_key] = src_model_dict[key]
#         self.load_state_dict(model_dict)

#     """ set, sample and get active sub-networks """
#     def sample_active_subnet(self):
#         ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
#             else self.__dict__['_ks_include_list']
#         # sample kernel size
#         ks_setting = []
#         if not isinstance(ks_candidates[0], list):
#             ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
#         for k_set in ks_candidates:
#             k = random.choice(k_set)
#             ks_setting.append(k)
#         self.set_active_subnet(ks_setting)
#         return {'ks': ks_setting,}

#     def set_active_subnet(self, ks=None):
#         # jiangrong: 跟踪 active_kernel_size 观察动态的过程
#         # width_mult_id = int2list(wid, 4 + len(self.block_group_info))
#         ks = int2list(ks, len(self.dynamic_kernal_ops) - 1)
#         # expand_ratio = int2list(e, len(self.blocks) - 1)
#         # depth = int2list(d, len(self.block_group_info))

#         for op, k in zip(self.dynamic_kernal_ops, ks):
#             if k is not None:
#                 op.active_kernel_size = k