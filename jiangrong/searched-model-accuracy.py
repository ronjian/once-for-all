# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import argparse

import sys; sys.path.append('/workspace/once-for-all')
# from ofa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_codebase.run_manager import ImagenetRunConfig
from ofa.imagenet_codebase.run_manager import RunManager
from ofa.model_zoo import ofa_net


parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-p',
#     '--path',
#     help='The path of imagenet',
#     type=str,
#     default='/dataset/ILSVRC2012')
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='')
parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=128)
parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=8)
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    default='ofa_mbv3_d234_e346_k357_w1.2',
    choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3'],
    help='OFA networks')

args = parser.parse_args()
# if args.gpu == 'all':
#     device_list = range(torch.cuda.device_count())
#     args.gpu = ','.join(str(_) for _ in device_list)
# else:
#     device_list = [int(_) for _ in args.gpu.split(',')]
# args.batch_size = args.batch_size * max(len(device_list), 1)
# ImagenetDataProvider.DEFAULT_PATH = args.path
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
ofa_network = ofa_net(args.net, pretrained=True)
run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
# ofa_network.sample_active_subnet()

import json
# place the search network config here
with open("/workspace/once-for-all/jiangrong/assets/searched.json", 'r') as fp:
    net_config = json.load(fp)
ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])

subnet = ofa_network.get_active_subnet(preserve_weight=True)

# load finetuned weights
# init = torch.load("/workspace/once-for-all/jiangrong/exp/finetune/checkpoint.pth.tar", map_location='cpu')['state_dict']
# subnet.load_weights_from_net(init)

""" Test sampled subnet 
"""
run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
print('searched model resolution is ', net_config['r'][0])
run_config.data_provider.assign_active_img_size(net_config['r'][0])
run_manager.reset_running_statistics(net=subnet)

print('Test random subnet:')
print(subnet.module_str)

loss, top1, top5 = run_manager.validate(net=subnet)
print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))
