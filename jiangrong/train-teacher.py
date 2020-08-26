import argparse
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

import horovod.torch as hvd
import torch

import sys; sys.path.append("/workspace/once-for-all")
from ofa.imagenet_codebase.run_manager import DistributedImageNetRunConfig
from ofa.imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from ofa.model_zoo import ofa_net
import time

parser = argparse.ArgumentParser()

args = parser.parse_args()
args.path = 'exp/indoor/mbv3'
args.manual_seed = 1028
args.warmup_lr = -1
args.image_size = '224'
args.warmup_epochs = 5
args.continuous_size = False
args.not_sync_distributed_image_size = True

args.dynamic_batch_size = 1
args.n_epochs = 125
args.base_lr = 3e-2
args.lr_schedule_type = 'cosine'
args.base_batch_size = 64
# args.valid_size = 10000
args.valid_size = 5000
args.opt_type = 'sgd'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = True
args.teacher_model = None
args.n_worker = 12
args.resize_scale = 0.08
args.distort_color = 'tf'
args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
# args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False
args.dataset='indoor'

os.makedirs(args.path, exist_ok=True)
hvd.init()
torch.cuda.set_device(hvd.local_rank())
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)
random.seed(args.manual_seed)

# image size
args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
if len(args.image_size) == 1:
    args.image_size = args.image_size[0]
MyRandomResizedCrop.CONTINUOUS = args.continuous_size
MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size
# build run config from args
args.lr_schedule_param = None
args.opt_param = {
    'momentum': args.momentum,
    'nesterov': not args.no_nesterov,
}
num_gpus = hvd.size()
args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
if args.warmup_lr < 0:
    args.warmup_lr = args.base_lr
args.train_batch_size = args.base_batch_size
args.test_batch_size = args.base_batch_size * 4
run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

# from ofa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3Large
# net = MobileNetV3Large()
# net.train()
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=False)
ofa_network.set_active_subnet(ks=7, d=4, e=6)
net = ofa_network.get_active_subnet(preserve_weight=True)

compression = hvd.Compression.fp16
run_manager = DistributedRunManager(
    args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0), init=False)
run_manager.save_config()
run_manager.reset_running_statistics()
# hvd broadcast
run_manager.broadcast()

run_manager.train(args, warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr)
