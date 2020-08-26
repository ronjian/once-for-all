import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
import json

import sys; sys.path.append('/workspace/once-for-all')
from ofa.model_zoo import ofa_net
from ofa.utils import download_url

from ofa.tutorial import AccuracyPredictor, LatencyTable, CustomizedLatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
imagenet_data_path = '/dataset/ILSVRC2012'


accuracy_predictor = AccuracyPredictor(
                            pretrained=True,
                            device='cpu')

target_hardware = 'note10'
# latency_table = LatencyTable(device=target_hardware)
latency_table = CustomizedLatencyTable(yaml_dir = "/workspace/once-for-all/jiangrong/assets/rv1126-latency-table")
latency_constraint = 23  # ms, suggested range [15, 33] ms

P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)
# start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search()
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
      (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))

# visualize the architecture of the searched sub-net
_, net_config, latency = best_info
print('input resolution is: ', net_config['r'][0])
ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
# print('net_config type', type(net_config))
with open("./assets/searched.json", 'w') as wf:
    json.dump(net_config, wf)
print('Architecture of the searched sub-net:')
print(ofa_network.module_str)

print('export the searched model to jit')
net = ofa_network.get_active_subnet(preserve_weight=True)
input_tensor = torch.Tensor(1,3, net_config['r'][0], net_config['r'][0])
_ = net(input_tensor)
traced = torch.jit.trace(net, (input_tensor, ))
traced.save('./assets/searched.jit')