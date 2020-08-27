#%%
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import os; 
import copy
import sys; sys.path.append('/workspace/once-for-all')
from ofa.model_zoo import ofa_net
from ofa.imagenet_codebase.run_manager import ImagenetRunConfig
from ofa.imagenet_codebase.run_manager import RunManager
# from ofa.utils import download_url

from ofa.tutorial import AccuracyPredictor
from ofa.tutorial.evolution_finder import ArchManager

STAGE = 2
os.environ["CUDA_VISIBLE_DEVICES"] = ''

# set random seed
random_seed = 1028
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
CONF_DIR = './assets/accuracy_data/ofa_mbv3_d234_e346_k357_w1.2/'

#%%
# accuracy_predictor = AccuracyPredictor(pretrained=True
#                                     ,device='cpu'
#                                     ,fname='./assets/accuracy_data/ofa_mbv3_d234_e346_k357_w1.2.pth'
#                                     ,dropout=0.0)
# with open('./assets/accuracy_data/ofa_mbv3_d234_e346_k357_w1.2/9.json', 'r') as rf:
#     netconf = json.load(rf)
# # with open('./assets/searched.json', 'r') as rf:
# #     netconf = json.load(rf)
# ks_list = copy.deepcopy(netconf['ks'])
# ex_list = copy.deepcopy(netconf['e'])
# d_list = copy.deepcopy(netconf['d'])
# r = copy.deepcopy(netconf['r'])[0]
# print(r,d_list,ks_list,ex_list)
# print(netconf['acc'])
# feats = AccuracyPredictor.spec2feats(ks_list, ex_list, d_list, r).reshape(1, -1).to('cpu')
# all_feats = [feats]
# all_feats = torch.cat(all_feats, 0)
# preds = accuracy_predictor.model(all_feats).to('cpu')
# print(preds)
#%%
if STAGE == 1:
    # Stage1: collect data
    arch_manager = ArchManager()
    csum = 1000
    while True:
        net_config = arch_manager.random_sample()
        ofa_network.set_active_subnet(ks=net_config['ks']
                                    , d=net_config['d']
                                    , e=net_config['e']
                                    )
        subnet = ofa_network.get_active_subnet(preserve_weight=True)
        run_config = ImagenetRunConfig(test_batch_size=256, n_worker=8)
        run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
        run_config.data_provider.assign_active_img_size(net_config['r'][0])
        run_manager.reset_running_statistics(net=subnet)

        # print('=========> net_config is:', net_config)
        # print('=========> Random subnet is:', subnet.module_str)

        _, top1, _ = run_manager.validate(net=subnet)
        # print('==========> Results:  top1=%.1f' % (top1))
        net_config['acc'] = top1
        with open('{}/{}.json'.format(CONF_DIR, csum), 'w') as wf:
            json.dump(net_config, wf)
        csum+=1
else:
    # Stage2: training
    accuracy_predictor = AccuracyPredictor(pretrained=False,device='cpu',dropout=0.0)
    # accuracy_predictor = AccuracyPredictor(pretrained=True,device='cpu',fname='./assets/accuracy_data/ofa_mbv3_d234_e346_k357_w1.2.pth')
    batch_size = 32
    net_confs = [os.path.join(CONF_DIR, each) for each in os.listdir(CONF_DIR)]
    optimizer = torch.optim.SGD(accuracy_predictor.model.parameters(), 1e-6, momentum=0.1, nesterov=True)
    # optimizer = torch.optim.Adam(accuracy_predictor.model.parameters(), 1e-6)
    try:
        while True:
            all_feats = []
            gts = []
            for i in range(batch_size):
                with open(random.choice(net_confs), 'r') as rf:
                    netconf = json.load(rf)
                ks_list = copy.deepcopy(netconf['ks'])
                ex_list = copy.deepcopy(netconf['e'])
                d_list = copy.deepcopy(netconf['d'])
                r = copy.deepcopy(netconf['r'])[0]
                gts.append(netconf['acc'])
                feats = AccuracyPredictor.spec2feats(ks_list, ex_list, d_list, r).reshape(1, -1).to('cpu')
                all_feats.append(feats)
            all_feats = torch.cat(all_feats, 0)
            preds = accuracy_predictor.model(all_feats).to('cpu')
            gts = torch.Tensor(gts).to('cpu')
            gts = gts / 100.0
            loss = F.mse_loss(preds, gts, reduction='sum')
            # loss = loss * 100.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss, gts.mean(), preds.mean(), gts[0], preds[0])
    except (KeyboardInterrupt, SystemExit):
        print('saving trained model')
        torch.save(accuracy_predictor.model.state_dict(), './assets/accuracy_data/ofa_mbv3_d234_e346_k357_w1.2.pth')
        exit()
