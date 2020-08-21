# import json
# import torch
# import sys; sys.path.append('/workspace/once-for-all')
# from ofa.model_zoo import ofa_net


# ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
# # place the search network config here
# with open("/workspace/once-for-all/jiangrong/assets/searched.json", 'r') as fp:
#     net_config = json.load(fp)
# ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])

# print('export the searched model to jit')
# net = ofa_network.get_active_subnet(preserve_weight=True)
# input_tensor = torch.Tensor(1,3, net_config['r'][0], net_config['r'][0])
# _ = net(input_tensor)
# traced = torch.jit.trace(net, (input_tensor, ))
# traced.save('./assets/searched.jit')