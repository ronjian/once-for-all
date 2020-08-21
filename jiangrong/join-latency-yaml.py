#%%
import os, sys, yaml
# %%
# opname_list = []
op_dict = {}
with open("./assets/jit-latency-lookuptable.meta", 'r') as rf:
    for line in rf:
        line = line.strip()
        idx,_,opname,_ = line.split(",")
        idx = int(idx)
        op_dict[opname] = idx
#         opname_list.append(opname)
# print(len(set(opname_list)))
# %%
latency_dict = {}
with open("./assets/rv1126_operation_cost.dat", 'r') as rf:
    for line in rf:
        line = line.strip()
        idx,postfix,latency = line.split(",")
        idx,postfix,latency = int(idx),postfix,float(latency)
        if latency_dict.get(idx):
            if postfix == "test":
                latency_dict[idx] += latency
            else:
                latency_dict[idx] -= latency
        else:
            if postfix == "test":
                latency_dict[idx] = latency
            else:
                latency_dict[idx] = -1. * latency
for k, v in latency_dict.items():
    if v < 0:
        latency_dict[k] = 0.0116
# %%
for resolution in [160, 176, 192, 208, 224]:
    fname = "./assets/{}_lookup_table.yaml".format(resolution)
    with open(fname, 'r') as fp:
        lut = yaml.load(fp)
        for key in lut.keys():
            if not latency_dict.get(op_dict[key]):
                lut[key]['mean'] = 5.0
            else:
                lut[key]['mean'] = latency_dict[op_dict[key]]
    with open("./assets/rv1126-latency-table/{}_lookup_table.yaml".format(resolution), 'w') as wf:
        yaml.dump(lut, wf)

# %%
# max(list(latency_dict.values())), min(list(latency_dict.values()))
# %%
