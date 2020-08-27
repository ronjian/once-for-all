#%%
import os
import json
import numpy as np
# %%
acc_dir = "/workspace/once-for-all/jiangrong/assets/accuracy_data/ofa_mbv3_d234_e346_k357_w1.2"
acc_files = [os.path.join(acc_dir, each) for each in os.listdir(acc_dir)]
# %%
len(acc_files)
# %%
acc_dicts = []
for acc_file in acc_files:
    with open(acc_file, 'r') as rf:
        acc_dicts.append(json.load(rf))
# %%
res_accs = {}
for each in acc_dicts:
    r = each['r'][0]
    acc = each['acc']
    res_accs[r] = (res_accs.get(r) or []) + [acc]

# %%
res_accs.keys()
# %%
for k, v in res_accs.items():
    print(k, np.mean(v), np.median(v), np.max(v), np.min(v))
# %%
