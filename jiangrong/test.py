#%%
def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]
# %%
import numpy as np
int2list(6,1)

# %%
int2list(1.0, 1)
# %%
int2list(3, 1)
# %%
int2list('3,5,7', 1)
# %%
