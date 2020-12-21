#%%
import torch
#%%
help(torch.ones)
# %%
import time
import math
time1 = time.time()
m = 0
n = 1
for i in range(1,10001):
    m += math.log2(i)
time2 = time.time()

time3 = time.time()
for j in range(10000):
    n *= 1.2
time4 = time.time()

print(time1-time2)
print(time3-time4)

# %%
