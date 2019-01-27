import lmdb
import os
import pickle
import numpy as np
import torch

a = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
b = np.array([[4,4,4],[4,4,4]], dtype=np.float32)
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
print(dist(a,b))
a = torch.from_numpy(a)
b = torch.from_numpy(b)
#result = torch.norm(a-b, p=2)
#print(result)
c = a.repeat(5,1,1)
print(c.size())