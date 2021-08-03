import torch
from torch import nn
from d2l import torch as d2l

X = torch.randn(1, 1, 2, 2)
print(X)
X = torch.randn(1, 2, 2, 2)
print(X)
X = torch.randn(2, 1, 2, 2)
print(X)