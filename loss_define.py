import torch.nn as nn
import torch

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        
    def forward(self, pred, truth):
        return  torch.mean(torch.mean((pred-truth)**2,1),0)