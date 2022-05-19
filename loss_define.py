import torch.nn as nn
import torch

class end_toend_loss(nn.Module):
    def __init__(self):
        super(end_toend_loss, self).__init__()

    def forward(self, pred, truth):
        return  criterion = nn.MSELoss()


class cross_loss(nn.Module):
    def __init__(self):
        super(cross_loss, self).__init__()

    def forward(self, pred, truth):
        return  torch.mean(torch.mean((pred-truth)**2,1),0)