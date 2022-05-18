
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ConvTranspose2d
from torch.utils.tensorboard import SummaryWriter


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.model1 = Sequential(
            Conv2d(1, 8, 3, padding=1),
            Conv2d(8, 8, 4, stride=2, padding=1),
            Conv2d(8, 16, 4, stride=2, padding=1),
            Conv2d(16, 16, 4, stride=2, padding=1),
            Conv2d(16, 32, 4, stride=2, padding=1), 
            Conv2d(32, 32, 4, stride=2, padding=1), #cov6
            
            
            ConvTranspose2d(32, 32, 4, stride=2, padding=1),   #decov4
            ConvTranspose2d(32, 16, 4, stride=2, padding=1),   #decov3
            ConvTranspose2d(16, 16, 4, stride=2, padding=1),   #decov2
            ConvTranspose2d(16, 8, 4, stride=2, padding=1),   #decov1
            ConvTranspose2d(8, 1, 4, stride=2, padding=1),   #decov0
        )

    def forward(self, x):
        x = self.model1(x)
        return x

SAE = auto_encoder()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)
writer.close()