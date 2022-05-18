
import torch
from torch import nn, strided
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
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
            
            
            
            Conv2d(32, 32, 4, stride=2, padding=1), #decov4
            # MaxPool2d(2),
            # Flatten(),
            # Linear(1024, 64),
            # Linear(64, 10)
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