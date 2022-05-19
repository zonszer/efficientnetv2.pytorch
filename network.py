
from ctypes.wintypes import RGB
from re import S
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ConvTranspose2d
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset_define import *

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

if __name__ == '__main__':
    #1. create my dataset
    transform = transforms.Compose([transforms.ToTensor()])        #opt:transforms.Grayscale(num_output_channels=1)，  transforms.resize(32)
    root_dir = "data"
    dataset_type = "train"
    RGB_dataset = MyData(root_dir, dataset_type, transform)
    
    #output---->注意data的输出格式应为[batch_size,C,H,W]
    train_loader = DataLoader(RGB_dataset, batch_size = 2, shuffle=False)      
    writer = SummaryWriter("attempt_1_logs")
    step = 0
    for data in train_loader:
        # writer.add_images('RGBs', data[0], step)         #data.shape([2, 211, 3, 32, 32])

        #对patches进行transform
        scale_to64 = transforms.Pad(padding=32)
        Random_scale_rotate_translate = transforms.RandomAffine(degrees=(-180, 180),   
                                                                translate = (0.2,0.2),
                                                                scale = (0.5, 1),
                                                                shear = None)
        imgs = data[1][0]
        imgs = scale_to64(imgs)
        imgs = Random_scale_rotate_translate(imgs)
        imgs = imgs.unsqueeze(0)
        for i in range(data.shape[1]):
            img = data[1][i]
            img = scale_to64(img)
            transformed_img = Random_scale_rotate_translate(img)
            transformed_img = transformed_img.unsqueeze(0)
            imgs = torch.cat([imgs, transformed_img], dim=0)
        
        SAE = auto_encoder()
        # infrared_patches = torch.reshape(imgs[1:], ( -1, 1, 64, 64))
        infrared_patches = imgs[1:]
        RGB_patches = data[0]
        infrared_output = SAE(infrared_patches)
        RGB_output = SAE(RGB_patches)

        writer.add_images('infrared_output', infrared_output, step)     
        writer.add_images('RGB_output', RGB_output, step)     
        step = step + 1
        writer.add_graph(SAE, infrared_patches)
    writer.close()
