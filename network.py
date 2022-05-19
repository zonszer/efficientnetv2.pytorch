
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
                                                                scale = (0.75, 1),
                                                                shear = None)
        infrared_imgs = data[1][0]; RGB_imgs = data[0][0]
        infrared_imgs = scale_to64(infrared_imgs); RGB_imgs = scale_to64(RGB_imgs)
        infrared_imgs = Random_scale_rotate_translate(infrared_imgs)
        infrared_imgs = infrared_imgs.unsqueeze(0); RGB_imgs = RGB_imgs.unsqueeze(0)
        for i in range(data.shape[1]):
            infrared_img = data[1][i]; RGB_img = data[0][i]
            infrared_img = scale_to64(infrared_img); RGB_img = scale_to64(RGB_img)
            transformed_img = Random_scale_rotate_translate(infrared_img)   #infrared
            transformed_img = transformed_img.unsqueeze(0)
            infrared_imgs = torch.cat([infrared_imgs, transformed_img], dim=0)

            transformed_img = RGB_img.unsqueeze(0)      #RGB
            infrared_imgs = torch.cat([RGB_imgs, transformed_img], dim=0)
        
        SAE = auto_encoder()
        loss = 
        infrared_patches = infrared_imgs[1:]; RGB_patches = RGB_imgs[1:]
        infrared_output = SAE(infrared_patches); RGB_output = SAE(RGB_patches)

        writer.add_images('infrared_output', infrared_output, step)     
        writer.add_images('RGB_output', RGB_output, step)     
        step = step + 1
        writer.add_graph(SAE, infrared_patches)


    writer.close()
