from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset_define import *


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
        writer.add_images('RGBs', data[0], step)         #data.shape([2, 211, 3, 32, 32])

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

        writer.add_images('infrared', imgs[1:], step)     
        step = step + 1
    writer.close()