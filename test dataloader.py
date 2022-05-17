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
    train_loader = DataLoader(RGB_dataset, batch_size = 1)      
    writer = SummaryWriter("attempt_1_logs")
    step = 0
    for data in train_loader:
        writer.add_images('RGBs', data[0], step)         #data.shape([6, 211, 3, 32, 32])
        step = step + 1
    writer.close()