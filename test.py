from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from dataset_define import *

def crop_image(img, height, width):
    """cut the image into patches- input/output: tensor """

    H = img.shape[1]
    W = img.shape[2]
    patch_num = min( H // height,  W // width)        #采取可用的最大的正方形面积

    imgs = img[:, 0:height*1, 0:width*1]
    imgs = imgs.unsqueeze(0)
    for i in range(2, patch_num + 1):
        for j in range(1, patch_num + 1):
            img_current = img[:, height*(i-1):height*i, height*(j-1):width*j]
            img_current = img_current.unsqueeze(0)
            imgs = torch.cat([imgs, img_current], dim=0)
            
    return imgs

if __name__ == '__main__':
    #1. create my dataset
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])        ##opt:transforms.resize(32)
    root_dir = "data"
    dataset_type = "train"
    label = "RGB"
    RGB_dataset = MyData(root_dir, label, dataset_type, transform)

    #2. crop into patches 
    img1 = RGB_dataset[0]['img']
    images = crop_image(img1, 32, 32)

    #output
    writer = SummaryWriter("attempt_1_logs")
    step = 0
    for i in range(images.shape[0]):
        writer.add_image('RGBs', images[i], step)      
        # writer.add_image("RGBs{}".format(i), images[i])         
        step = step + 1
