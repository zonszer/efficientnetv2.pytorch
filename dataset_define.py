from PIL import Image
import os
from torch.utils.data import Dataset
import torch

class MyData(Dataset):
    def __init__(self, root_dir, dataset_type, transform):
        self.root_dir = root_dir
        self.dataset_type = dataset_type 
        # self.label_dir = "position_label"
        self.image_path = os.path.join(self.root_dir, self.dataset_type)
        # self.label_path = os.path.join(self.image_path, self.label_dir)

        self.image_list = os.listdir(self.image_path)  # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        # self.label_list = os.listdir(self.label_path)

        self.transform = transform
    
    def create_onehot_position_label(self, patch_num):
        """将整数标签转为one-hot编码(感觉应该被crop函数调用)"""
        tensor = torch.tensor(range(patch_num))
        N = tensor.size(0)
        num_classes = patch_num
        one_hot = torch.zeros(N, num_classes).long()
        return one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
    
    def crop_image(self, img, height, width):
        """cut the image into patches- input/output: tensor """

        H = img.shape[1]
        W = img.shape[2]
        patch_num = min( H // height,  W // width)        #采取可用的最大的正方形面积patch_num = imgs.shape[0]

        imgs = img[:, 0:height*1, 0:width*1]
        imgs = imgs.unsqueeze(0)

        for i in range(2, patch_num + 1):
            for j in range(1, patch_num + 1):
                img_current = img[:, height*(i-1):height*i, height*(j-1):width*j]
                img_current = img_current.unsqueeze(0)
                imgs = torch.cat([imgs, img_current], dim=0)
        self.patch_num = imgs.shape[0]      
        self.one_hot = self.create_onehot_position_label(patch_num)
        
        return imgs

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        # label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.dataset_type, img_name)

        # label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        img = self.transform(img)
        
        imgs = self.crop_image(img, 32, 32)      #大图crop into 小图

        return imgs

    def transform_patches(self, patch_num):
        

    def __len__(self):
        "return the num_len of the cutten patches of one figure"
        # return self.len
        return len(self.image_list)