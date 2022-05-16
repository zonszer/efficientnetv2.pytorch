from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from image_crop import crop_image

class MyData(Dataset):

    def __init__(self, root_dir,  label_dir, dataset_type, transform):
        self.root_dir = root_dir
        self.dataset_type = dataset_type 
        self.image_path = os.path.join(self.root_dir, self.dataset_type)
        self.image_list = os.listdir(self.image_path)  # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        # self.label_list = os.listdir(self.label_path)

        #cut to patches
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        # label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.dataset_type, img_name)

        # label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        img = self.transform(img)
        
        imgs = crop_image(img, 32, 32)      #大图crop into 小图

        self.len = imgs.shape[0]
        return imgs

    def __len__(self):
        return self.len