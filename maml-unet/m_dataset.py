import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import random

def list_files_in_directory(folder_path):
    try:
        # 获取指定文件夹中的文件列表
        files = os.listdir(folder_path)

        # 输出文件名列表


        # 返回文件名列表
        return files

    except FileNotFoundError:
        print("Error: The specified folder '{}' was not found.".format(folder_path))
        return None

# 指定要查看的文件夹路径，替换成你想要的路径



class DriveDataset(Dataset):
    def __init__(self, root: [], train: bool, data_num:int,transforms=None):
        super(DriveDataset, self).__init__()
        folder_path = r"/home/chenxing/maml-u-net/data"
        self.flag = "Domain" if train else "target"
        folder_path = os.path.join(folder_path, self.flag)
        self.transforms = transforms
        file_list = list_files_in_directory(folder_path)
        self.img_list=[]
        self.manual=[]
        for i in root:
            iroot=file_list[i]
            data_root = os.path.join(folder_path,iroot)
            assert os.path.exists(data_root), f"path '{data_root}' does not exists."
            img_names0 = [i for i in os.listdir(os.path.join(data_root, "image")) if i.endswith(".png")]
            s_data = [random.randint(0, len(img_names0) - 1) for i in range(data_num)]
            img_names = [img_names0[i] for i in s_data]
            img_list0 = [os.path.join(data_root, "image", i) for i in img_names]
            manual0 = [os.path.join(data_root, "mask", i) for i in img_names]
            self.img_list =self.img_list+img_list0
            self.manual=self.manual+manual0


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.manual[idx]).convert('L')
        """
        pixels = mask.getdata()
        new_pixels = [int(value * 255) for value in pixels]
        mask = Image.new("L",mask.size)
        mask.putdata(new_pixels)
        """
        
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

