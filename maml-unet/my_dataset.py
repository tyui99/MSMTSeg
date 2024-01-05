import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import random
class DriveDataset(Dataset):
    def __init__(self, datanum:int,root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "train"
        data_root = root
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "image")) if i.endswith(".png")]
        s_data = [random.randint(0, len(img_names) - 1) for i in range(datanum)]
        img_names = [img_names[i] for i in s_data]

        self.img_list = [os.path.join(data_root, "image", i) for i in img_names]
        self.manual =[os.path.join(data_root, "mask", i) for i in img_names]

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.manual[idx]).convert('L')
  
        #manual = np.array(manual) / 255
        #roi_mask = Image.open(self.roi_mask[idx]).convert('L')
       #roi_mask = 255 - np.array(roi_mask)
        #mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        #mask = Image.fromarray(mask)

        #mask_ma = np.array(mask)/255
        #mask = Image.fromarray(mask_ma)
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

