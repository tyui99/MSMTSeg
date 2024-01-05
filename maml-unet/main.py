import os
import time
import datetime


import torch
import random
from model.unet import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from m_dataset import DriveDataset
import transforms as T
import numpy as np
import collections
import train_utils.distributed_utils as utils
from train_utils.train_and_eval import criterion

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model



import torch
import os
import time
import datetime
import matplotlib.pyplot as plt
import torch
class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)
import train_utils.distributed_utils as utils
def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)
from model.unet import UNet
import random
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from m_dataset import DriveDataset
import transforms as T
import time

mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)

task_data=[]
for i in range(3):
    n_way_data = [random.randint(1, 4) for i in range(3)]
    task_data.append(n_way_data)

s_dataset = DriveDataset(root=task_data[i],
                         train=True,
                         data_num=5,
                         transforms=get_transform(train=True, mean=mean, std=std))
num_workers = 0
s_loader = torch.utils.data.DataLoader(s_dataset,
                                       batch_size=2,
                                       num_workers=num_workers,
                                       shuffle=True,
                                       pin_memory=True,
                                       collate_fn=s_dataset.collate_fn)

model = create_model(num_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()