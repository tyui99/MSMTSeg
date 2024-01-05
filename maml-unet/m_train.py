import os
import time
import datetime
import torchvision.transforms as transform
import pickle
from torchvision.utils import save_image
from sklearn.metrics import f1_score
import torch
import random
from model.unet import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from m_dataset import DriveDataset
from my_dataset  import DriveDataset as ddataset
import transforms as T
import numpy as np
import collections
import train_utils.distributed_utils as utils
from train_utils.train_and_eval import criterion
import torchvision.transforms.functional as F
from PIL import Image
import copy
from torch.nn import Module

test_root = r"/home/chenxing/maml-u-net/data/test"
q_root = r"/home/chenxing/maml-u-net/data/ROIs"
def list_files_in_directory(folder_path):
    try:
        # 获取指定文件夹中的文件列表
        files = os.listdir(folder_path)
        return files

    except FileNotFoundError:
        print("Error: The specified folder '{}' was not found.".format(folder_path))
        return None

def count_parameters(model: Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2* base_size)
        #*0.8
        """
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        """
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

   # if train:
        #return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
   # else:
    return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def train_one_task(args,model_g,task_data, optimizer_l):
    m_loss=[]
    loss_weight = None
    device=args.device
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    s_dataset = DriveDataset(root=task_data,
                            train=True,
                            data_num=(args.s_data+args.q_data),
                            transforms=get_transform(train=True, mean=mean, std=std))
                  
    s_loader = torch.utils.data.DataLoader(s_dataset,
                                                           batch_size=args.batch_size,
                                                           num_workers=0,
                                                           shuffle=True,
                                                           pin_memory=True,
                                                           collate_fn=s_dataset.collate_fn)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    model_g.train()
    for i, data in enumerate(s_loader):
            image, target = data
            save_image(image, '/home/chenxing/maml-u-net/image/output_image.png')
            tensor_data_uint8 = target.to(torch.uint8)
            to_pil = transform.ToPILImage()
            image_pil = to_pil(tensor_data_uint8)
            image_pil.save("/home/chenxing/maml-u-net/target/saved_image.png")
            image, target = image.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model_g(image)
                if i < args.s_data:
                    s_loss = criterion(output, target, loss_weight, num_classes=args.num_classes + 1, ignore_index=-1)
                    optimizer_l.zero_grad()
                    s_loss.backward()
                    optimizer_l.step()
                else:
                    meat_loss = criterion(output, target, loss_weight, num_classes=args.num_classes + 1,
                                          ignore_index=-1)
                    m_loss.append(meat_loss)
    m_loss=torch.stack(m_loss).mean()
    return m_loss


def train_one_task_1(args, model_g,task_data,q_root,optimizer_l):
    PATH_g = r"./model_g.pt"
    m_loss = []
    loss_weight = None
    device = args.device
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    s_dataset = DriveDataset(root=task_data,
                             train=True,
                             data_num=args.s_data,
                             transforms=get_transform(train=True, mean=mean, std=std))

    s_loader = torch.utils.data.DataLoader(s_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=0,
                                           shuffle=True,
                                           pin_memory=True,
                                           collate_fn=s_dataset.collate_fn)

    q_dataset = ddataset(root=q_root,
                             datanum=args.q_data,
                             train=False,
                             transforms=get_transform(train=True, mean=mean, std=std))

    q_loader = torch.utils.data.DataLoader(q_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=0,
                                           shuffle=True,
                                           pin_memory=True,
                                           collate_fn=s_dataset.collate_fn)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    initial_state_dict = model_g.state_dict()
    model_g.train()
    for i, data in enumerate(s_loader):
        image, target = data
        save_image(image, '/home/chenxing/maml-u-net/image/output_image.png')
        tensor_data_uint8 = target.to(torch.uint8)
        to_pil = transform.ToPILImage()
        image_pil = to_pil(tensor_data_uint8)
        image_pil.save("/home/chenxing/maml-u-net/target/saved_image.png")
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model_g(image)
            s_loss = criterion(output, target, loss_weight, num_classes=args.num_classes + 1, ignore_index=-1)
            optimizer_l.zero_grad()
            s_loss.backward()
            optimizer_l.step()

    for i, data in enumerate(q_loader):
        image, target = data
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model_g(image)
            meat_loss = criterion(output, target, loss_weight, num_classes=args.num_classes + 1,
                                      ignore_index=-1)
            m_loss.append(meat_loss)


    optimizer_l.zero_grad()
    m_loss = torch.stack(m_loss).mean()
    m_loss.backward()
    model_g.load_state_dict(initial_state_dict)
    optimizer_l.step()
    for param in model_g.parameters():
        print(param.grad is not None)



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    #模型初始化
    model_g = create_model(num_classes=num_classes)
    model_g.to(device)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_g.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    params_to_optimize_l = [p for p in model_g.parameters() if p.requires_grad]
    best_dice = 0.
    start_time = time.time()
    loss_weight = None
    best_loss = 10
    PATH_g = r"./model_g.pt"
    torch.save(model_g, PATH_g)
    #输出model_g参数数量
    #开始训练
    for epoch in range(args.start_epoch, args.epochs):
        print("第{}轮元学习开始：".format(epoch+1))
        meta_loss = []
        task_data = []

        optimizer_l = torch.optim.SGD(
            params_to_optimize_l,
            lr= args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
        # 读取每个task中的数据
        task_data= [random.randint(1, 157) for i in range(args.n_way)]
        #更新model_g参数
        train_one_task_1(args, model_g, task_data, q_root, optimizer_l)
        #meta_loss=torch.stack(meta_loss).mean()
        #对未更新的model_g进行测试
        num_params = count_parameters(model_g)
        print(f"Number of parameters in the model: {num_params}")
        model=torch.load(PATH_g)
        model.to(device)
        t_dataset = ddataset(root=q_root,
                             datanum=10,
                             train=False,
                             transforms=get_transform(train=True, mean=mean, std=std))
        num_workers = 0
        t_loader = torch.utils.data.DataLoader(t_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=t_dataset.collate_fn)
        test_loss=[]
        model.eval()
        for i, data in enumerate(t_loader):
            image, target = data
            image, target = image.to(device), target.to(device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target, loss_weight, num_classes=num_classes,ignore_index=255)
                test_loss.append(loss)

        test_loss = torch.stack(test_loss).mean()
        test_loss = test_loss.item()
        print("本轮元学习测试损失为:{}".format(test_loss))
        model_g.eval()
        # 保存最优权重
        test_loss=[]
        for i, data in enumerate(t_loader):
            image, target = data
            image, target = image.to(device), target.to(device)
            with torch.no_grad():
                output = model_g(image)
                loss = criterion(output, target, loss_weight, num_classes=num_classes,ignore_index=255)
                test_loss.append(loss)
        test_loss = torch.stack(test_loss).mean()
        test_loss = test_loss.item()
        print("梯度更新后损失为：{}".format(test_loss))
        if best_loss >test_loss:
            best_loss = test_loss
            PATH = r"./model_best.pt"
            torch.save(model_g, PATH)
        torch.save(model_g, PATH_g)
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default=r"/home/chenxing/maml-u-net/data", help="DRIVE root")
    # exclude background2
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--n_way", default=3, type=int)
    parser.add_argument("--s_data", default=10, type=int)
    parser.add_argument("--q_data", default=5, type=int)
    parser.add_argument("--t_task", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=20000, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='The learning rate of of the support set.')

    parser.add_argument('--lr', default=0.00005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)