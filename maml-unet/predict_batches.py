import os
import time
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

#rom src import UNet,UNetWithResnet50Encoder,VGG16UNet,deeplabv3_resnet50


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 3  # exclude background
    weights_path = "/home/chenxing/maml-u-net/model_best.pt"
    #weights_path='/home/amaxv1004/Data/LXY/MTL_Segmentation/logs/meta/Fewshot_UNet_MTL_shot3_way3_query1_step20_gamma0.5_lr10.0005_lr20.005_batch50_maxepoch200_baselr0.01_updatestep20_stepsize20_exp1/max_iou.pth'
    img_path= "/home/chenxing/maml-u-net/data/ROIs/test/image"
    final_path='/home/chenxing/maml-u-net/data/1'
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    #roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    #assert os.path.exists(weights_path), f"weights {weights_path} not found."
    #assert os.path.exists(img_path), f"image {img_path} not found."
    #assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    print("using {} device.".format(device))

    # create model
    #model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    #model = UNetWithResnet50Encoder(n_classes=classes+1).cuda()
    #model = deeplabv3_resnet50(aux=True, num_classes=classes+1)

    # load weights
    #model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    #model.load_state_dict(torch.load(weights_path, map_location='cpu'))['params']
    model=torch.load(weights_path)
    model.to(device)

    # load roi mask
    #roi_img = Image.open(roi_mask_path).convert('L')
    #roi_img = np.array(roi_img)
    pathDir = os.listdir(img_path)    #取图片的原始路径
    for name in tqdm(pathDir):
    # load image
        img=os.path.join(img_path,name)
        #target=os.path.join(target_path,name)

        original_img = Image.open(img).convert('RGB')
        #target_img = Image.open(target).convert('L')

    # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
    # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        model.eval()  # 进入验证模式


    
        with torch.no_grad():
        # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))
            print(output)
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            #prediction[prediction == 1] = 38
            #prediction[prediction == 2] = 75
            #prediction[prediction == 3] = 14
            #prediction[prediction == 4] = 113
        # 将不敢兴趣的区域像素设置成0(黑色)
            #prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            mask.save(os.path.join(final_path,name))


if __name__ == '__main__':
    main()
