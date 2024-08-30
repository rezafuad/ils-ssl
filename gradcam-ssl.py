###################################3
# Visualize HearMap by sum
# Zheng, Zhedong, Liang Zheng, and Yi Yang. "A discriminatively learned cnn embedding for person reidentification." ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14, no. 1 (2018): 13.
###################################

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from model import ft_net_swinv2large_384
#from utils import load_network
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM, \
        ScoreCAM, \
        GradCAMPlusPlus, \
        AblationCAM, \
        XGradCAM, \
        EigenCAM, \
        EigenGradCAM, \
        LayerCAM

from pytorch_grad_cam.utils.image import show_cam_on_image, \
        preprocess_image



def reshape_transform(tensor, height=12, width=12):
    print(tensor.shape)
    result = tensor.reshape(tensor.size(0),
            height, width, tensor.size(3))
    
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


"""
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
                height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
"""

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--index', default=1, type=int, help='index')

opt = parser.parse_args()

config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
opt.fp16 = config['fp16']
opt.which_epoch = "last-10" ## for PKU
#opt.which_epoch = "last-4" ## for MaSk1K
opt.h = 384
opt.w = 384

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=1) for x in ['gallery','query']}

#imgpath = image_datasets['query'].imgs
imgpath = image_datasets['gallery'].imgs


######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',opt.name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


model_structure = ft_net_swinv2large_384(opt.nclasses, input_size=(384,384), linear_num=0, cutnum=4)

model = load_network(model_structure)

#model.classifier.classifier = nn.Sequential()
model = model.eval().cuda()

print(model)

#target_layers = [model.model.layers[-1].blocks[-1].norm1]
target_layers = [model.model.norm] #layers[-1].blocks[-1].norm2]

#import random

#randind = random.randint(1, 100)
#print(randind)

#data = None
#for i in range(randind):
#for i in range(opt.index):
#    data = next(iter(dataloaders['gallery']))
#data = image_datasets['query'][opt.index-1]
data = image_datasets['gallery'][opt.index-1]
print(opt.index, imgpath[opt.index-1])

img, label = data
img = img.unsqueeze(0)
#print(img.size(), label)
#import sys
#sys.exit()

cam = GradCAM(model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform)

target_category = None

grayscale_cam = cam(input_tensor=img.cuda(),
        targets=None,
        eigen_smooth=True,
        aug_smooth=True)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

rgb_img = cv2.imread(imgpath[opt.index-1][0], 1)[:, :, ::-1]
print(rgb_img.shape)
h, w, c = rgb_img.shape
rgb_img = cv2.resize(rgb_img, (384, 384))
rgb_img = np.float32(rgb_img) / 255

print(grayscale_cam.max())
cam_image = show_cam_on_image(rgb_img, grayscale_cam)

cam_image = cv2.resize(cam_image, (w, h))

cv2.imwrite(f'gradcam-msk-'+str(label)+'-'+str(opt.index)+'.png', cam_image)


