###################################3
# Visualize HearMap by sum
# Zheng, Zhedong, Liang Zheng, and Yi Yang. "A discriminatively learned cnn embedding for person reidentification." ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14, no. 1 (2018): 13.
###################################

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
from utils import load_network
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')

opt = parser.parse_args()

config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_swin_vhdpl = config['use_swin_vhdpl']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
else:
    opt.h = 224
    opt.w = 224

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


def heatmap2d(img, arr):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Image")
    ax1 = fig.add_subplot(122, title="Heatmap")

    ax0.imshow(Image.open(img))
    heatmap = ax1.imshow(arr, cmap='viridis')
    fig.colorbar(heatmap)
    #plt.show()
    fig.savefig('heatmap')

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=1) for x in ['train']}

imgpath = image_datasets['train'].imgs
model, _, epoch = load_network(opt.name, opt)
model.classifier.classifier = nn.Sequential()
model = model.eval().cuda()

data = next(iter(dataloaders['train']))
img, label = data
with torch.no_grad():
    #x = model.model.conv1(img.cuda())
    #x = model.model.bn1(x)
    #x = model.model.relu(x)
    #x = model.model.maxpool(x)
    #x = model.model.layer1(x)
    #x = model.model.layer2(x)
    #output = model.model.layer3(x)
    #output = model.model.layer4(x)
    x = model.model.patch_embed(img.cuda())
    if model.model.absolute_pos_embed is not None:
        x = x + model.model.absolute_pos_embed
    x = model.model.pos_drop(x)
    
    numlayer = len(model.model.layers)
    for i in range(numlayer):
        winsize = model.model.layers[i].blocks[-1].window_size
        x = model.model.layers[i](x)
        if i < numlayer-1:
            print(i,"1",x.size())
            H, W = model.model.layers[i].blocks[-1].input_resolution
            print(i,"10", H, W)
            H, W = H // 2, W // 2
            B, L, C = x.shape
            ##
            x = x.view(B, H, W, C)
            ## taken from window_partition
            x = x.view(B, H // winsize, winsize, W // winsize, winsize, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() #.view(-1, window_size, window_size, C)
            ###
            print(i,"2",windows.size())
            dfn = getattr(model, "dropoutlayer" + str(i+1))
            ## drop horizontal
            if W // winsize > 1:
                xh1 = windows[:, :, :, 0, :, :]
                xh1 = dfn(xh1)
                xv1 = windows[:, 0, :, :, :, :]
                xv1 = dfn(xv1)
                xh4 = windows[:, :, :, -1, :, :]
                xh4 = dfn(xh4)
                xv4 = windows[:, -1, :, :, :, :]
                xv4 = dfn(xv4)
                windows[:, :, :, 0, :, :] = xh1
                windows[:, :, :, -1, :, :] = xh4
                windows[:, 0, :, :, :, :] = xv1
                windows[:, -1, :, :, :, :] = xv4
                x = windows
            else:
                x = dfn(windows)
            ###
            x = x.view(B, H // winsize, W // winsize, winsize, winsize, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            print(i,"3",x.size())
            ##
            x = x.view(B, H * W, C)
            ##
    
    
    x = model.model.norm(x)  # B L C
    print(i,"4",x.size())
    #x = model.dropoutlayer4(x)
    x = x.view(B, H, W, C)
    output = x.permute(0, 3, 1, 2)

    #x = model.model.avgpool(x.transpose(1, 2))  # B C 1
    #x = torch.flatten(x, 1)

    #x = model.classifier(x)

###########################



print(output.shape)
heatmap = output.squeeze().sum(dim=0).cpu().numpy()
print(heatmap.shape)
#test_array = np.arange(100 * 100).reshape(100, 100)
# Result is saved tas `heatmap.png`
heatmap2d(imgpath[0][0],heatmap)
