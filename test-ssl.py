# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_swin, ft_net_swinv2, ft_net_swinlarge, ft_net_swinv2large
from model import ft_net_swinv2large_384
from utils import fuse_all_conv_bn
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
####
parser.add_argument('--resx',default=-1, type=int,help='resolution x')
parser.add_argument('--resy',default=-1, type=int,help='resolution y')



opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
opt.fp16 = config['fp16'] 
opt.stride = config['stride']
if 'use_swin' in config:
    opt.use_swin = config['use_swin']
if 'use_swinlarge' in config:
    opt.use_swinlarge = config['use_swinlarge']
if 'use_swinv2' in config:
    opt.use_swinv2 = config['use_swinv2']
if 'use_swinv2large' in config:
    opt.use_swinv2large = config['use_swinv2large']

opt.use_swinv2large_res384 = False
if 'use_swinv2large_res384' in config:
    opt.use_swinv2large_res384 = config['use_swinv2large_res384']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

if 'linear_num' in config:
    opt.linear_num = config['linear_num']
if 'numcut' in config:
    opt.numcut = config['numcut']

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
h, w = 224, 224
if opt.use_swinv2 or opt.use_swinv2large:
    h, w = 256, 256
if opt.use_swinv2large_res384:
    h, w = 384, 384

if opt.resx > 0:
    h, w = opt.resy, opt.resx


data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])



data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    if opt.linear_num <= 0:
        if opt.use_swin or opt.use_swinv2:
            opt.linear_num = 1024
        elif opt.use_swinv2large or opt.use_swinv2large_res384:
            opt.linear_num = 1536
        else:
            opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    s_x = (input_img.size(3)-w)//2
                    s_y = (input_img.size(2)-h)//2
                    input_img = input_img[:, :, s_y:s_y+h, s_x:s_x+w]
                #print(input_img.size())
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        
        #if iter == 0:
        #    features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        features = torch.cat((features,ff.data.cpu()), 0)
        #start = iter*opt.batchsize
        #end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        #features[ start:end, :] = ff
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        #label = filename.split(".")[0].split("_")[0] #[0:4]    # for PKU-Sketch-REID
        label = path.split("/")[-2] #[0:4]                  # for RegDB
        #label = filename[0:4]
        #camera = filename.split('c')[1]
        #camera = (1 if len(filename.split(".")[0].split("_")) > 1 else 0)  # for PKU-Sketch-REID
        camera = 0 if path.find('query') >= 0 else 1                        # for RegDB
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        #camera_id.append(int(camera[0]))
        camera_id.append(int(camera))      # for PKU-Sketch-REID / RegDB
        print(path, v, label, camera)
    ###

    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_swin:
    model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num, imgsize=(opt.resx if opt.resx > 0 else w), cutnum=opt.numcut)
elif opt.use_swinlarge:
    model_structure = ft_net_swinlarge(opt.nclasses, linear_num=opt.linear_num, imgsize=(opt.resx if opt.resx > 0 else w), cutnum=opt.numcut)
elif opt.use_swinv2:
    model_structure = ft_net_swinv2(opt.nclasses, (h,w),  linear_num=opt.linear_num, cutnum=opt.numcut)
elif opt.use_swinv2large:
    model_structure = ft_net_swinv2large(opt.nclasses, (h,w),  linear_num=opt.linear_num, cutnum=opt.numcut)
elif opt.use_swinv2large_res384:
    model_structure = ft_net_swinv2large_384(opt.nclasses, (h,w),  linear_num=opt.linear_num, cutnum=opt.numcut)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num, cutnum=opt.numcut)


#if opt.fp16:
#    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()


#print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
#model = fuse_all_conv_bn(model)

# We can optionally trace the forward method with PyTorch JIT so it runs faster.
# To do so, we can call `.trace` on the reparamtrized module with dummy inputs
# expected by the module.
# Comment out this following line if you do not want to trace.
#dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
#model = torch.jit.trace(model, dummy_forward_input)

print(model)
# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])
    if opt.multi:
        mquery_feature = extract_feature(model,dataloaders['multi-query'])
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.name)
result = './model/%s/result.txt'%opt.name
os.system('python evaluate_gpu.py | tee -a %s'%result)

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)
