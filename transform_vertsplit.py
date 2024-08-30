# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Any, Dict

import numpy as np
import numpy.random
import torch
import torchvision
from PIL import Image

class RandomVerticalCut(object):

    def __init__(self, cutnum=3, randomvjit=0.1):
        self.cutnum = cutnum
        self.randomvjit = randomvjit

    def __call__(self, img):
        cutsize = int(img.size[0] / self.cutnum)
        cropsize = int(cutsize * (1 + self.randomvjit))
        
        randind = np.random.randint(2000) % self.cutnum

        for i in range(self.cutnum):
            if randind == i: ##
                y_start = cutsize*i
                y_end = y_start + cropsize
                break
            ###
        ###

        if y_end > img.size[0]:
            y_end = img.size[0]
            y_start = y_end - cropsize
        ###
        img = np.array(img)
        cropimg = np.copy(img[:, y_start:y_end, :])

        cropimg = Image.fromarray(cropimg)

        return cropimg, randind


class RandomHorizontalCut(object):

    def __init__(self, cutnum=3, randomvjit=0.1):
        self.cutnum = cutnum
        self.randomvjit = randomvjit

    def __call__(self, img):
        cutsize = int(img.size[0] / self.cutnum)
        cropsize = int(cutsize * (1 + self.randomvjit))
        
        randind = np.random.randint(2000) % self.cutnum

        for i in range(self.cutnum):
            if randind == i: ##
                x_start = cutsize*i
                x_end = x_start + cropsize
                break
            ###
        ###

        if x_end > img.size[1]:
            x_end = img.size[1]
            x_start = x_end - cropsize
        ###
        img = np.array(img)
        cropimg = np.copy(img[x_start:x_end, :, :])

        cropimg = Image.fromarray(cropimg)

        return cropimg, randind

class RandomVertHorCut(object):

    def __init__(self, cutnum=9, randomvjit=0.1):
        self.cutnum = cutnum
        self.randomvjit = randomvjit

    def __call__(self, img):
        cutsizex = int(img.size[1] / int(self.cutnum**0.5))
        cutsizey = int(img.size[0] / int(self.cutnum**0.5))
        cropsizex = int(cutsizex * (1 + self.randomvjit))
        cropsizey = int(cutsizey * (1 + self.randomvjit))
        
        randind = np.random.randint(2000) % self.cutnum

        for i in range(self.cutnum):
            if randind == i: ##
                x_start = cutsizex*(i%3)
                x_end = x_start + cropsizex
                y_start = cutsizey*(i//int(self.cutnum**0.5))
                y_end = y_start + cropsizey
                break
            ###
        ###

        if x_end > img.size[1]:
            x_end = img.size[1]
            x_start = x_end - cropsizex
        ###
        if y_end > img.size[0]:
            y_end = img.size[0]
            y_start = y_end - cropsizey
        img = np.array(img)
        cropimg = np.copy(img[:, y_start:x_start, x_start:x_end])

        cropimg = Image.fromarray(cropimg)

        return cropimg, randind


