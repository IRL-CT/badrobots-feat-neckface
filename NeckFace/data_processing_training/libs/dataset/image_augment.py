import os
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np
import math
import random

def cal_scale(row, col, angel):
    theta0 = math.atan(row/col)
    theta1 = theta0 + abs(angel / 180 * math.pi)
    return math.sin(theta1) / math.sin(theta0)

def random_rotate(img, angel, lower, upper, odd, odd2):
    if random.random() > odd: return img
    else:
        rows,cols,_ = img.shape
        rotate = 0#random.uniform(-angel, angel)
        print(rotate)
        if random.random() > odd2: scale = random.uniform(lower, upper)
        else: scale = 1
        #scale = cal_scale(rows,cols,rotate)
        scale = 0.7
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),rotate,1)
        M=M*scale
        dst = cv2.warpAffine(img,M,(int(cols),int(rows)))
        return dst

def random_move(img, delta_x, delta_y, odd):
    if random.random() > odd: return img
    else:
        rows,cols,_ = img.shape
        movex = random.randrange(-delta_x, delta_x)
        movey = random.randrange(-delta_y, delta_y)
        #print(movex, movey, rows, cols)
        if movex > 0:
            downx, upx = 0, rows - movex
        else: downx, upx = -movex, rows
        if movey > 0:
            downy , upy = 0, cols - movey
        else: downy , upy = -movey, cols
        dst = img[downx : upx, downy : upy, :]
        last = cv2.resize(dst, (cols, rows))
        return last

def random_trans(image, lower, upper, odd1, delta_x, delta_y, odd2, angel, odd3):
    rows,cols,_ = image.shape
    #print(image.shape)
    if random.random() > odd1: scale = 1
    else:   scale = random.uniform(lower, upper)
    p1 = cv2.resize(image, None, fx= scale, fy = scale)
    if random.random() > odd2: rotate = 0
    else:  rotate = random.uniform(-angel, angel)
    rows1,cols1,_ = p1.shape   
    M = cv2.getRotationMatrix2D(((cols1-1)/2.0,(rows1-1)/2.0),rotate,1)
    if random.random() > odd3:
        movex = 0
        movey = 0
    else:
        movex = random.randrange(-delta_x, delta_x)
        movey = random.randrange(-delta_y, delta_y)
    M[0,2] += movex + (cols-cols1)//2

    M[1,2] += movey + (rows-rows1)//2 
    dst = cv2.warpAffine(p1, M, (int(cols),int(rows)))
    
    return dst


def augment(src, config):
    if config["if_augment"] == 0: return src
    #print("augment!!!")
    move_config = config["move"]
    rotate_config = config["rotate"]
    scale_config = config["scale"]
    for i in range(0, src.shape[0]):
        image = src[i].numpy()
        image = np.swapaxes(image,0,2)
        #cv2.imshow('src', image)
        #image = random_move(image, move_config[0], move_config[1], move_config[2])
        #image = random_rotate(image, rotate_config[0], scale_config[0], scale_config[1] ,rotate_config[1], scale_config[2])
        image = random_trans(image, scale_config[0], scale_config[1], scale_config[2], move_config[0], move_config[1], move_config[2], rotate_config[0], rotate_config[1] )
        #cv2.imshow('dst', image)
        image = np.swapaxes(image,0,2)
        src[i] = torch.Tensor(image)
        #cv2.waitKey(0)
    return src

def test():
    image = cv2.imread("test.png")
    cv2.imshow('roqt',image )
    imager = random_trans(image, 0.8, 1.2, 1, 10, 10, 1, 10, 1 )
    cv2.imshow('rot',imager )
    cv2.waitKey(0)

