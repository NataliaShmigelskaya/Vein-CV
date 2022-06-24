#!/usr/bin/env python
# -*- coding: utf-8 -*-

# How to run this script:
# 1. Without GIMP python-console.
# Place the file filter_test1.py in a folder C:\Program Files\GIMP 2\gim\lib\gimp\2.0\plug-ins and then open GIMP. 
# In the upper panel there will be a 'Filter' button that will allow you to apply the plugin to the image.

# 2. With python-console in debugging mode.
# Comment the register and main block. After that, put file filter_test1.py in folder 'C:\Users\.gimp-2.8\plug-ins'. 
# Then open python-console and enter the commands: 
# >>> import filter_test1
# >>> image = gimp.image_list()[1] 
# >>> filter_test1.process(image , image.active_layer)


from gimpfu import *
import cv2
import numpy as np


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype = np.uint8).reshape(layer.height, layer.width, bpp)


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes()
    rl = gimp.Layer(image, name, image.width, image.height, 0, 100, NORMAL_MODE)
    # rl = gimp.layer(image,name,image.width,image.height,image.active_layer.type,100,NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()


def add_mask(image):
    #changed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    return cl_img

def process(img, layer):
    imgmat = channelData(layer)
    imgmat = imgmat[:,:,0]

    #imgmat = imgmat.astype('int32')
    #print(imgmat.dtype)
    #a = np.zeros_like(imgmat)
    # print(imgmat)
    pred = add_mask(imgmat)
    #pred_new = pred[:,:,np.newaxis]
    pred_new = np.repeat(pred[:, :, np.newaxis], 3, axis=2)

    createResultLayer(img, 'final' , pred_new)


register(
    "python-fu-add-mask",
    "Добавление фильтра createCLAHE",
    "createCLAHE filter" ,
    "Наталья Шмигельская",
    "Наталья Шмигельская (shmigelskaya.no@phystech.edu)",
    "21.06.2022",
    "Добавить маску ",
    "*",
    [   (PF_IMAGE, "image", "Исходное изображение", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None)],
    [],
    process,
    menu = "<Image>/Filter/")

main()
