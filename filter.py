#!/usr/bin/env python
# -*- coding: utf-8 -*-

# How to run this script:
# 1. Without GIMP python-console.
# Place the file filter.py in a folder C:\Program Files\GIMP 2\gim\lib\gimp\2.0\plug-ins and then open GIMP. 
# In the upper panel there will be a 'Filter' button that will allow you to apply the plugin to the image.

# 2. With python-console in debugging mode.
# Comment the register and main block. After that, put file filter.py in folder 'C:\Users\.gimp-2.8\plug-ins'. 
# Then open python-console and enter the commands: 
# >>> import filter
# >>> image = gimp.image_list()[0] 
# >>> filter.process(image , image.active_layer)

from gimpfu import *
import cv2
import numpy as np


def channelData(layer):  
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  
    bpp = region.bpp
    f = np.frombuffer(pixChars, dtype = np.uint8).reshape(layer.height, layer.width, bpp)
    return f


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes()
    rl = gimp.Layer(image, name, image.width, image.height, 0, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()


def add_mask_CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    return cl_img

def add_mask_Gaussian(image):
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), sigmaX = 0)
    return gaussian_blur

def add_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def process(img, layer, mask):
    imgmat = channelData(layer)
    if mask == "1":
	imgmat = cv2.cvtColor(imgmat, cv2.COLOR_BGR2GRAY )
    	pred = add_mask_CLAHE(imgmat)
    	pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
    if mask == "2":
	pred = add_mask_Gaussian(imgmat)
    if mask == "3":
	pred = add_unsharp_mask(imgmat)
    createResultLayer(img, 'final' , pred)


register(
    "python-fu-add-mask",
    "Adding filters: 1 - CLAHE filter, 2 - Gaussian smoothing, 3 - unsharp mask",
    "createCLAHE filter" ,
    "Natalia Shmigelskaya",
    "Natalia Shmigelskaya (shmigelskaya.no@phystech.edu)",
    "21.06.2022",
    "Add mask ",
    "*",
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_STRING, "mask", "1 - CLAHE, 2 - Gaussian, 3 - Unsharp", "1")],
    [],
    process,
    menu = "<Image>/Filter/")

main()
