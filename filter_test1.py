#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gimpfu import *
import cv2
import numpy as np


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes();
    rl = gimp.Layer(image, name, image.width, image.height, 0, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()

def add_mask(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    return(cl_img)

def process(img, layer):
    imgmat = channelData(layer)
    print(imgmat)
    pred = add_mask(imgmat)
    createResultLayer(img, 'final_' + layer.name, pred)

# Регистрируем функцию в PDB
register(
    "python-fu-add-mask",
    "Добавление фильтра createCLAHE",
    "...dddd" ,
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