# Modifed from https://github.com/seoungwugoh/ivs-demo

import numpy as np
import os
import copy
import cv2
import glob

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import models
from dataset.range_transform import im_normalization


def images_to_torch(frames, device):
    frames = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float().unsqueeze(0)/255
    b, t, c, h, w = frames.shape
    for ti in range(t):
        frames[0, ti] = im_normalization(frames[0, ti])
    return frames.to(device)

def load_images(path, min_side=None):
    fnames = sorted(glob.glob(os.path.join(path, '*.jpg')))
    if len(fnames) == 0:
        fnames = sorted(glob.glob(os.path.join(path, '*.png')))
    frame_list = []
    for i, fname in enumerate(fnames):
        if min_side:
            image = Image.open(fname).convert('RGB')
            w, h = image.size
            new_w = (w*min_side//min(w, h))
            new_h = (h*min_side//min(w, h))
            frame_list.append(np.array(image.resize((new_w, new_h), Image.BICUBIC), dtype=np.uint8))
        else:
            frame_list.append(np.array(Image.open(fname).convert('RGB'), dtype=np.uint8))
    frames = np.stack(frame_list, axis=0)
    return frames

def load_masks(path, min_side=None):
    fnames = sorted(glob.glob(os.path.join(path, '*.png')))
    frame_list = []

    first_frame = np.array(Image.open(fnames[0]))
    binary_mask = (first_frame.max() == 255)

    for i, fname in enumerate(fnames):
        if min_side:
            image = Image.open(fname)
            w, h = image.size
            new_w = (w*min_side//min(w, h))
            new_h = (h*min_side//min(w, h))
            frame_list.append(np.array(image.resize((new_w, new_h), Image.NEAREST), dtype=np.uint8))
        else:
            frame_list.append(np.array(Image.open(fname), dtype=np.uint8))

    frames = np.stack(frame_list, axis=0)
    if binary_mask:
        frames = (frames > 128).astype(np.uint8)
    return frames