import cv2, imutils
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
from fastprogress import progress_bar
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt

def minmaxscaler(x): return (x - x.min()) / (x.max() - x.min())

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def sharpen(img):
    # SHARPEN
    kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')
    return cv2.filter2D(img, -1, kernel_sharp)

def bounds_per_dimension(ndarray):
#     https://stackoverflow.com/questions/54567986/python-numpy-remove-empty-zeroes-border-of-3d-array/54570293
    return map(
        lambda e: range(e.min(), e.max() + 1),
        np.where(ndarray != 0)
    )

def zero_trim_ndarray(ndarray):
    return ndarray[np.ix_(*bounds_per_dimension(ndarray))]

def resizeAndPad(img, size, padColor=0):
    # https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv
    h, w = img.shape[:2]
    sh, sw = size
    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

# def preprocess_image(img, min_contour_area=10, out_size=(256,256)):
#     pad_all = 5
#     im = minmaxscaler(img)
#     im = 1 - img.copy()
#     mulp = round(1024 / max(img.shape))
#     im = cv2.resize(im, (im.shape[1]*mulp, im.shape[0]*mulp), interpolation=cv2.INTER_NEAREST)
# #     print(img.shape, im.shape)
#     im = sharpen(np.stack((im, im, im), axis=-1))
#     im = cv2.dilate(im, kernel=np.ones((3,3)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#     im = (im * 255).astype(np.uint8) # black background
#     # find contours in the thresholded image
#     cnts = cv2.findContours(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     # print("[INFO] {} unique contours found".format(len(cnts)))
#     # loop over the contours
#     for (i, c) in enumerate(cnts):
#         if cv2.contourArea(c) < min_contour_area:
#             cv2.fillPoly(im, [c], color=(255, 0, 0))
#     im = im[...,-1]
#     if im.shape != img.shape:
#         im = cv2.resize(im, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
#     im = zero_trim_ndarray(im)
#     im = cv2.copyMakeBorder(im, 0, 0, pad_all, pad_all, borderType=cv2.BORDER_CONSTANT, value=0)
#     im_scaled = resizeAndPad(im, out_size, 0)
#     return im_scaled

def preprocess_image(imgpath, min_contour_area=10, out_size=(256,256)):
    img = Image.open(imgpath)
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.MaxFilter(3))
    pad_all = 5
    w, h = img.size
    im = np.array(img)
    # find contours in the thresholded image
    cnts = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print("[INFO] {} unique contours found".format(len(cnts)))
    # loop over the contours
    im = np.stack((im, im, im), axis=-1)
    for (i, c) in enumerate(cnts):
        if cv2.contourArea(c) < min_contour_area:
            cv2.fillPoly(im, [c], color=(255, 0, 0))
    im = im[...,-1]
    if im.shape != img.size:
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
    im = zero_trim_ndarray(im)
    im = cv2.copyMakeBorder(im, 0, 0, pad_all, pad_all, borderType=cv2.BORDER_CONSTANT, value=0)
    im_scaled = resizeAndPad(im, out_size, 0)
    return Image.fromarray(im_scaled)