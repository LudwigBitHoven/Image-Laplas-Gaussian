import math
import numpy as np
import scipy.signal
import cv2
from skimage import img_as_float, img_as_ubyte
from numpy.fft import fft2, fftshift


def gauss_func(sigma, x, y):
    a = 2 * math.pi * sigma ** 2
    power = (-(x ** 2) - (y ** 2)) / (2 * sigma ** 2)
    b = math.exp(power)
    return b / a


def gauss_matrix(s):
    x = int(round(s * 6) + 1)
    res = []
    for i in range(-x // 2 + 1, x // 2 + 1):
        res.append([gauss_func(s, i, j) for j in range(x // 2, -x // 2, -1)])
    return np.array(res) / np.sum(res)


def rgb_image_blur(img, res):
    r = scipy.signal.convolve2d(img[:, :, 0], res, mode='same')
    g = scipy.signal.convolve2d(img[:, :, 1], res, mode='same')
    b = scipy.signal.convolve2d(img[:, :, 2], res, mode='same')
    return np.dstack((r, g, b)).astype(np.uint8)


def rgb_image_freq(img):
    img_f = img_as_float(img)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    freq = 20 * np.log(1 + abs(fftshift(fft2(Y))))
    return freq


def vignette(img):
    border = [[img[0, i, :] for i in range(img.shape[1])] for j in range(2)]
    img = np.insert(img, 0, border, axis=0)
    border = [[img[img.shape[0] - 1, i, :] for i in range(img.shape[1])] for j in range(2)]
    img = np.insert(img, img.shape[0], border, axis=0)
    border = [[img[i, 0, :] for i in range(img.shape[0])] for j in range(2)]
    img = np.insert(img, 0, border, axis=1)
    border = [[img[i, img.shape[1] - 1, :] for i in range(img.shape[0])] for j in range(2)]
    img = np.insert(img, img.shape[1], border, axis=1)
    return img
