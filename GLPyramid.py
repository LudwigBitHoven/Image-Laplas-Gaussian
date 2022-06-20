from Gaussian import *
from skimage.io import imread, imshow, imsave
import sys


"""
Однажды это станет склейкой
def stitching(img1, img2, mask):
    combine = img1 * mask + (1 - mask) * img2
    result = np.clip(sum(combine), 0, 1)
    return [50 * combine, result]
"""


lst = [imread("1.jpg"), imread("2.jpg")]
for j in range(1, 3):
    # Построение гауссовой пирамиды
    img = lst[j - 1]
    temp_img = img.copy()
    for i in range(1, 6):
        res = gauss_matrix(0.66)
        temp_img = rgb_image_blur(temp_img, res)
        imsave(f"gaussian_{j}_{i}.jpg", temp_img)
    img = lst[j - 1]

    # построение лапласовской пирамиды
    imsave(f"pyramid_{j}_1.jpg", img - imread(f"gaussian_{j}_1.jpg"))
    for i in range(2, 5):
        temp_img = imread(f"gaussian_{j}_{i}.jpg")
        temp_img = temp_img - imread(f"gaussian_{j}_{i + 1}.jpg")
        imsave(f"pyramid_{j}_{i}.jpg", temp_img)

    # построение карты частот гауссовых изображений
    for i in range(1, 6):
        temp_img = imread(f"gaussian_{j}_{i}.jpg")
        temp_img = rgb_image_freq(temp_img)
        imsave(f"freq_{j}_{i + 1}.jpg", temp_img.astype("uint8"))

temp_img = imread("mask_1.jpg")
for i in range(1, 5):
    res = gauss_matrix(0.66)
    temp_img = rgb_image_blur(temp_img, res)
    imsave(f"gaussian_3_{i}.jpg", temp_img)
sys.exit()
