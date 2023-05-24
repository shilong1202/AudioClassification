import cv2
import os
from PIL import Image, ImageEnhance

# input_file = "path/to/input/file.png"
# output_file = "path/to/output/file.png"

path_HC = 'E:\softSpace\PycharmSpaces\pytorch37\\audioClassification\input\Spirals\\training\HC'
path_PD = 'E:\softSpace\PycharmSpaces\pytorch37\\audioClassification\input\Spirals\\training\PD'
import numpy as np
import random


def turn(file_path):
    output_file = file_path.replace('.png', '_5.png')
    print(output_file)
    with Image.open(file_path) as im:
        # 图像转为numpy
        array = np.asarray(im)
        # 定义噪声参数
        mean = 0
        std_dev = 25  # 标准差
        # 生成高斯噪音图像
        noise = np.zeros(array.shape, dtype=np.uint8)  # 全零矩阵，用于存储噪声像素
        cv2.randn(noise, mean, std_dev)  # 生成高斯分布的噪声像素
        noisy_array = np.clip((array + noise), 0, 255).astype(np.uint8)  # 添加噪声像素并限制像素值在 0~255 范围内
        # 将噪声图像转换回 PIL.Image 对象
        noisy_image = Image.fromarray(noisy_array)

        # noisy_image.show()

        noisy_image.save(output_file)

        # rotated_im = im.transpose(method=Image.FLIP_TOP_BOTTOM)
    # rotated_im.save(output_file)


def enhane(file_path):
    output_file = file_path.replace('.png', '_6.png')
    print(output_file)
    with Image.open(file_path) as image:
        brightness_factor = 1.2
        contrast_factor = 1.5
        saturation_factor = 1.2

        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)

        image.save(output_file)



for file in os.listdir(path_HC):
    enhane(path_HC + '\\' + file)

for file in os.listdir(path_PD):
    enhane(path_PD + '\\' + file)
