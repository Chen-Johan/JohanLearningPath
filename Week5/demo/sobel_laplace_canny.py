#!/usr/bin/env python
# encoding=utf-8

import cv2  
import numpy as np  
from matplotlib import pyplot as plt  

img = cv2.imread("lenna.png", 1)  

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  

'''
Sobel算子
Sobel函数的原型如下：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
前四个是必选参数：
第一个是需要处理的图像
第二个是图像的深度，-1表示输出图像的深度与原图像一致
dx和dy表示导数的方向，0表示不计算导数，1表示计算一阶导数，2表示计算二阶导数
后面是可选参数：
dst是目标图像
ksize是Sobel算子的大小，必须为1、3、5或7
scale是缩放导数的比例常数，默认情况下没有伸缩系数
delta是一个可选的增量值，将该值加到结果中，默认情况下没有加任何值
borderType是判断图像边界的模式，默认值为cv2.BORDER_DEFAULT
'''

img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # x方向
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # y方向

# Laplace 算子  
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)  

# Canny 算子  
img_canny = cv2.Canny(img_gray, 100, 150)  

plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")  
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")  
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")  
plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("Laplace")  
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")  
plt.show()
