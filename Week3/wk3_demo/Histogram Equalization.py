#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist Histogram Equalization
Function prototype: equalizeHist(src, dst=None)
src: Source image (single-channel image)
dst: Default is None
'''

# Read the grayscale image 灰度图像直方图均衡化
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image_gray", gray)

# Histogram equalization of the grayscale image 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# Histogram of the equalized image 直方图均衡化后的直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])  #计算直方图 # 0:通道索引 # None:掩膜 # 256:直方图尺寸 # 0,256:像素值范围

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)


'''
# Histogram equalization of a color image 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# For color image equalization, each channel needs to be processed separately 彩色图像均衡化，需要分别处理每个通道
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# Merge each channel 合并每个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''
