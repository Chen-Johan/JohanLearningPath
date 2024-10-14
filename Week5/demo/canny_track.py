#!/usr/bin/env python
# encoding=utf-8

'''
Canny边缘检测：优化的代码
'''
import cv2
import numpy as np 

def CannyThreshold(lowThreshold):  
    #detected_edges = cv2.GaussianBlur(gray,(3,3),0) #高斯滤波 
    detected_edges = cv2.Canny(gray,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #边缘检测

     #将原始图像与检测到的边缘相结合
     #位运算“与”操作：对每个像素，只有当两个图像的对应位都为1时，结果图像的对应位才为1
     #src1和src2表示要进行位运算的两幅图像
     #mask 是可选操作参数，指定操作的掩码，只在掩码对应位置的像素进行位运算
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  
    cv2.imshow('canny result',dst)  

lowThreshold = 0  
max_lowThreshold = 100  
ratio = 3  
kernel_size = 3  

img = cv2.imread('lenna.png')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #转换彩色图像为灰度图像

cv2.namedWindow('canny result')  

#创建滑动条
'''
以下是创建滑动条的cv2.createTrackbar()函数
有5个参数，分别解释如下：
第一个参数是滑动条的名称
第二个参数是滑动条所属窗口的名称
第三个参数是滑动条的默认值
第四个参数是滑动条的最大值(0~count)
第五个参数是滑动条变化时调用的回调函数
'''
cv2.createTrackbar('Min threshold','canny result',lowThreshold, max_lowThreshold, CannyThreshold)  

CannyThreshold(0)  # initialization  
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()
