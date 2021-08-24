#encoding:utf-8
#By:Eastmount CSDN 2020-11-12
import cv2  
import numpy as np  
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('miao.png')

#获取图像高度和宽度
height = img.shape[0]
width = img.shape[1]

#创建一幅图像
grayimg = np.zeros((height, width, 3), np.uint8)
print grayimg

#图像平均灰度处理方法
for i in range(height):
    for j in range(width):
        #灰度值为RGB三个分量的平均值
        gray = (int(img[i,j][0]) + int(img[i,j][1]) + int(img[i,j][2]))  /  3
        grayimg[i,j] = np.uint8(gray)

#显示图像
cv2.imshow("src", img)
cv2.imshow("gray", grayimg)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
