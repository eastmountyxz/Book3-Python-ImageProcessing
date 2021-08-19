# -*- coding:utf-8 -*-
# By: Eastmount CSDN 2021-01-26
import cv2
import numpy as np

#读取图片
img = cv2.imread("Lena.png")

#拆分通道
b, g, r = cv2.split(img)

#合并通道
m = cv2.merge([b, g, r])
cv2.imshow("Merge", m)
           
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
