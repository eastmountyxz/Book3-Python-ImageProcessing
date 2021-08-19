# -*- coding:utf-8 -*-
import cv2
import numpy as np

#读取图片
img = cv2.imread("Lena.png")

#定义200×200矩阵 3对应BGR
face = np.ones((200, 200, 3))

#显示原始图像
cv2.imshow("Demo", img)

#显示ROI区域
face = img[150:350, 150:350]
cv2.imshow("face", face)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
