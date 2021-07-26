# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib

#读取图像
img = cv2.imread('Lena.png', 0)

#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
res1= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))

#傅里叶逆变换
ishift = np.fft.ifftshift(dftshift)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']

#显示图像
plt.subplot(131), plt.imshow(img, 'gray'), plt.title(u'(a)原始图像')
plt.axis('off')
plt.subplot(132), plt.imshow(res1, 'gray'), plt.title(u'(b)傅里叶变换处理')
plt.axis('off')
plt.subplot(133), plt.imshow(res2, 'gray'), plt.title(u'(b)傅里叶变换逆处理')
plt.axis('off')
plt.show()
