import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

# ***** WCZYTANIE ZDJĘCIA *****
img = cv2.imread("r2.png")

# ***** ZDEFINIOWANIE KRAWĘDZI X i Y *****
# ***** SOBEL *****
kernel_Sobel_x = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]])
kernel_Sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]])
# ***** ROBERTS *****
kernel_Roberts_x = np.array([
    [1, 0],
    [0, -1]
    ])
kernel_Roberts_y = np.array([
    [0, -1],
    [1, 0]
    ])
# ***** POŁĄCZENIE ZDJĘĆ X i Y *****
sob_x = cv2.filter2D(img, cv2.CV_64F, kernel_Sobel_x)
sob_y = cv2.filter2D(img, cv2.CV_64F, kernel_Sobel_y)
sob = sob_x + sob_y
rob_x = cv2.filter2D(img, cv2.CV_64F, kernel_Roberts_x)
rob_y = cv2.filter2D(img, cv2.CV_64F, kernel_Roberts_y)
rob = rob_x + rob_y


plt.subplot(4,2,1),plt.imshow(img,cmap = 'gray')
plt.title('ORYGINAŁ'), plt.xticks([]), plt.yticks([])
plt.subplot(4,2,3),plt.imshow(sob_x,cmap = 'gray')
plt.title('SOB_X'), plt.xticks([]), plt.yticks([])
plt.subplot(4,2,5),plt.imshow(sob_y,cmap = 'gray')
plt.title('SOB_Y'), plt.xticks([]), plt.yticks([])
plt.subplot(4,2,7),plt.imshow(sob,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(4,2,4),plt.imshow(rob_x,cmap = 'gray')
plt.title('ROB_X'), plt.xticks([]), plt.yticks([])
plt.subplot(4,2,6),plt.imshow(rob_y,cmap = 'gray')
plt.title('ROB_Y'), plt.xticks([]), plt.yticks([])
plt.subplot(4,2,8),plt.imshow(rob,cmap = 'gray')
plt.title('ROB'), plt.xticks([]), plt.yticks([])
plt.show()

