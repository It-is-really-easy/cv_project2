import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img1 = cv2.imread('1.png')  # 左侧图像
img2 = cv2.imread('2.png')  # 右侧图像

# 转为灰度图像
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 提取 SIFT 特征点和描述符
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 使用 BFMatcher 匹配特征点
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 按距离排序匹配点
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点的位置
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 计算单应性矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 透视变换图像
h, w, _ = img1.shape
img1_warped = cv2.warpPerspective(img1, M, (w + img2.shape[1], max(h, img2.shape[0])))
img1_warped[0:img2.shape[0], 0:img2.shape[1]] = img2

# 显示拼接结果
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB))
plt.title("Image Stitching using OpenCV SIFT")
plt.axis('off')
plt.show()
