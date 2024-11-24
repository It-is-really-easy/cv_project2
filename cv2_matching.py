import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像并转为灰度
img1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)  # 查询图像
img2 = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)  # 目标图像

# 初始化 SIFT 特征检测器
sift = cv2.SIFT_create()

# 检测关键点并计算描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用 BFMatcher 进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # L2 距离，启用交叉检查
matches = bf.match(des1, des2)

# 根据距离排序匹配结果（更小的距离表示更好的匹配）
matches = sorted(matches, key=lambda x: x.distance)

# 画出前 50 个匹配结果
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
plt.figure(figsize=(12, 8))
plt.imshow(matched_img)
plt.title('Keypoint Matching with OpenCV BFMatcher')
plt.axis('off')
plt.show()
