import numpy as np
import cv2
import pysift
from matplotlib import pyplot as plt

# 读取图像
img1 = cv2.imread('1.png')  # 左侧图像
img2 = cv2.imread('2.png')  # 右侧图像

# 转为灰度图像
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 提取 SIFT 特征点和描述符
kp1, des1 = pysift.computeKeypointsAndDescriptors(gray1)
kp2, des2 = pysift.computeKeypointsAndDescriptors(gray2)

# 使用 FLANN 匹配器匹配特征点
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 低比值测试筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 计算单应性矩阵
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 透视变换图像
    h, w, _ = img1.shape
    img1_warped = cv2.warpPerspective(img1, M, (w + img2.shape[1], max(h, img2.shape[0])))
    img1_warped[0:img2.shape[0], 0:img2.shape[1]] = img2

    # 显示拼接结果
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB))
    plt.title("Image Stitching using pysift")
    plt.axis('off')
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), 10))
