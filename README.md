# SIFT 算法实现与图像拼接项目

## 项目简介

本项目是计算机视觉课程的一个技术报告，实现了尺度不变特征转换（Scale-Invariant Feature Transform, SIFT）算法，并应用于图像拼接。SIFT 算法由 David Lowe 提出，用于检测和描述图像中的局部特征，对旋转、尺度缩放、光照变化等具有强鲁棒性。

## 实现细节

### 算法特点

- **尺度空间不变性**：通过高斯差分（DoG）检测显著区域，对旋转、尺度缩放具有强鲁棒性。
- **对环境影响的适应性**：在光照变化、噪声干扰、遮挡和杂物场景中，SIFT 提取的关键点仍然稳定。
- **高独特性和信息量**：每个特征点的描述符（通常为 128 维向量）包含丰富的局部信息，适合用于特征匹配任务。
- **多样性与冗余性**：图像中即使只有少数目标，也能生成大量特征点，为后续匹配提供更多冗余信息。
- **优化后的实时性**：SIFT 算法经过优化可用于实时处理，尤其在嵌入式环境中的轻量化版本逐渐发展成熟。

### 应用场景

- **图像拼接**：通过提取并匹配图像之间的 SIFT 特征点，完成多张图像的无缝拼接（如全景图生成）。
- **目标识别与跟踪**：在视频监控中，SIFT 可用于在动态场景中检测和跟踪目标。
- **增强现实（AR）**：通过实时匹配图像特征，实现虚拟对象与现实场景的叠加。
- **机器人导航**：SIFT 特征帮助机器人理解周围环境，实现路径规划和目标识别。

### 实验结果

实验部分展示了 SIFT 算法从原理到实现的全面探索，包括高斯差分金字塔的构建、关键点的提取和描述符的生成。通过与 OpenCV 的 SIFT 实现进行对比，分析了匹配点分布对拼接结果的影响，并探讨了如何平衡精度和效率。

## 使用方法

本项目包含两个文件：

1. **project2-实验报告.pdf**：详细描述了 SIFT 算法的背景、原理、实现方法和实验结果。
2. **pysift.py**：Python 代码实现，包括 SIFT 特征提取和图像拼接功能。

### 安装依赖

```bash
pip install numpy opencv-python matplotlib
```

### 运行代码

```bash
python connecting.py
python matching.py
```

### 额外功能

如果想要尝试以下额外功能，需要取消注释以下几行代码：

- **绘制关键点**：取消注释 `pysift.py` 中的第 14-19 行代码，可以在图像上绘制出检测到的 SIFT 关键点。
- **显示直方图**：取消注释 `pysift.py` 中的第 252-266 行代码，可以显示关键点梯度方向的直方图。

## 总结与展望

通过本项目的实现，深入理解了 SIFT 算法的设计思想和实现细节，并探索了其在实际场景中的应用潜力。未来可以通过利用硬件加速或简化计算步骤进一步提升 SIFT 的效率，并尝试将 SIFT 扩展到更复杂的场景，如多图拼接或实时动态目标检测。

## 参考文献

1. Lowe, D. G. "Distinctive image features from scale-invariant keypoints." International Journal of Computer Vision 60.2 (2004): 91-110.
2. Mikolajczyk, K., & Schmid, C. "A performance evaluation of local descriptors." IEEE Transactions on Pattern Analysis and Machine Intelligence 27.10 (2005): 1615-1630.
3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. "ImageNet classification with deep convolutional neural networks." Communications of the ACM 60.6 (2017): 84-90.
4. Bay, H., Tuytelaars, T., & Van Gool, L. "SURF: Speeded up robust features." European Conference on Computer Vision. Springer, 2006.
5. Rublee, E., et al. "ORB: An efficient alternative to SIFT or SURF." International Conference on Computer Vision (2011): 2564-2571.
6. [PythonSIFT](https://github.com/rmislam/PythonSIFT) - 本项目参考的开源 SIFT 实现。

## 更新
该项目已上传https://github.com/It-is-really-easy/cv_project2
