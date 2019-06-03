一个小项目，从人脸图像判断是否疲劳驾驶。
# 算法

## 1. 基于分类方法

detect_drowsiness_cls  
使用的是ResNetV2分类，加入attention模块。 
TODO: 模型调整为多任务模型。

## 2. 基于landmark方法

detect_drowsiness_dlib
从眼部landmark的相对位置确定眼睛的睁闭。 
准确度较差。
