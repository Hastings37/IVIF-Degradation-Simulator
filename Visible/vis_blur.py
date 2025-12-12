import os
from random import randint  # 从 a b 中间随机生成数字内容；

import cv2
import numpy as np

DEGRADATION_LEVELS = {
    "clean": {"blur_length": 0},  # 无模糊
    "slight": {"blur_length": 3},  # 极轻微
    "moderate": {"blur_length": 7},  # 明显
    "severe": {"blur_length": 12},  # 严重
    "extreme": {"blur_length": 17}  # 极度 (已修改为20)
}

LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
# LEVEL_PROBS = [0.15, 0.20, 0.30, 0.25, 0.10]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]


def add_motion_blur(img, level=None):
    """
    添加运动模糊 (Motion Blur).
    只返回处理后的图像 (numpy.ndarray).
    """

    # --- 1. 确定等级 ---
    if level is None:
        selected_level = "moderate"
    elif level == "random":
        # 依赖外部定义的 LEVEL_KEYS 和 LEVEL_PROBS
        selected_level = np.random.choice(LEVEL_KEYS, p=LEVEL_PROBS)
    else:
        if level not in DEGRADATION_LEVELS:
            raise ValueError(f"未知的退化等级: {level}")
        selected_level = level

    # --- 2. 获取参数 ---
    # 假设 DEGRADATION_LEVELS 已经包含 blur_length
    params = DEGRADATION_LEVELS[selected_level]
    length = params.get("blur_length", 0)

    # Clean 或 长度无效时直接返回
    if length <= 0:
        return img

    # --- 3. 确定模糊角度 (普遍范围) ---
    # 限制在 -30 到 +30 度之间，模拟常见的手抖或水平运动
    angle = np.random.randint(-30, 31)

    # --- 4. 生成模糊核 ---
    # 确保核大小足够容纳 length，且为奇数 (cv2要求)
    ksize = max(length, 3)
    if ksize % 2 == 0:
        ksize += 1

    # 初始化黑色的核
    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    # 计算中心
    center = ksize // 2

    # 在核的中心画一条水平线
    # 长度由 length 控制
    half_len = length // 2
    kernel[center, center - half_len: center + half_len + 1] = 1.0

    # 旋转核 (生成指定角度的模糊)
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))

    # 归一化 (保证亮度不变)
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel /= kernel_sum

    # --- 5. 应用滤波 ---
    blurred_img = cv2.filter2D(img, -1, kernel)

    return blurred_img
