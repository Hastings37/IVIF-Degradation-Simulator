import cv2
import os
import numpy as np

# 定义退化等级参数字典
DEGRADATION_LEVELS = {
    "clean":    {"alpha": 1.00, "beta": 0},    # 无退化：完全保留原图信息，无附加干扰
    "slight":   {"alpha": 0.85, "beta": 15},
    "moderate": {"alpha": 0.65, "beta": 45},
    "severe":   {"alpha": 0.40, "beta": 90},
    "extreme":  {"alpha": 0.15, "beta": 130}
}

# 概率分布: [Clean 15%, Slight 20%, Moderate 30%, Severe 25%, Extreme 10%]
LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]


def adjust_contrast(image, level=None):
    """
    使用线性变换调整图像的对比度 (模拟红外低对比度退化)。

    参数:
        image: 输入的灰度图像 (uint8)
        level:
            - None: 默认为 'moderate'
            - 'random': 按概率随机选择等级
            - 指定字符串: 'clean', 'slight', ...

    返回:
        adjusted_image: 处理后的图像
        selected_level: 实际使用的等级名称 (str)
    """

    # 1. 确定等级逻辑
    if level is None:
        # 默认情况
        selected_level = "moderate"
    elif level == "random":
        # 随机情况：从概率分布中采样
        selected_level = np.random.choice(LEVEL_KEYS, p=LEVEL_PROBS)
    else:
        # 指定情况：检查是否在字典中，不在则默认 moderate
        if level in DEGRADATION_LEVELS:
            selected_level = level
        else:
            # print(f"警告: 未知的等级 '{level}'，已回退到 moderate")
            selected_level = "moderate"

    # 2. 获取对应的参数
    params = DEGRADATION_LEVELS[selected_level]
    alpha = params["alpha"]
    beta = params["beta"]

    # 3. 执行线性变换 (new = alpha * image + beta)
    # cv2.convertScaleAbs 会自动处理截断(0-255)和类型转换(uint8)，非常方便
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image



