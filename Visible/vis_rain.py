import numpy as np
import cv2
import os

# ================= 1. 全局配置区域 =================

LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]

# 统一参数字典
# 统一参数字典 (雨天参数 - Extreme 已调小)
DEGRADATION_LEVELS = {
    "clean": {
        "rain_value": 0, "rain_length": 0, "rain_alpha": 1.0, "rain_width": 0
    },
    "slight": {
        "rain_value": 30, "rain_length": 10, "rain_alpha": 0.9, "rain_width": 0.5
    },
    "moderate": {
        "rain_value": 100, "rain_length": 25, "rain_alpha": 0.8, "rain_width": 1
    },
    "severe": {
        "rain_value": 150, "rain_length": 45, "rain_alpha": 0.7, "rain_width": 1.5
    },
    "extreme": {
        # value: 从 250 降到 200 (密度降低)
        # length: 从 60 降到 50 (雨丝变短)
        # alpha: 从 0.55 提至 0.6 (原图更清晰，遮挡减少)
        # width: 保持 2 (比 severe 略粗，体现暴雨感)
        "rain_value": 200, "rain_length": 50, "rain_alpha": 0.6, "rain_width": 2
    }
}


# ================= 2. 辅助功能函数 =================

def _get_rain_noise(img, value):
    """生成初始的随机噪点图"""
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, k)
    return noise


def _apply_rain_blur(noise, length, angle, w):
    """将噪点拉伸成雨滴"""
    if length <= 0:
        return noise

    # 生成旋转矩阵
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)

    # 生成对角矩阵
    dig = np.diag(np.ones(length))

    # 旋转模糊核
    k = cv2.warpAffine(dig, trans, (length, length))

    # 【关键修复】处理雨滴粗细 w
    # 1. 强制转为 int (处理 0.5 这种情况)
    w = int(w)
    # 2. 确保宽度至少为 1 (防止 int(0.5) -> 0 导致错误)
    if w < 1:
        w = 1

    # 3. 确保是奇数 (GaussianBlur 要求核大小为奇数)
    w_kernel = w if w % 2 == 1 else w + 1

    # 对核进行高斯模糊 (注意：这里传入的一定是整数了)
    k = cv2.GaussianBlur(k, (w_kernel, w_kernel), 0)

    # 滤波
    blurred = cv2.filter2D(noise, -1, k)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return np.array(blurred, dtype=np.uint8)


# ================= 3. 主调用函数 =================

def add_rain(img, level=None):
    """
    添加雨天效果 (Rain).
    """

    # --- A. 确定等级 ---
    if level is None:
        selected_level = "moderate"
    elif level == "random":
        selected_level = np.random.choice(LEVEL_KEYS, p=LEVEL_PROBS)
    else:
        if level not in DEGRADATION_LEVELS:
            raise ValueError(f"未知的退化等级: {level}")
        selected_level = level

    # --- B. 获取参数 ---
    params = DEGRADATION_LEVELS[selected_level]
    value = params["rain_value"]
    length = params["rain_length"]
    alpha = params["rain_alpha"]
    width = params["rain_width"]

    # Clean 或 无雨滴 时直接返回
    if value <= 0:
        return img

    # --- C. 随机角度生成 ---
    angle = np.random.randint(-45, -14)

    # --- D. 生成雨滴层 ---
    noise = _get_rain_noise(img, value)
    rain_layer = _apply_rain_blur(noise, length, angle, width)

    # 假设输入 img 是 BGR (H, W, 3)
    if img.ndim == 3:
        rain_layer = np.expand_dims(rain_layer, 2)
        rain_layer = np.repeat(rain_layer, 3, 2)

    # --- E. 图像融合 ---
    result = cv2.addWeighted(img, alpha, rain_layer, 1 - alpha, 0)

    return result