import numpy as np
import cv2
import os

# ================= 全局配置区域 =================

LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]

# 统一参数字典 (均匀雾霾版)
# t: 透射率 (1.0 = 无雾, 0.0 = 全白)
# A: 大气光亮度 (0.0 = 黑, 1.0 = 白)
DEGRADATION_LEVELS = {
    "clean":    {
        "haze_t_min": 1.0,  "haze_t_max": 1.0,
        "haze_A_min": 0,    "haze_A_max": 0
    },
    "slight":   {
        "haze_t_min": 0.85, "haze_t_max": 0.95, # 轻微朦胧
        "haze_A_min": 0.8,  "haze_A_max": 0.95
    },
    "moderate": {
        "haze_t_min": 0.65, "haze_t_max": 0.85, # 明显雾气
        "haze_A_min": 0.8,  "haze_A_max": 0.95
    },
    "severe":   {
        "haze_t_min": 0.40, "haze_t_max": 0.65, # 严重遮挡
        "haze_A_min": 0.8,  "haze_A_max": 0.95
    },
    "extreme":  {
        "haze_t_min": 0.25, "haze_t_max": 0.40, # 极度浓雾 (保留一点点信息)
        "haze_A_min": 0.8,  "haze_A_max": 0.95
    }
}


def add_uniform_haze(img, level=None):
    """
    添加均匀雾霾 (Uniform Haze).
    无需深度图，全图统一透射率.

    参数:
        img: 输入图像 (BGR, uint8)
        level: 退化等级

    返回:
        numpy.ndarray: 处理后的图像 (uint8)
    """

    # --- 1. 确定等级 ---
    if level is None:
        selected_level = "moderate"
    elif level == "random":
        selected_level = np.random.choice(LEVEL_KEYS, p=LEVEL_PROBS)
    else:
        if level not in DEGRADATION_LEVELS:
            raise ValueError(f"未知的退化等级: {level}")
        selected_level = level

    # --- 2. 获取参数 ---
    params = DEGRADATION_LEVELS[selected_level]
    t_min = params["haze_t_min"]
    t_max = params["haze_t_max"]

    # Clean 模式 (t=1) 直接返回
    if t_min >= 1.0:
        return img

    # --- 3. 生成随机参数 ---
    # 随机透射率 t (全图统一)
    t = np.random.uniform(t_min, t_max)

    # 随机大气光 A (全图统一)
    A_val = np.random.uniform(params["haze_A_min"], params["haze_A_max"])
    A = np.array([A_val, A_val, A_val])  # 纯白光

    # --- 4. 应用公式 ---
    # 归一化输入
    img_norm = img.astype(np.float32) / 255.0

    # I = J * t + A * (1 - t)
    # 因为 t 是标量，这里直接广播计算
    hazy_img = img_norm * t + A * (1 - t)

    # --- 5. 截断与输出 ---
    out = np.clip(hazy_img, 0, 1) * 255.0
    return out.astype(np.uint8)