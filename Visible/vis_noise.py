import numpy as np
import cv2
import os

# ================= 全局配置区域 =================

LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]

# 统一参数字典
DEGRADATION_LEVELS = {
    "clean":    {
        "gaussian_std": 0,  "poisson_val": None, "sp_prob": 0
    },
    "slight":   {
        "gaussian_std": 5,  "poisson_val": 150,  "sp_prob": 0.0001
    },
    "moderate": {
        "gaussian_std": 15, "poisson_val": 80,   "sp_prob": 0.0005
    },
    "severe":   {
        "gaussian_std": 25, "poisson_val": 50,   "sp_prob": 0.002
    },
    "extreme":  {
        "gaussian_std": 35,
        "poisson_val": 30,
        "sp_prob": 0.005
    }
}


def add_noise(img, level=None):
    """
    添加混合噪声 (高斯 -> 泊松 -> 椒盐).
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

    # Clean 模式直接返回
    if selected_level == "clean":
        return img

    # --- 3. 准备数据 (转float防止溢出) ---
    noisy = img.astype(np.float32)

    # --- 4. 添加高斯噪声 (Gaussian) ---
    std = params["gaussian_std"]
    if std > 0:
        mean = 0
        noise_g = np.random.normal(mean, std, noisy.shape)
        noisy = noisy + noise_g

    # --- 5. 添加泊松噪声 (Poisson) ---
    val = params["poisson_val"]
    if val is not None:
        noisy_temp = np.clip(noisy, 0, 255)
        noisy = np.random.poisson(noisy_temp / 255.0 * val) / val * 255

    # --- 6. 截断并转回 uint8 (为椒盐做准备) ---
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # --- 7. 添加椒盐噪声 (Salt & Pepper) ---
    prob = params["sp_prob"]
    if prob > 0:
        # 【修正重点】：
        # 1. 只需要生成二维的随机矩阵 (H, W)，不需要变成 (H, W, 1)
        # 2. NumPy 会自动把二维掩码应用到三维图像的所有通道上
        rnd = np.random.rand(*noisy.shape[:2])

        # 椒噪声 (黑点, 0)
        # 这里的赋值会自动广播：如果 noisy 是 (H, W, 3)，
        # 掩码为 True 的位置，其 RGB 三个通道都会被设为 0
        noisy[rnd < (prob / 2)] = 0

        # 盐噪声 (白点, 255)
        noisy[rnd > (1 - prob / 2)] = 255

    return noisy