import numpy as np
import cv2

# ================= 1. 全局配置区域 =================

# 采样概率分布 (符合正态分布设定)
LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]

# 统一参数字典 (仅展示条纹噪声相关参数)
DEGRADATION_LEVELS = {
    "clean": {"stripe_beta": 0},  # 无条纹
    "slight": {"stripe_beta": 2},  # 几乎不可见
    "moderate": {"stripe_beta": 6},  # 明显可见 (原设定)
    "severe": {"stripe_beta": 15},  # 强烈条纹
    "extreme": {"stripe_beta": 25}  # 极度干扰
}


# ================= 2. 条纹噪声函数 =================

def add_stripe(img, level=None):
    """
    添加红外竖直条纹噪声 (Fixed Pattern Noise).

    参数:
        img: 输入图像 (BGR 3通道, uint8)
        level:
            - None: 默认为 'moderate'
            - 'random': 按概率随机选择
            - 指定字符串: 'clean', 'slight', ...

    返回:
        numpy.ndarray: 处理后的图像 (uint8)
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
    # 注意：这里假设 DEGRADATION_LEVELS 字典中一定包含 stripe_beta 键
    params = DEGRADATION_LEVELS[selected_level]
    beta = params.get("stripe_beta", 0)  # 使用 .get 防止键不存在报错

    # 如果强度 <= 0，直接返回原图
    if beta <= 0:
        return img

    # --- C. 生成噪声 ---
    h, w = img.shape[:2]

    # 1. 生成一维竖直条纹基准 (只在宽度方向随机)
    noise_col = np.random.normal(0, beta, w)

    # 2. 沿高度方向复制，形成 (H, W) 的条纹图
    S_noise = np.tile(noise_col, (h, 1))

    # 3. 适配 3通道图像 (BGR)
    # 因为输入是 cv2 读取的 3通道图，需要把噪声叠成 3层
    if img.ndim == 3:
        S_noise = np.stack([S_noise] * 3, axis=2)

    # --- D. 叠加与截断 ---
    # 转 float32 防止溢出 (uint8 加法会回卷)
    noisy_img = img.astype(np.float32) + S_noise

    # 截断回 0-255 并转 uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img