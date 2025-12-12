import cv2
import numpy as np

DEGRADATION_LEVELS = {
    "clean": {"cast_severity": 0.0},  # 无色偏
    "slight": {"cast_severity": 0.15},  # 轻微氛围感 (如傍晚)
    "moderate": {"cast_severity": 0.35},  # 明显偏色 (如室内暖灯)
    "severe": {"cast_severity": 0.55},  # 严重偏色 (如路灯下)
    "extreme": {"cast_severity": 0.75}  # 极度偏色 (保留物体轮廓，但白平衡完全错误)
}

# 采样概率分布 (符合正态分布设定)
LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]


def add_color_cast(img, level=None):
    """
    添加空间自适应色偏 (Spatially Varying Color Cast).
    完全基于 NumPy/OpenCV 实现.

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
        # 依赖外部定义的 LEVEL_KEYS 和 LEVEL_PROBS
        selected_level = np.random.choice(LEVEL_KEYS, p=LEVEL_PROBS)
    else:
        if level not in DEGRADATION_LEVELS:
            raise ValueError(f"未知的退化等级: {level}")
        selected_level = level

    # --- 2. 获取参数 ---
    # 假设 DEGRADATION_LEVELS 已经包含 cast_severity
    params = DEGRADATION_LEVELS[selected_level]
    severity = params.get("cast_severity", 0.0)

    # Clean 或 强度为0 时直接返回
    if severity <= 0:
        return img

    # --- 3. 准备数据 ---
    # 转为 float32 [0, 1] 方便乘法计算
    img_float = img.astype(np.float32) / 255.0
    h, w, c = img_float.shape

    # --- 4. 生成随机基础色调 ---
    # 生成 3 个随机通道增益 (RGB/BGR)
    # +0.5 保证每个通道至少保留一半的亮度，避免变成纯黑
    rgb_gains = np.random.rand(c) + 0.5

    # 归一化：除以均值，保持图像整体亮度大致不变 (有的通道变亮，有的变暗)
    rgb_gains /= rgb_gains.mean()

    # --- 5. 生成空间不均匀分布 (4x4 网格) ---
    # 相比全局统一偏色，这种方法能模拟光照方向的不均匀
    grid_size = 4

    # 构建基础色网格 (4, 4, 3)
    base_grid = np.tile(rgb_gains, (grid_size, grid_size, 1))

    # 加入随机扰动 (Noise) 使光照更自然
    # 0.1 的扰动量既能产生变化，又不会产生突兀的色斑
    noise = np.random.normal(0, 0.1, (grid_size, grid_size, c))
    rough_map = base_grid + noise

    # 确保所有增益非负
    rough_map = np.abs(rough_map)

    # --- 6. 双线性插值放大 ---
    # 将 (4, 4) 的粗糙网格放大到 (H, W)
    # cv2.resize 自动进行双线性插值，形成平滑过渡
    smooth_map = cv2.resize(rough_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- 7. 混合与应用 ---
    # 根据 severity 进行加权混合
    # 当 severity=0 时，Map=1 (原图)；当 severity=1 时，Map=smooth_map (全色偏)
    final_map = (1 - severity) + severity * smooth_map

    # 应用色偏 (逐元素相乘)
    out = img_float * final_map

    # --- 8. 截断与转换 ---
    out = np.clip(out, 0, 1) * 255.0
    return out.astype(np.uint8)
