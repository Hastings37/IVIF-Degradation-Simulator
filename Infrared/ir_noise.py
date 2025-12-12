import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 定义5个等级的噪声参数 (根据您的反馈调整)
# Gaussian: std (标准差) -> 控制颗粒感大小
# Poisson: value (光子数) -> 值越小噪声越大 (注意：Extreme 设定为 30，避免过低)
# S&P: prob (总概率) -> 坏点比例 (Extreme 设定为 0.005，即 0.5% 的零星坏点)

DEGRADATION_LEVELS = {
    "clean": {"gaussian_std": 0, "poisson_val": None, "sp_prob": 0},
    "slight": {"gaussian_std": 5, "poisson_val": 150, "sp_prob": 0.0001},
    "moderate": {"gaussian_std": 15, "poisson_val": 80, "sp_prob": 0.0005},
    "severe": {"gaussian_std": 25, "poisson_val": 50, "sp_prob": 0.002},
    "extreme": {"gaussian_std": 35, "poisson_val": 30, "sp_prob": 0.005}
}

# 2. 定义正态分布采样概率 (15% Clean, 20% Slight, 30% Moderate, 25% Severe, 10% Extreme)
LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.17, 0.22, 0.33, 0.28, 0.0]

# 3. 输入输出路径
input_dir = 'YOUR_INPUT_PATH_HERE'  # 请替换为您的输入路径
output_dir = 'YOUR_OUTPUT_PATH_HERE'  # 请替换为您的输出路径


# ================= 功能函数 =================

def get_random_noise_params():
    """根据预设概率随机选择一个退化等级"""
    chosen_level = np.random.choice(LEVEL_KEYS, p=LEVEL_PROBS)
    return chosen_level, DEGRADATION_LEVELS[chosen_level]


def add_gaussian_noise(image, std):
    """添加高斯噪声"""
    if std <= 0: return image
    mean = 0
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_poisson_noise(image, value):
    """添加泊松噪声"""
    if value is None: return image
    # 为了增加一点随机性，可以在设定值附近微调，或者直接使用固定值
    # 这里我们允许 value 在一定范围内浮动，模拟真实的不稳定性
    actual_value = np.random.randint(value - 5, value + 6)

    noisy_image = np.random.poisson(image / 255.0 * actual_value) / actual_value * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def add_salt_and_pepper_noise(image, total_prob):
    """添加椒盐噪声 (零星坏点)"""
    if total_prob <= 0: return image

    noisy_image = np.copy(image)
    # 将总概率分配给盐(白点)和椒(黑点)，各占一半
    salt_prob = total_prob / 2
    pepper_prob = total_prob / 2

    # 生成随机掩码
    rnd = np.random.rand(*image.shape)

    # 椒噪声 (0)
    noisy_image[rnd < pepper_prob] = 0
    # 盐噪声 (255)
    noisy_image[rnd > (1 - salt_prob)] = 255

    return noisy_image


def add_noise(img, level=None):
    """
    根据指定等级添加混合噪声 (高斯 -> 泊松 -> 椒盐).
    只返回处理后的图像。
    """

    # 1. 确定退化等级
    if level is None:
        selected_level = "moderate"
    elif level == "random":
        selected_level = np.random.choice(LEVEL_KEYS, p=LEVEL_PROBS)
    else:
        # 这里假设您使用的是存储噪声参数的字典 (DEGRADATION_LEVELS)
        if level not in DEGRADATION_LEVELS:
            # 如果您把噪声参数也放进了 DEGRADATION_LEVELS，请改用 DEGRADATION_LEVELS
            raise ValueError(f"未知的退化等级: {level}")
        selected_level = level

    # 2. 获取参数
    params = DEGRADATION_LEVELS[selected_level]

    # 如果是 Clean，直接返回原图
    if selected_level == "clean":
        return img

    # 转换浮点数进行计算，避免溢出
    noisy = img.astype(np.float32)

    # 3. 添加高斯噪声 (Gaussian)
    std = params["gaussian_std"]
    if std > 0:
        gaussian = np.random.normal(0, std, noisy.shape)
        noisy = noisy + gaussian

    # 截断以适应泊松分布要求
    noisy = np.clip(noisy, 0, 255)

    # 4. 添加泊松噪声 (Poisson)
    val = params["poisson_val"]
    if val is not None:
        # 严格锁定参数
        noisy = np.random.poisson(noisy / 255.0 * val) / val * 255

    # 再次截断并转回 uint8 准备加椒盐
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # 5. 添加椒盐噪声 (Salt & Pepper)
    prob = params["sp_prob"]
    if prob > 0:
        # 生成统一的随机矩阵
        rnd = np.random.rand(*noisy.shape)

        # 椒噪声 (黑点, 0)
        noisy[rnd < (prob / 2)] = 0

        # 盐噪声 (白点, 255)
        noisy[rnd > (1 - prob / 2)] = 255

    return noisy

# ================= 主程序 =================


# 需要重写的代码内容；
if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历图像
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    print(f"开始处理 {len(file_list)} 张图像...")

    # 使用 tqdm 显示进度条
    for filename in tqdm(file_list):
        image_path = os.path.join(input_dir, filename)

        # 读取灰度图 (保持您的设置)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # 1. 获取当前图片的随机退化等级和参数
        level_name, params = get_random_noise_params()

        # 2. 只有在非 Clean 模式下才进行处理
        if level_name == "clean":
            final_image = image
        else:
            # 串行叠加三种噪声：高斯 -> 泊松 -> 椒盐
            img_g = add_gaussian_noise(image, std=params["gaussian_std"])
            img_p = add_poisson_noise(img_g, value=params["poisson_val"])
            final_image = add_salt_and_pepper_noise(img_p, total_prob=params["sp_prob"])

        # 3. 保存图像
        # 建议：直接使用原文件名保存，这样可以和 GT 文件夹一一对应
        # 也可以在文件名中加上等级后缀以便调试，例如: name_severe.png
        # 这里默认保持原名
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, final_image)

        # (可选) 打印调试信息，如果不需要可以注释掉
        # tqdm.write(f"{filename} -> Level: {level_name}")

    print("\n处理完成！所有图像已保存至:", output_dir)