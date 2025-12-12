import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 导入模块并重命名以防冲突 =================
# 假设您的文件夹结构为 Infrared/ 和 Visible/ 且在当前路径或 PYTHONPATH 下

# 红外退化模块
from Infrared.ir_noise import add_noise as add_ir_noise  # 重命名 IR 噪声
from Infrared.ir_stripe_noise import add_stripe
from Infrared.ir_low_contrast import adjust_contrast

# 可见光退化模块
from Visible.vis_blur import add_motion_blur
from Visible.vis_rain import add_rain
from Visible.vis_haze import add_uniform_haze
from Visible.vis_color_shift import add_color_cast
from Visible.vis_noise import add_noise as add_vis_noise  # 重命名 Vis 噪声

# ================= 配置区域 =================
# 用于在 main 函数中生成随机等级
LEVEL_KEYS = ["clean", "slight", "moderate", "severe", "extreme"]
LEVEL_PROBS = [0.15, 0.20, 0.30, 0.25, 0.10]


def add_degradation(Infrared_gt_path, Visible_gt_path, Infrared_path, Visible_path, level='random'):
    '''
    批量处理红外和可见光图像对，添加一致的退化效果。

    Args:
        Infrared_gt_path: 红外 GT (Ground Truth) 输入文件夹
        Visible_gt_path: 可见光 GT 输入文件夹
        Infrared_path: 红外退化图像 输出文件夹
        Visible_path: 可见光退化图像 输出文件夹
        level: 退化等级 ('random', 'clean', 'slight', 'moderate', 'severe', 'extreme')
    '''

    # 1. 创建输出目录
    if not os.path.exists(Infrared_path):
        os.makedirs(Infrared_path)
    if not os.path.exists(Visible_path):
        os.makedirs(Visible_path)

    # 2. 获取文件列表 (以红外文件夹为基准)
    # 假设 IR 和 Vis 的文件名是一一对应的
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    file_list = [f for f in os.listdir(Infrared_gt_path) if f.lower().endswith(valid_ext)]

    print(f"开始处理，共 {len(file_list)} 对图像...")
    print(f"全局设定等级: {level}")

    # 3. 循环处理
    max_iter=100
    count=0
    for filename in tqdm(file_list):
        count+=1
        if count>max_iter:
            break
        # --- 路径构建 ---
        ir_gt_file = os.path.join(Infrared_gt_path, filename)
        vis_gt_file = os.path.join(Visible_gt_path, filename)

        ir_out_file = os.path.join(Infrared_path, filename)
        vis_out_file = os.path.join(Visible_path, filename)

        # --- 检查配对文件是否存在 ---
        if not os.path.exists(vis_gt_file):
            # 尝试处理后缀不一致的情况 (例如 IR是.png, Vis是.jpg)
            # 这里简单跳过，视具体数据集情况而定
            # tqdm.write(f"警告: 找不到对应的可见光图像 {filename}，跳过。")
            continue

        # --- 读取图像 ---
        # 红外通常读取为灰度或RGB均可，这里为了兼容后续stack操作，建议统一读取为RGB
        img_ir = cv2.imread(ir_gt_file)
        img_vis = cv2.imread(vis_gt_file)

        if img_ir is None or img_vis is None:
            continue

        # ================= 核心逻辑：确定当前对的等级 =================
        # 方案 A：一对图片使用同一个随机等级，保持语义一致性
        # 1. 传感器热噪声
        ir_deg = add_ir_noise(img_ir, level=level) # 这里还是使用的这个啊；

        # 2. 条纹噪声 (非均匀性)
        ir_deg = add_stripe(ir_deg, level=level)

        # 3. 低对比度 (放在最后，模拟整体信号衰减)
        ir_deg = adjust_contrast(ir_deg, level=level)

        # ================= 处理 B: 可见光图像 (Visible) =================
        # 顺序: 雨 -> 模糊 -> 雾 -> 色偏 -> 噪声

        # 1. 雨天 (遮挡物体)
        vis_deg = add_rain(img_vis, level=level)

        # 2. 运动模糊 (雨和背景一起糊)
        vis_deg = add_motion_blur(vis_deg, level=level)

        # 3. 均匀雾霾 (降低整体对比度，覆盖在雨之上)
        vis_deg = add_uniform_haze(vis_deg, level=level)

        # 4. 色偏 (光照环境/白平衡)
        vis_deg = add_color_cast(vis_deg, level=level)

        # 5. 传感器噪声 (最后成像阶段产生)
        vis_deg = add_vis_noise(vis_deg, level=level)

        # ================= 保存结果 =================
        cv2.imwrite(ir_out_file, ir_deg)
        cv2.imwrite(vis_out_file, vis_deg)

    print("\n所有图像处理完成！")




if __name__=="__main__":
    Infrared_gt_path = r'E:\IVIF_DataSets\DDL-12-CD\train\Infrared_gt'
    Visible_gt_path = r'E:\IVIF_DataSets\DDL-12-CD\train\Visible_gt'
    Infrared_path = r'E:\IVIF_DataSets\DDL-12-CD\train\Infrared'
    Visible_path = r'E:\IVIF_DataSets\DDL-12-CD\train\Visible'
    add_degradation(Infrared_gt_path,Visible_gt_path,Infrared_path,Visible_path,level='random')