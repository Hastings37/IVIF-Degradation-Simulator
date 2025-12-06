import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveColorCastSimulator(nn.Module):
    """
    科研专用：空间自适应色偏模拟器 (Spatially Varying Color Cast)
    基于 Von Kries 物理模型: I_degraded(x,y) = I_clean(x,y) * Map(x,y)
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

    def forward(self, img: torch.Tensor, severity: float = 0.5, mode: str = 'local') -> torch.Tensor:
        """
        Args:
            img: 输入图像 [B, 3, H, W], 范围 [0, 1]
            severity: 色偏强度 [0, 1].
            mode:
                - 'global': 全局统一色偏 (整张图偏色一样)
                - 'local':  空间不均匀色偏 (模拟混合光源，光照不均) -> 你选了"涉及"，主要用这个
        """
        b, c, h, w = img.shape

        # 1. 确定基础色调 (随机选择一种主导色)
        # 格式: [R_gain, G_gain, B_gain]
        # 为了模拟真实环境，我们通常增强某一通道，抑制其他通道
        base_color = self._sample_random_color_direction(b)  # [B, 3]

        if mode == 'global':
            # --- 模式1：全局色偏 ---
            # 形状扩展为 [B, 3, 1, 1]
            color_map = base_color.view(b, c, 1, 1)
            # 根据 severity 插值: 强度0时为全1(无色偏)，强度1时为 base_color
            final_map = (1 - severity) * torch.ones_like(color_map) + severity * color_map

        elif mode == 'local':
            # --- 模式2：空间不均匀色偏 (主流科研方法) ---
            # 核心思路：生成低分辨率的随机噪声，然后双线性插值放大，形成平滑的渐变

            # 生成一个低分辨率的网格 (例如 4x4)，模拟光照分布的低频变化
            grid_size = 4
            # 在基础色调附近引入随机扰动
            # noise shape: [B, 3, 4, 4]
            noise = torch.randn(b, c, grid_size, grid_size).to(self.device) * 0.2

            # 将基础色调扩展到网格大小
            base_grid = base_color.view(b, c, 1, 1).repeat(1, 1, grid_size, grid_size)

            # 混合基础色与空间扰动
            rough_map = base_grid + noise

            # 强制所有增益 > 0
            rough_map = torch.abs(rough_map)

            # 双线性插值上采样到原图大小 [B, 3, H, W]
            smooth_map = F.interpolate(rough_map, size=(h, w), mode='bilinear', align_corners=False)

            # 根据 severity 混合：使得 map 均值在 1.0 附近 (保持亮度)，但带有颜色倾向
            # 这里我们让 severity 控制 map 偏离 (1,1,1) 的程度
            ones = torch.ones_like(smooth_map)
            final_map = (1 - severity) * ones + severity * smooth_map

        # 2. 应用 Von Kries 模型 (逐元素相乘)
        out = img * final_map

        # 3. 物理约束：截断到 [0, 1] 范围 (模拟传感器饱和)
        return torch.clamp(out, 0.0, 1.0)

    def _sample_random_color_direction(self, batch_size):
        """随机生成主导色增益，模拟不同色温 (冷色/暖色/特殊光)"""
        # 随机生成 RGB 权重
        colors = torch.rand(batch_size, 3).to(self.device)
        # 归一化，使得某个通道增强(>1)，某个减弱(<1)，但整体亮度大致保持
        # 加上 0.5 防止某个通道过于接近 0
        colors = colors + 0.5
        # 归一化均值为 1
        means = colors.mean(dim=1, keepdim=True)
        colors = colors / means
        return colors


# --- 验证代码 (你可以直接运行这段来看看效果) ---
if __name__ == "__main__":
    # 模拟一张灰色图片
    import cv2

    img = cv2.imread(r'E:\Recurrence\IVIF-Degradation-Simulator\Images\Visible\00001D.png')
    img = torch.from_numpy(img).float() / 255.0  # HWC
    img = img.permute(2, 0, 1).unsqueeze(0).cuda()  # [1,3,H,W]

    # BGR → RGB
    img = img.index_select(1, torch.tensor([2, 1, 0], device=img.device))

    input_img = img

    simulator = AdaptiveColorCastSimulator().cuda()

    # 1. 模拟不均匀色偏 (你的需求)
    degraded_local = simulator(input_img, severity=0.7, mode='local')

    img=degraded_local

    import matplotlib.pyplot as plt

    # degraded_local: shape = (1, 3, H, W)  range = [0,1]
    img = degraded_local[0]  # (3, H, W)

    # 转成 numpy + HWC
    img = img.permute(1, 2, 0).detach().cpu().numpy()

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print(f"输入尺寸: {input_img.shape}")
    print(f"输出尺寸: {degraded_local.shape}")
    print("不均匀色偏模拟完成。输出图像包含平滑过渡的颜色变化。")


