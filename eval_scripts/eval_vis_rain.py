import cv2
import os
import numpy as np
from Visible.vis_rain import add_rain,DEGRADATION_LEVELS


img_path = r'E:\Recurrence\IVIF-Degradation-Simulator\vis_gt.png'

img = cv2.imread(img_path)  # 默认按照bgr的形式读取到其内容；
for level in DEGRADATION_LEVELS:
    adjusted_img = add_rain(img, level)
    save_path = os.path.join(r'E:\Recurrence\IVIF-Degradation-Simulator\results\vis_rain', f'vis_rain_{level}.png')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print(cv2.imwrite(save_path, adjusted_img))
    print(f"Saved adjusted image at level {level} to {save_path}")