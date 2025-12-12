import os
import shutil
from pathlib import Path
from tqdm import tqdm  # 如果没有安装 tqdm，可以使用 pip install tqdm，或者去掉相关的进度条代码


def move_gt_files(source_root, dataset_root):
    """
    将 source_root 下所有 train/val 中的 GT 数据移动到 dataset_root 的统一结构中。
    策略：移动 (Move) 且 覆盖 (Overwrite)
    """
    source_path = Path(source_root)
    dataset_path = Path(dataset_root)

    # 1. 定义需要处理的子文件夹名称
    # 键是源目录中的文件夹名，值是它在 dataset 中的父级目录名
    split_map = {
        'train': 'train',
        'val': 'val'
    }

    # 定义需要查找的 GT 类型
    gt_types = ['Infrared_gt', 'Visible_gt']

    # 2. 预先构建目标目录结构
    for split_name in split_map.values():
        for gt_type in gt_types:
            dest_dir = dataset_path / split_name / gt_type
            os.makedirs(dest_dir, exist_ok=True)
            print(f"✅ 确保目录存在: {dest_dir}")

    print(f"\n🚀 开始扫描并移动文件... 从 {source_path} 到 {dataset_path}")
    print("-" * 50)

    # 3. 遍历源目录
    moved_count = 0
    overwritten_count = 0

    for root, dirs, files in os.walk(source_path):
        current_dir_name = os.path.basename(root)

        # 检查当前文件夹是不是 'train' 或 'val'
        if current_dir_name in split_map:
            target_split = split_map[current_dir_name]  # 'train' or 'val'

            # 检查这个 train/val 下面是否有 Infrared_gt 或 Visible_gt
            for gt_type in gt_types:
                src_gt_dir = Path(root) / gt_type

                if src_gt_dir.exists() and src_gt_dir.is_dir():
                    # 确定目标路径: dataset/train/Infrared_gt
                    dest_gt_dir = dataset_path / target_split / gt_type

                    # 获取该目录下所有文件
                    files_in_gt = os.listdir(src_gt_dir)

                    if not files_in_gt:
                        continue

                    # 移动文件
                    for filename in files_in_gt:
                        src_file = src_gt_dir / filename
                        dst_file = dest_gt_dir / filename

                        # 过滤掉非文件内容（比如子文件夹）
                        if not src_file.is_file():
                            continue

                        # 处理覆盖逻辑
                        if dst_file.exists():
                            os.remove(dst_file)  # 删除旧文件
                            overwritten_count += 1

                        # 移动
                        shutil.move(str(src_file), str(dst_file))
                        moved_count += 1

                    print(f"已处理: {src_gt_dir} -> {dest_gt_dir} (移动了 {len(files_in_gt)} 个文件)")

    print("-" * 50)
    print(f"🎉 处理完成!")
    print(f"📂 总共移动文件数: {moved_count}")
    print(f"⚠️ 发生覆盖的文件数: {overwritten_count}")


# ================= 配置区域 =================
# 请在这里修改你的实际路径
# target: 你的原始数据根目录 (例如 '~/MyFrame/DDL-12')
# dataset: 你想要存放结果的目录

target_folder = r'E:\IVIF_DataSets\DDL-12'  # 修改这里
dataset_folder = r'E:\IVIF_DataSets\DDL-12-CD'  # 修改这里，建议新建一个空文件夹

# 执行
if __name__ == '__main__':
    # 二次确认防止误操作
    print(f"警告：此操作将从 [{target_folder}] 移动文件到 [{dataset_folder}]")
    print("同名文件将被覆盖。源文件将被删除。")
    user_input = input("确认继续吗？(输入 y 继续): ")

    if user_input.lower() == 'y':
        move_gt_files(target_folder, dataset_folder)
    else:
        print("操作已取消。")