import os
import shutil
from pathlib import Path


def merge_val_into_train(dataset_root):
    """
    将 dataset_root/val 下的 Infrared_gt 和 Visible_gt 内容移动到 train 对应目录下。
    策略: 移动 (Move), 遇重名跳过 (Skip), 仅限特定图片格式。
    """
    base_path = Path(dataset_root)

    # 定义源目录和目标目录的子结构
    src_split = 'val'
    dst_split = 'train'

    # 需要处理的子文件夹
    sub_folders = ['Infrared_gt', 'Visible_gt']

    # 支持的文件格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    print(f"🚀 开始任务: 将 [{src_split}] 合并入 [{dst_split}]")
    print(f"📂 数据集根目录: {base_path}")
    print("-" * 50)

    total_moved = 0
    total_skipped = 0

    for folder_name in sub_folders:
        # 构建完整路径
        src_dir = base_path / src_split / folder_name
        dst_dir = base_path / dst_split / folder_name

        # 检查源目录是否存在
        if not src_dir.exists():
            print(f"⚠️ 源目录不存在，跳过: {src_dir}")
            continue

        # 确保目标目录存在 (如果没有则创建，防止报错)
        os.makedirs(dst_dir, exist_ok=True)

        print(f"\n正在处理: {folder_name} ...")

        # 获取源目录下的所有文件
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_extensions)]

        if not files:
            print("   (目录为空，无文件需要移动)")
            continue

        folder_moved_count = 0
        folder_skipped_count = 0

        for filename in files:
            src_file_path = src_dir / filename
            dst_file_path = dst_dir / filename

            # 策略：跳过同名文件
            if dst_file_path.exists():
                # print(f"   [跳过] 目标已存在: {filename}")
                folder_skipped_count += 1
                total_skipped += 1
            else:
                try:
                    # 执行移动
                    shutil.move(str(src_file_path), str(dst_file_path))
                    folder_moved_count += 1
                    total_moved += 1
                except Exception as e:
                    print(f"   [错误] 无法移动 {filename}: {e}")

        print(f"   ✅ 完成。 移动: {folder_moved_count}, 跳过: {folder_skipped_count}")

    print("-" * 50)
    print("🎉 所有操作结束")
    print(f"📊 总计移动: {total_moved} 张图片")
    print(f"📊 总计跳过: {total_skipped} 张重复图片")


# ================= 配置与执行 =================

if __name__ == '__main__':
    # 请修改为您实际的 dataset 文件夹路径
    dataset_folder = '/public/home/lijieheng3/IVIF-CD'

    # 二次确认
    print(f"警告：此操作将把 [{dataset_folder}/val] 中的图片【移动】到 [train] 中。")
    print("重复的文件将被留在 val 中，不进行覆盖。")
    user_input = input("确认执行吗？(输入 y 继续): ")

    if user_input.lower() == 'y':
        merge_val_into_train(dataset_folder)
    else:
        print("操作已取消。")