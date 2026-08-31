"""
数据集划分和配置文件生成脚本
按8:2比例划分训练集和验证集
"""
import os
import shutil
import random

# 设置随机种子，保证可复现
random.seed(42)

# 路径配置
base_dir = r'd:\Project\大数据课设'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
dataset_dir = os.path.join(base_dir, 'dataset')

train_img_dir = os.path.join(dataset_dir, 'images', 'train')
val_img_dir = os.path.join(dataset_dir, 'images', 'val')
train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')

# 获取所有图片文件
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
image_files.sort()  # 排序保证一致性

print(f"总图片数: {len(image_files)}")

# 按8:2划分
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")

# 复制文件到对应目录
def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for filename in file_list:
        # 复制图片
        src_img = os.path.join(src_img_dir, filename)
        dst_img = os.path.join(dst_img_dir, filename)
        shutil.copy2(src_img, dst_img)
        
        # 复制并清理标注（11列→9列，移除confidence和track_id）
        label_filename = filename.replace('.jpg', '.txt')
        src_lbl = os.path.join(src_lbl_dir, label_filename)
        dst_lbl = os.path.join(dst_lbl_dir, label_filename)
        if os.path.exists(src_lbl):
            with open(src_lbl, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 11:  # 11列格式：class + 8坐标 + confidence + track_id
                    # 只保留前9列
                    cleaned_line = ' '.join(parts[:9]) + '\n'
                    cleaned_lines.append(cleaned_line)
                else:
                    # 其他格式保持不变
                    cleaned_lines.append(line)
            
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)

print("\n正在复制训练集...")
copy_files(train_files, images_dir, labels_dir, train_img_dir, train_lbl_dir)

print("正在复制验证集...")
copy_files(val_files, images_dir, labels_dir, val_img_dir, val_lbl_dir)

print("\n✅ 数据集划分完成！")
print(f"训练集路径: {train_img_dir}")
print(f"验证集路径: {val_img_dir}")

# 生成yolo_test.yaml配置文件
yaml_content = """# YOLO OBB 训练配置文件
# 数据集路径
path: d:/Project/大数据课设/dataset  # 数据集根目录
train: images/train  # 训练集图片相对路径
val: images/val      # 验证集图片相对路径

# 类别信息
nc: 3  # 类别数量
names:
  0: car
  1: pedestrian
  2: two-wheeler

# 训练参数（可选，可在train.py中覆盖）
# epochs: 100
# batch: 16
# imgsz: 640
"""

yaml_path = os.path.join(base_dir, 'yolo_test.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"\n✅ 配置文件已生成: {yaml_path}")
print("\n下一步：开始模型训练")
