"""
模式B：基于给定Track ID生成可视化视频
直接读取labels/*.txt中的标注数据，不进行YOLO推理
确保Track ID与标注完全一致，无匹配误差
"""
import os
import cv2
import glob
import numpy as np
from collections import defaultdict

print("=" * 60)
print("🎬 模式B：基于给定Track ID生成可视化视频")
print("=" * 60)

# 配置
labels_dir = 'labels/'
images_dir = 'images/'
output_dir = 'runs/obb/track_given_id'

print(f"\n📂 输入目录:")
print(f"   - 标注文件: {labels_dir}")
print(f"   - 图片目录: {images_dir}")
print(f"\n📁 输出目录: {output_dir}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取所有图片和标注文件（按顺序）
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))

print(f"\n📊 找到 {len(image_files)} 张图片, {len(label_files)} 个标注文件")

if len(image_files) != len(label_files):
    print(f"⚠️  警告：图片和标注数量不一致！")
    print(f"   图片: {len(image_files)}, 标注: {len(label_files)}")

# 类别映射和颜色
class_names = {0: 'car', 1: 'pedestrian', 2: 'two-wheeler'}
class_colors = {
    0: (0, 255, 0),      # car - 绿色
    1: (255, 0, 0),      # pedestrian - 蓝色
    2: (0, 0, 255)       # two-wheeler - 红色
}

# Track ID颜色缓存（确保同一ID始终用同一颜色）
track_id_colors = {}
def get_track_color(track_id_str):
    """为每个Track ID分配固定颜色"""
    if track_id_str not in track_id_colors:
        # 使用hash生成随机但固定的颜色
        import hashlib
        hash_obj = hashlib.md5(track_id_str.encode())
        hash_bytes = hash_obj.digest()
        color = (int(hash_bytes[0]) % 200 + 55, 
                int(hash_bytes[1]) % 200 + 55, 
                int(hash_bytes[2]) % 200 + 55)
        track_id_colors[track_id_str] = color
    return track_id_colors[track_id_str]

# 读取第一张图片获取尺寸
first_image_path = os.path.join(images_dir, image_files[0])
first_frame = cv2.imread(first_image_path)
if first_frame is None:
    print(f"❌ 无法读取图片: {first_image_path}")
    exit(1)

height, width = first_frame.shape[:2]
print(f"📐 图片尺寸: {width}x{height}")

# 创建视频写入器
output_video = os.path.join(output_dir, 'tracking_result.mp4')
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not video_writer.isOpened():
    print("❌ 无法创建视频文件")
    exit(1)

print(f"\n🎥 生成可视化视频: {output_video}")
print("-" * 60)

total_detections = 0

for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    frame_num = i + 1
    image_path = os.path.join(images_dir, image_file)
    
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ⚠️  跳过无法读取的图片: {image_file}")
        continue
    
    # 读取标注文件
    detections_in_frame = 0
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        # 检查是否有Track ID（11个字段）
        if len(parts) < 11:
            continue
        
        try:
            # 解析标注数据
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:9]]  # x1,y1,x2,y2,x3,y3,x4,y4
            confidence = float(parts[9])
            track_id_str = parts[10]
            
            # 跳过无效的Track ID
            if not track_id_str or track_id_str == 'None':
                continue
            
            # 转换归一化坐标为像素坐标
            x1_px = int(coords[0] * width)
            y1_px = int(coords[1] * height)
            x2_px = int(coords[2] * width)
            y2_px = int(coords[3] * height)
            x3_px = int(coords[4] * width)
            y3_px = int(coords[5] * height)
            x4_px = int(coords[6] * width)
            y4_px = int(coords[7] * height)
            
            pts = np.array([[x1_px, y1_px], [x2_px, y2_px], 
                           [x3_px, y3_px], [x4_px, y4_px]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # 获取Track ID对应的颜色
            track_color = get_track_color(track_id_str)
            
            # 绘制OBB框
            cv2.polylines(frame, [pts], True, track_color, 2)
            
            # 提取Track ID的数字部分用于显示
            try:
                tid_num = track_id_str.split('_')[1]
            except:
                tid_num = track_id_str
            
            # 在框的左上角显示Track ID和类别
            label_text = f"{class_names.get(cls, 'unknown')}_{tid_num}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # 计算文字背景大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness)
            
            # 绘制文字背景
            cv2.rectangle(frame, 
                         (x1_px, y1_px - text_height - 10),
                         (x1_px + text_width, y1_px),
                         track_color, -1)
            
            # 绘制文字
            cv2.putText(frame, label_text,
                       (x1_px, y1_px - 5),
                       font, font_scale, (255, 255, 255), thickness)
            
            detections_in_frame += 1
            
        except (ValueError, IndexError) as e:
            continue
    
    total_detections += detections_in_frame
    
    # 在左上角显示帧信息
    info_text = f"Frame: {frame_num}/{len(image_files)} | Mode B (Given Track ID) | Detections: {detections_in_frame}"
    cv2.putText(frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 写入视频帧
    video_writer.write(frame)
    
    # 显示进度
    if (i + 1) % 5 == 0 or i == len(image_files) - 1:
        print(f"  [{i+1}/{len(image_files)}] {image_file}: {detections_in_frame} 个目标")

print("-" * 60)
print("\n✅ 视频生成完成！")

# 释放资源
video_writer.release()

print("\n" + "=" * 60)
print("🎉 模式B可视化完成！")
print("=" * 60)
print(f"\n📁 结果保存在: {output_dir}/")
print(f"   ✅ tracking_result.mp4 - 带Track ID的可视化视频")
print(f"   📊 总检测数: {total_detections}")
print(f"\n💡 特点:")
print(f"   - 直接读取labels/*.txt，不进行YOLO推理")
print(f"   - Track ID与标注完全一致，无匹配误差")
print(f"   - 每个Track ID有固定颜色便于追踪")
