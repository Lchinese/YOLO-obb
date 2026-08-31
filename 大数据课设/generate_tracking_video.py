"""
生成目标跟踪可视化视频
使用跟踪结果在原始图片上绘制检测框和Track ID，生成MP4视频
"""
from ultralytics import YOLO
import cv2
import os
import json
from collections import defaultdict
import numpy as np

print("=" * 60)
print("🎬 生成目标跟踪可视化视频")
print("=" * 60)

# 配置参数
images_dir = 'images/'
track_data_file = 'runs/obb/track/tracks_merged.json'  # 使用合并后的轨迹
output_video = 'runs/obb/track/tracking_result.mp4'

# 加载模型进行推理
print(f"\n📦 加载模型: runs/obb/yolo_obb_training/weights/best.pt")
model = YOLO('runs/obb/yolo_obb_training/weights/best.pt')

# 加载跟踪数据（用于匹配Track ID）
print(f"📦 加载跟踪数据: {track_data_file}")
with open(track_data_file, 'r', encoding='utf-8') as f:
    all_tracks = json.load(f)

print(f"   📊 共 {len(all_tracks)} 条轨迹")

# 获取所有图片文件（按顺序）
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
print(f"   📷 共 {len(image_files)} 张图片")

# 读取第一张图片获取尺寸
first_image_path = os.path.join(images_dir, image_files[0])
first_frame = cv2.imread(first_image_path)
if first_frame is None:
    print(f"❌ 无法读取图片: {first_image_path}")
    exit(1)

height, width = first_frame.shape[:2]
print(f"   📐 视频尺寸: {width}x{height}")

# 创建视频写入器（使用MP4V编码，兼容性最好）
fps = 10  # 帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V编码，最兼容
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not video_writer.isOpened():
    print("❌ 无法创建视频文件")
    exit(1)

print(f"\n🎥 视频输出: {output_video}")
print(f"   📹 FPS: {fps}")
print(f"   ⏱️  预计时长: {len(image_files)/fps:.1f}秒")
print(f"   🎬 编码格式: {chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)}")
print("\n" + "=" * 60)

# 类别颜色映射
class_colors = {
    0: (0, 255, 0),      # car - 绿色
    1: (255, 0, 0),      # pedestrian - 蓝色
    2: (0, 0, 255)       # two-wheeler - 红色
}
class_names = ['car', 'pedestrian', 'two-wheeler']

# 为每个Track ID分配固定颜色（便于追踪）
track_colors = {}
np.random.seed(42)  # 固定随机种子，保证颜色一致

def get_track_color(track_id):
    """为每个Track ID生成固定颜色"""
    if track_id not in track_colors:
        # 生成明亮的颜色
        color = tuple(np.random.randint(50, 255, size=3).tolist())
        track_colors[track_id] = color
    return track_colors[track_id]

def match_tracks_to_detections(tracks_at_frame, detections):
    """
    将跟踪数据中的Track ID匹配到当前帧的检测
    使用最近邻匹配，考虑类别和位置
    """
    matched_track_ids = {}
    
    for det_idx, det in enumerate(detections):
        det_x, det_y = det['x'], det['y']
        det_cls = det['class']
        
        best_track_id = None
        best_distance = float('inf')
        
        # 根据类别设置不同的匹配阈值
        if det_cls == 0:  # car
            max_distance = 30  # 汽车移动较快
        elif det_cls == 1:  # pedestrian
            max_distance = 15  # 行人移动较慢
        else:  # two-wheeler
            max_distance = 25  # 两轮车移动中等
        
        # 遍历该类别的所有轨迹点
        for tid_str, track_points in tracks_at_frame.items():
            for point in track_points:
                if point['class'] == det_cls:
                    dist = ((point['x'] - det_x) ** 2 + (point['y'] - det_y) ** 2) ** 0.5
                    if dist < best_distance and dist < max_distance:
                        best_distance = dist
                        best_track_id = int(tid_str)
        
        matched_track_ids[det_idx] = best_track_id
    
    return matched_track_ids

print("\n🚀 开始生成视频...")
print("-" * 60)

# 先进行推理获取所有检测结果
print("\n🔍 正在推理所有图片...")
results = model.predict(
    source=images_dir,
    conf=0.1,  # 低阈值保留更多检测
    iou=0.7,
    save=False,
    verbose=False
)

# 预处理：按帧组织跟踪数据
print("📊 正在预处理跟踪数据...")
tracks_by_frame = defaultdict(list)
for tid_str, track_points in all_tracks.items():
    for point in track_points:
        tracks_by_frame[point['frame']].append({
            'track_id': int(tid_str),
            'x': point['x'],
            'y': point['y'],
            'class': point['class']
        })
print(f"   ✅ 已处理 {len(tracks_by_frame)} 帧的跟踪数据")

for i, (image_file, result) in enumerate(zip(image_files, results)):
    image_path = os.path.join(images_dir, image_file)
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"  ⚠️  跳过无法读取的图片: {image_file}")
        continue
    
    frame_num = i + 1
    
    # 获取当前帧的所有检测结果（obb格式）
    current_detections = []
    if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
        obb = result.obb
        xywhr = obb.xywhr.cpu().numpy()
        classes = obb.cls.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        
        for j in range(len(obb)):
            cx, cy, w, h, angle = xywhr[j]
            cls = int(classes[j])
            conf = float(confs[j])
            
            current_detections.append({
                'track_id': None,  # 稍后匹配
                'x': cx,
                'y': cy,
                'w': w,
                'h': h,
                'class': cls,
                'confidence': conf
            })
        
        # 匹配Track ID
        if frame_num in tracks_by_frame:
            # 组织跟踪数据为字典格式
            tracks_dict = defaultdict(list)
            for item in tracks_by_frame[frame_num]:
                tracks_dict[str(item['track_id'])].append(item)
            
            matched_ids = match_tracks_to_detections(tracks_dict, current_detections)
            for det_idx, track_id in matched_ids.items():
                current_detections[det_idx]['track_id'] = track_id
    
    # 绘制检测框和Track ID
    for det in current_detections:
        track_id = det['track_id']
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        cls = det['class']
        conf = det['confidence']
        
        # 计算边界框坐标
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        # 获取颜色：有Track ID用固定颜色，没有用类别颜色
        if track_id is not None:
            color = get_track_color(track_id)
        else:
            color = class_colors.get(cls, (255, 255, 255))  # 默认白色
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本（有Track ID才显示）
        if track_id is not None:
            label = f"ID:{track_id} {class_names[cls]} {conf:.2f}"
        else:
            label = f"{class_names[cls]} {conf:.2f}"
        
        # 绘制标签背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 标签背景矩形
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(frame, label, (x1, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    # 添加帧信息
    info_text = f"Frame: {frame_num}/{len(image_files)} | Objects: {len(current_detections)}"
    cv2.putText(frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 写入视频帧
    video_writer.write(frame)
    
    # 显示进度
    if (i + 1) % 10 == 0 or i == len(image_files) - 1:
        print(f"  [{i+1}/{len(image_files)}] {image_file}: {len(current_detections)} 个目标")

# 释放资源
video_writer.release()

print("-" * 60)
print("\n✅ 视频生成完成！")
print("=" * 60)
print(f"\n📁 视频保存在: {output_video}")
print(f"📊 总帧数: {len(image_files)}")
print(f"⏱️  视频时长: {len(image_files)/fps:.1f}秒")
print(f"🎨 颜色说明:")
print(f"   - 绿色框: car (汽车)")
print(f"   - 蓝色框: pedestrian (行人)")
print(f"   - 红色框: two-wheeler (两轮车)")
print(f"   - 每个Track ID有固定颜色便于追踪")
print(f"\n💡 提示: 可以使用VLC或其他播放器查看视频")
