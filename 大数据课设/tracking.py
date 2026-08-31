"""
YOLO OBB 目标跟踪复现脚本
使用训练好的模型进行多目标跟踪，生成可视化结果和轨迹数据
"""
from ultralytics import YOLO
import os
import json
from collections import defaultdict

print("=" * 60)
print("🎯 YOLO OBB 目标跟踪复现")
print("=" * 60)

# 加载训练好的模型
model_path = 'runs/obb/yolo_obb_training/weights/best.pt'
print(f"\n📦 加载模型: {model_path}")
model = YOLO(model_path)

# 跟踪配置（针对十字路口密集场景优化）
conf_threshold = 0.35  # 进一步提高置信度阈值，过滤更多误检
iou_threshold = 0.65   # 降低IoU阈值，适应密集场景
tracker = "bytetrack.yaml"  # 使用优化后的ByteTrack配置

# ByteTrack自定义参数（通过配置文件或命令行传递）
# track_high_thresh: 高置信度阈值（用于首次匹配）
# track_low_thresh: 低置信度阈值（用于二次匹配）
# track_buffer: 轨迹缓冲帧数（目标消失后保留的帧数，越大越不容易丢失ID）

print(f"\n⚙️  跟踪参数:")
print(f"   - 置信度阈值: {conf_threshold}")
print(f"   - IoU阈值: {iou_threshold}")
print(f"   - 跟踪器: {tracker}")
print("\n" + "=" * 60)

# 获取所有图片文件（按顺序）
images_dir = 'images/'
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
print(f"\n📊 找到 {len(image_files)} 张图片")
print(f"   范围: {image_files[0]} ~ {image_files[-1]}")

# 创建输出目录
output_dir = 'runs/obb/track'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualized'), exist_ok=True)

# 存储所有轨迹数据
all_tracks = defaultdict(list)  # track_id -> [(frame, x, y, w, h, angle, class)]

print("\n🚀 开始跟踪...")
print("-" * 60)

# 执行跟踪（传入整个图片列表以维持ID连续性）
results = model.track(
    source=images_dir,  # 传入目录而非单张图片
    save=False,
    conf=conf_threshold,
    iou=iou_threshold,
    tracker=tracker,
    show=False,
    verbose=True,
    persist=True  # 保持跟踪状态
)

# 处理跟踪结果
for i, result in enumerate(results):
    image_file = image_files[i]
    
    # OBB跟踪模式下，使用obb属性而非boxes
    if not hasattr(result, 'obb') or result.obb is None:
        print(f"  [{i+1}/{len(image_files)}] {image_file}: 无目标 (obb=None)")
        continue
    
    obb = result.obb
    
    if len(obb) == 0:
        print(f"  [{i+1}/{len(image_files)}] {image_file}: 无目标")
        continue
    
    # 获取跟踪ID、坐标、类别和置信度
    track_ids = obb.id.cpu().numpy() if obb.id is not None else None
    xywhr = obb.xywhr.cpu().numpy()  # cx, cy, w, h, angle
    classes = obb.cls.cpu().numpy()
    confs = obb.conf.cpu().numpy()
    
    frame_num = i + 1
    
    # 记录轨迹数据
    for j in range(len(obb)):
        if track_ids is not None:
            track_id = int(track_ids[j])
            cx, cy, w, h, angle = xywhr[j]
            cls = int(classes[j])
            
            all_tracks[track_id].append({
                'frame': frame_num,
                'x': float(cx),
                'y': float(cy),
                'w': float(w),
                'h': float(h),
                'angle': float(angle),
                'class': cls,
                'confidence': float(confs[j])
            })
    
    print(f"  [{i+1}/{len(image_files)}] {image_file}: {len(obb)} 个目标, Track IDs: {track_ids[:5] if track_ids is not None else 'None'}...")

print("-" * 60)
print("\n✅ 跟踪完成！")

# 保存轨迹数据
print("\n💾 保存轨迹数据...")
tracks_file = os.path.join(output_dir, 'tracks.json')
with open(tracks_file, 'w', encoding='utf-8') as f:
    json.dump(all_tracks, f, indent=2, ensure_ascii=False)
print(f"   ✅ 轨迹数据: {tracks_file}")
print(f"   📊 共 {len(all_tracks)} 条轨迹")

# 生成轨迹统计
print("\n📈 轨迹统计:")
for track_id, points in sorted(all_tracks.items()):
    if len(points) > 1:
        classes = set([p['class'] for p in points])
        class_names = ['car', 'pedestrian', 'two-wheeler']
        class_str = ', '.join([class_names[c] for c in classes])
        print(f"   Track {track_id}: {len(points)} 帧, 类别: {class_str}")

# 保存逐帧坐标文件
print("\n💾 保存逐帧坐标文件...")
coords_file = os.path.join(output_dir, 'coordinates.txt')
with open(coords_file, 'w', encoding='utf-8') as f:
    f.write("# Frame TrackID Class X Y W H Angle Confidence\n")
    for track_id in sorted(all_tracks.keys()):
        for point in all_tracks[track_id]:
            class_names = ['car', 'pedestrian', 'two-wheeler']
            f.write(f"{point['frame']} {track_id} {class_names[point['class']]} "
                   f"{point['x']:.2f} {point['y']:.2f} {point['w']:.2f} "
                   f"{point['h']:.2f} {point.get('angle', 0):.4f} {point['confidence']:.4f}\n")
print(f"   ✅ 坐标文件: {coords_file}")

print("\n" + "=" * 60)
print("🎉 跟踪复现完成！")
print("=" * 60)
print(f"\n📁 结果保存在: {output_dir}/")
print(f"   - tracks.json: 轨迹数据")
print(f"   - coordinates.txt: 逐帧坐标")
print(f"\n💡 提示: 可视化图片需要单独生成")
