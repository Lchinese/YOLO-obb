"""
模式B：使用labels中给定的Track ID进行跟踪
直接读取labels/*.txt中的track_id字段，生成轨迹数据
不依赖ByteTrack算法
"""
import os
import json
import glob
from collections import defaultdict

print("=" * 60)
print("🎯 模式B：使用给定Track ID进行跟踪")
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

# 获取所有标注文件（按顺序）
label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

print(f"\n📊 找到 {len(label_files)} 个标注文件")
print(f"   范围: {os.path.basename(label_files[0])} ~ {os.path.basename(label_files[-1])}")

# 存储所有轨迹数据
all_tracks = defaultdict(list)  # track_id -> [(frame, x, y, w, h, angle, class, confidence)]

# 类别映射
class_names = {0: 'car', 1: 'pedestrian', 2: 'two-wheeler'}

print("\n🚀 开始处理标注数据...")
print("-" * 60)

for i, label_file in enumerate(label_files):
    filename = os.path.basename(label_file)
    frame_num = i + 1
    
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    detections_in_frame = 0
    track_ids_in_frame = []
    
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
            
            # 计算中心点和宽高（从OBB坐标）
            x1, y1, x2, y2, x3, y3, x4, y4 = coords
            cx = (x1 + x2 + x3 + x4) / 4.0
            cy = (y1 + y2 + y3 + y4) / 4.0
            
            # 计算宽度（取最长边）
            import math
            dist_12 = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            dist_23 = math.sqrt((x3-x2)**2 + (y3-y2)**2)
            w = max(dist_12, dist_23)
            h = min(dist_12, dist_23)
            
            # 计算角度（简化处理）
            angle = math.atan2(y2-y1, x2-x1) if dist_12 > dist_23 else math.atan2(y3-y2, x3-x2)
            
            # 记录轨迹数据
            all_tracks[track_id_str].append({
                'frame': frame_num,
                'x': float(cx),
                'y': float(cy),
                'w': float(w),
                'h': float(h),
                'angle': float(angle),
                'class': cls,
                'confidence': confidence
            })
            
            detections_in_frame += 1
            track_ids_in_frame.append(track_id_str)
            
        except (ValueError, IndexError) as e:
            print(f"   ⚠️  解析错误 {filename}: {e}")
            continue
    
    # 显示进度
    unique_ids = len(set(track_ids_in_frame))
    print(f"  [{i+1}/{len(label_files)}] {filename}: {detections_in_frame} 个目标, {unique_ids} 个唯一Track ID")

print("-" * 60)
print("\n✅ 数据处理完成！")

# 保存轨迹数据
print("\n💾 保存轨迹数据...")
tracks_file = os.path.join(output_dir, 'tracks.json')
with open(tracks_file, 'w', encoding='utf-8') as f:
    json.dump(all_tracks, f, indent=2, ensure_ascii=False)
print(f"   ✅ 轨迹数据: {tracks_file}")
print(f"   📊 共 {len(all_tracks)} 条轨迹")

# 生成轨迹统计
print("\n📈 轨迹统计:")
total_detections = sum(len(points) for points in all_tracks.values())
print(f"   总检测数: {total_detections}")
print(f"   平均每条轨迹持续帧数: {total_detections / len(all_tracks):.1f}" if all_tracks else "   N/A")

# 按类别统计
class_stats = defaultdict(int)
for track_id, points in all_tracks.items():
    classes = set([p['class'] for p in points])
    for cls in classes:
        class_stats[cls] += 1

print(f"\n   各类别轨迹数:")
for cls_id, count in sorted(class_stats.items()):
    print(f"      {class_names.get(cls_id, 'unknown')}: {count} 条")

# 显示前10条长轨迹
print(f"\n   最长的10条轨迹:")
sorted_tracks = sorted(all_tracks.items(), key=lambda x: len(x[1]), reverse=True)[:10]
for track_id, points in sorted_tracks:
    classes = set([p['class'] for p in points])
    class_str = ', '.join([class_names[c] for c in classes])
    print(f"      {track_id}: {len(points)} 帧, 类别: {class_str}")

# 保存逐帧坐标文件
print("\n💾 保存逐帧坐标文件...")
coords_file = os.path.join(output_dir, 'coordinates.txt')
with open(coords_file, 'w', encoding='utf-8') as f:
    f.write("# Frame TrackID Class X Y W H Angle Confidence\n")
    for track_id in sorted(all_tracks.keys()):
        for point in all_tracks[track_id]:
            f.write(f"{point['frame']} {track_id} {class_names[point['class']]} "
                   f"{point['x']:.4f} {point['y']:.4f} {point['w']:.4f} "
                   f"{point['h']:.4f} {point.get('angle', 0):.4f} {point['confidence']:.4f}\n")
print(f"   ✅ 坐标文件: {coords_file}")

print("\n" + "=" * 60)
print("🎉 模式B跟踪完成！")
print("=" * 60)
print(f"\n📁 结果保存在: {output_dir}/")
print(f"   - tracks.json: 轨迹数据")
print(f"   - coordinates.txt: 逐帧坐标")
print(f"\n💡 下一步: 运行 generate_video_with_given_ids.py 生成可视化视频")
