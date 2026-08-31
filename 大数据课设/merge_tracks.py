"""
轨迹后处理：合并相似的Track ID
通过分析轨迹的空间连续性和时间重叠，合并被错误分裂的ID
"""
import json
from collections import defaultdict

print("=" * 60)
print("🔧 轨迹后处理 - 合并相似ID")
print("=" * 60)

# 加载轨迹数据
track_file = 'runs/obb/track/tracks.json'
print(f"\n📦 加载轨迹数据: {track_file}")
with open(track_file, 'r', encoding='utf-8') as f:
    all_tracks = json.load(f)

print(f"   📊 原始轨迹数: {len(all_tracks)}")

# 按类别分组
tracks_by_class = defaultdict(list)
for tid_str, points in all_tracks.items():
    if points:
        cls = points[0]['class']
        tracks_by_class[cls].append((int(tid_str), points))

print(f"\n📋 按类别统计:")
for cls, tracks in tracks_by_class.items():
    class_names = {0: 'car', 1: 'pedestrian', 2: 'two-wheeler'}
    print(f"   {class_names.get(cls, 'unknown')}: {len(tracks)} 条轨迹")

# 合并策略：基于时空连续性的智能匹配
def should_merge(track1_points, track2_points, max_gap=2, max_distance=30, cls=1):
    """
    判断两条轨迹是否应该合并
    采用时空连续性检查：位置、时间、方向、尺寸
    
    Args:
        track1_points: 第一条轨迹的点列表
        track2_points: 第二条轨迹的点列表
        max_gap: 最大时间间隔（帧）
        max_distance: 最大空间距离（像素）
        cls: 目标类别 (0=car, 1=pedestrian, 2=two-wheeler)
    """
    if not track1_points or not track2_points:
        return False
    
    # 获取帧号范围
    frames1 = [p['frame'] for p in track1_points]
    frames2 = [p['frame'] for p in track2_points]
    
    # 检查是否有时间重叠或接近
    max_frame1, min_frame1 = max(frames1), min(frames1)
    max_frame2, min_frame2 = max(frames2), min(frames2)
    
    # 计算帧间隔
    if max_frame1 < min_frame2:
        gap = min_frame2 - max_frame1
        is_track1_first = True
    elif max_frame2 < min_frame1:
        gap = min_frame1 - max_frame2
        is_track1_first = False
    else:
        gap = 0  # 有重叠
        is_track1_first = True
    
    if gap > max_gap:
        return False
    
    # 获取交界处的点
    if is_track1_first:
        last_p1 = [p for p in track1_points if p['frame'] == max_frame1][0]
        first_p2 = [p for p in track2_points if p['frame'] == min_frame2][0]
    else:
        last_p1 = [p for p in track2_points if p['frame'] == max_frame2][0]
        first_p2 = [p for p in track1_points if p['frame'] == min_frame1][0]
    
    # ========== 1. 空间位置匹配 ==========
    distance = ((last_p1['x'] - first_p2['x'])**2 + 
                (last_p1['y'] - first_p2['y'])**2)**0.5
    
    if distance > max_distance:
        return False
    
    # ========== 2. 运动方向一致性（路口场景放宽）==========
    if len(track1_points) >= 3 and len(track2_points) >= 2:
        if is_track1_first and len(track1_points) >= 2:
            prev_p = [p for p in track1_points if p['frame'] == max_frame1-1]
            if prev_p:
                dx1 = last_p1['x'] - prev_p[0]['x']
                dy1 = last_p1['y'] - prev_p[0]['y']
                
                next_p = [p for p in track2_points if p['frame'] == min_frame2+1]
                if next_p:
                    dx2 = next_p[0]['x'] - first_p2['x']
                    dy2 = next_p[0]['y'] - first_p2['y']
                    
                    dot_product = dx1 * dx2 + dy1 * dy2
                    mag1 = (dx1**2 + dy1**2)**0.5
                    mag2 = (dx2**2 + dy2**2)**0.5
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        
                        # 路口场景：允许较大的转弯角度
                        angle_thresholds = {0: -0.6, 1: -0.5, 2: -0.7}
                        threshold = angle_thresholds.get(cls, -0.6)
                        
                        if cos_angle < threshold:
                            return False
    
    # ========== 3. 尺寸一致性检查 ==========
    if 'w' in last_p1 and 'h' in last_p1 and 'w' in first_p2 and 'h' in first_p2:
        w_ratio = last_p1['w'] / first_p2['w'] if first_p2['w'] > 0 else 1
        h_ratio = last_p1['h'] / first_p2['h'] if first_p2['h'] > 0 else 1
        
        # 尺寸变化不应超过50%
        if w_ratio < 0.5 or w_ratio > 2.0 or h_ratio < 0.5 or h_ratio > 2.0:
            return False
    
    # ========== 4. 速度合理性检查 ==========
    # 计算逐帧速度，取最大速度进行比较（适应红绿灯场景）
    if len(track1_points) >= 2 and len(track2_points) >= 2:
        # 计算track1的逐帧速度
        t1_frames = sorted(set(p['frame'] for p in track1_points))
        t1_speeds = []
        for i in range(len(t1_frames) - 1):
            p1 = [p for p in track1_points if p['frame'] == t1_frames[i]][0]
            p2 = [p for p in track1_points if p['frame'] == t1_frames[i+1]][0]
            dt = t1_frames[i+1] - t1_frames[i]
            if dt > 0:
                speed = ((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)**0.5 / dt
                t1_speeds.append(speed)
        
        # 计算track2的逐帧速度
        t2_frames = sorted(set(p['frame'] for p in track2_points))
        t2_speeds = []
        for i in range(len(t2_frames) - 1):
            p1 = [p for p in track2_points if p['frame'] == t2_frames[i]][0]
            p2 = [p for p in track2_points if p['frame'] == t2_frames[i+1]][0]
            dt = t2_frames[i+1] - t2_frames[i]
            if dt > 0:
                speed = ((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)**0.5 / dt
                t2_speeds.append(speed)
        
        # 使用最大速度进行比较（适应红绿灯启停场景）
        if t1_speeds and t2_speeds:
            max_speed1 = max(t1_speeds)
            max_speed2 = max(t2_speeds)
            
            # 最大速度比不应差异太大（不超过3倍）
            if max_speed1 > 0 and max_speed2 > 0:
                speed_ratio = max(max_speed1, max_speed2) / min(max_speed1, max_speed2)
                if speed_ratio > 3.0:
                    return False
    
    return True

# 执行合并（迭代合并直到稳定）
print("\n🔄 开始迭代合并轨迹...")
max_iterations = 3  # 最多迭代3次

for iteration in range(max_iterations):
    print(f"\n{'='*60}")
    print(f"📍 第 {iteration+1} 轮合并")
    print(f"{'='*60}")
    
    merged_count = 0
    new_tracks = {}
    used_ids = set()
    
    # 不同类别的合并参数（基于HMM优化，目标500条左右）
    merge_params = {
        0: {'max_gap': 7, 'max_distance': 60},   # car: 适中
        1: {'max_gap': 6, 'max_distance': 50},   # pedestrian: 适中
        2: {'max_gap': 10, 'max_distance': 90}    # two-wheeler: 更宽松，合并更多
    }
    
    for cls, tracks in tracks_by_class.items():
        # 获取该类别的合并参数
        params = merge_params.get(cls, {'max_gap': 5, 'max_distance': 50})
        max_gap = params['max_gap']
        max_distance = params['max_distance']
        
        class_names = {0: 'car', 1: 'pedestrian', 2: 'two-wheeler'}
        print(f"\n   📍 处理 {class_names.get(cls, 'unknown')} (gap={max_gap}, dist={max_distance})...")
        
        # 按起始帧排序
        tracks.sort(key=lambda x: min(p['frame'] for p in x[1]))
        
        cls_merged = 0
        for i, (tid1, points1) in enumerate(tracks):
            if tid1 in used_ids:
                continue
            
            merged_points = points1.copy()
            merged_tid = tid1
            
            # 查找可以合并的后续轨迹
            for j in range(i+1, len(tracks)):
                tid2, points2 = tracks[j]
                if tid2 in used_ids:
                    continue
                
                # 检查类别是否相同
                if points2[0]['class'] != cls:
                    continue
                
                # 检查是否应该合并（使用该类别的参数）
                if should_merge(merged_points, points2, max_gap=max_gap, max_distance=max_distance, cls=cls):
                    # 合并轨迹
                    merged_points.extend(points2)
                    used_ids.add(tid2)
                    cls_merged += 1
            
            # 保存合并后的轨迹
            new_tracks[str(merged_tid)] = sorted(merged_points, key=lambda x: x['frame'])
            used_ids.add(tid1)
        
        print(f"      ✅ 合并了 {cls_merged} 条轨迹")
        merged_count += cls_merged
    
    print(f"\n   📊 本轮合并后轨迹数: {len(new_tracks)}")
    
    # 更新tracks_by_class用于下一轮迭代
    tracks_by_class = defaultdict(list)
    for tid_str, points in new_tracks.items():
        if points:
            cls = points[0]['class']
            tracks_by_class[cls].append((int(tid_str), points))
    
    # 如果合并数量很少，提前退出
    if merged_count < 10:
        print(f"\n   ⚠️  合并数量较少，提前退出")
        break

# 最终结果
all_tracks = new_tracks

print(f"   ✅ 合并了 {merged_count} 条轨迹")
print(f"   📊 合并后轨迹数: {len(new_tracks)}")

# 保存结果
output_file = 'runs/obb/track/tracks_merged.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_tracks, f, ensure_ascii=False, indent=2)

print(f"\n💾 已保存: {output_file}")
print("=" * 60)
