"""
轨迹路线图生成脚本（两种模式共用）
从tracks.json读取轨迹数据，生成可视化的轨迹折线图
支持模式A（ByteTrack）和模式B（给定Track ID）
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.font_manager import FontProperties

print("=" * 60)
print("🗺️  轨迹路线图生成器")
print("=" * 60)

# 配置参数
# 可以根据需要修改为不同的模式
mode = 'B'  # 'A' 或 'B'

if mode == 'A':
    track_data_file = 'runs/obb/track/tracks_merged.json'
    output_dir = 'runs/obb/track'
    title_prefix = '模式A - ByteTrack跟踪'
else:
    track_data_file = 'runs/obb/track_given_id/tracks.json'
    output_dir = 'runs/obb/track_given_id'
    title_prefix = '模式B - 给定Track ID'

output_image = os.path.join(output_dir, 'trajectory_map.png')

print(f"\n📂 输入文件: {track_data_file}")
print(f"📁 输出目录: {output_dir}")
print(f"🖼️  输出图片: {output_image}")
print(f"🎯 跟踪模式: {title_prefix}")

# 检查文件是否存在
if not os.path.exists(track_data_file):
    print(f"\n❌ 轨迹文件不存在: {track_data_file}")
    exit(1)

# 加载轨迹数据
print(f"\n📦 加载轨迹数据...")
with open(track_data_file, 'r', encoding='utf-8') as f:
    all_tracks = json.load(f)

print(f"   📊 共 {len(all_tracks)} 条轨迹")

# 类别映射
class_names = {0: 'car', 1: 'pedestrian', 2: 'two-wheeler'}
class_colors = {
    'car': '#2E86AB',        # 蓝色
    'pedestrian': '#A23B72', # 紫色
    'two-wheeler': '#F18F01' # 橙色
}

# 按类别分组轨迹
tracks_by_class = defaultdict(list)
for track_id, points in all_tracks.items():
    if len(points) > 0:
        cls = points[0].get('class', 0)
        class_name = class_names.get(cls, 'unknown')
        tracks_by_class[class_name].append((track_id, points))

print(f"\n📈 各类别轨迹数量:")
for cls_name, tracks in tracks_by_class.items():
    print(f"   - {cls_name}: {len(tracks)} 条")

# 创建画布
fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
fig.suptitle(f'{title_prefix} - 轨迹路线图', fontsize=16, fontweight='bold', y=0.98)

# 设置中文字体
try:
    # Windows系统
    font_prop = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=10)
except:
    # 备用方案
    font_prop = FontProperties(size=10)

# 绘制每个子图
for idx, (class_name, ax) in enumerate(zip(['car', 'pedestrian', 'two-wheeler'], axes)):
    tracks = tracks_by_class.get(class_name, [])
    
    if not tracks:
        ax.text(0.5, 0.5, f'无{class_names.get(idx, "")}轨迹', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='gray')
        ax.set_title(f'{class_names.get(idx, "")} (0条)', fontproperties=font_prop)
        continue
    
    # 设置标题
    ax.set_title(f'{class_names.get(idx, "")} ({len(tracks)}条)', 
                 fontproperties=font_prop, fontsize=12, fontweight='bold')
    
    # 绘制每条轨迹
    for track_id, points in tracks:
        if len(points) < 2:
            continue
        
        # 提取坐标
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        
        # 归一化坐标到像素范围（假设原始图片尺寸）
        # 这里使用相对坐标，实际使用时可根据需要调整
        xs_pixel = [x * 1000 for x in xs]  # 放大便于观察
        ys_pixel = [y * 1000 for y in ys]
        
        # 绘制轨迹线
        ax.plot(xs_pixel, ys_pixel, '-', color=class_colors[class_name], 
                linewidth=1.5, alpha=0.6, label=None)
        
        # 标记起点和终点
        ax.plot(xs_pixel[0], ys_pixel[0], 'o', color='green', 
                markersize=6, alpha=0.8, label='起点' if track_id == tracks[0][0] else None)
        ax.plot(xs_pixel[-1], ys_pixel[-1], 's', color='red', 
                markersize=6, alpha=0.8, label='终点' if track_id == tracks[0][0] else None)
    
    # 添加图例
    if tracks:
        ax.legend(loc='upper right', prop=font_prop, fontsize=9)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('X 坐标', fontproperties=font_prop)
    ax.set_ylabel('Y 坐标', fontproperties=font_prop)
    
    # 保持纵横比
    ax.set_aspect('equal')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
print(f"\n💾 保存轨迹路线图...")
plt.savefig(output_image, dpi=150, bbox_inches='tight', facecolor='white')
print(f"   ✅ 已保存: {output_image}")

# 显示统计信息
print(f"\n📊 轨迹统计:")
total_points = sum(len(points) for points in all_tracks.values())
avg_length = total_points / len(all_tracks) if all_tracks else 0
print(f"   - 总轨迹数: {len(all_tracks)} 条")
print(f"   - 总轨迹点数: {total_points} 个")
print(f"   - 平均轨迹长度: {avg_length:.1f} 帧")

# 找出最长轨迹
longest_track = max(all_tracks.items(), key=lambda x: len(x[1]))
print(f"   - 最长轨迹: {longest_track[0]} ({len(longest_track[1])} 帧)")

print(f"\n✅ 轨迹路线图生成完成！")
print(f"💡 提示: 可以使用图片查看器打开 {output_image}")
