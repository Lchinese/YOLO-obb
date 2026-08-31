"""
Track ID 迁移脚本
从原始labels提取Track ID，迁移到更新后的labels
"""
import os
import re
from collections import defaultdict

def parse_original_label(line):
    """解析原始标签格式: class x1 y1 x2 y2 x3 y3 x4 y4 confidence track_id"""
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    
    cls = int(parts[0])
    coords = [float(x) for x in parts[1:9]]
    confidence = float(parts[9]) if len(parts) > 9 else 0.0
    track_id = parts[10] if len(parts) > 10 else None
    
    return {
        'class': cls,
        'coords': coords,
        'confidence': confidence,
        'track_id': track_id
    }

def parse_new_label(line):
    """解析新标签格式: class x1 y1 x2 y2 x3 y3 x4 y4"""
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    
    cls = int(parts[0])
    coords = [float(x) for x in parts[1:9]]
    
    return {
        'class': cls,
        'coords': coords
    }

def calculate_distance(coords1, coords2):
    """计算两个OBB框的中心点距离"""
    # 计算中心点
    cx1 = sum([coords1[i] for i in [0, 2, 4, 6]]) / 4
    cy1 = sum([coords1[i] for i in [1, 3, 5, 7]]) / 4
    cx2 = sum([coords2[i] for i in [0, 2, 4, 6]]) / 4
    cy2 = sum([coords2[i] for i in [1, 3, 5, 7]]) / 4
    
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

def match_tracks(original_detections, new_detections, max_distance=0.02):
    """
    将原始Track ID匹配到新检测
    使用贪心算法，按顺序匹配
    """
    matched_track_ids = {}
    used_original_indices = set()
    
    for new_idx, new_det in enumerate(new_detections):
        best_original_idx = None
        best_distance = float('inf')
        
        for orig_idx, orig_det in enumerate(original_detections):
            if orig_idx in used_original_indices:
                continue
            
            # 检查类别是否相同
            if orig_det['class'] != new_det['class']:
                continue
            
            # 计算距离
            dist = calculate_distance(orig_det['coords'], new_det['coords'])
            
            if dist < best_distance and dist < max_distance:
                best_distance = dist
                best_original_idx = orig_idx
        
        if best_original_idx is not None:
            matched_track_ids[new_idx] = original_detections[best_original_idx]['track_id']
            used_original_indices.add(best_original_idx)
    
    return matched_track_ids

def process_single_file(original_file, new_file, output_file):
    """处理单个文件"""
    # 读取原始labels
    original_detections = []
    if os.path.exists(original_file):
        with open(original_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    det = parse_original_label(line)
                    if det and det['track_id']:
                        original_detections.append(det)
    
    # 读取新labels
    new_detections = []
    new_lines = []
    if os.path.exists(new_file):
        with open(new_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    det = parse_new_label(line)
                    if det:
                        new_detections.append(det)
                        new_lines.append(line)
    
    # 匹配Track ID
    matched_ids = match_tracks(original_detections, new_detections)
    
    # 生成输出
    output_lines = []
    for idx, (new_line, new_det) in enumerate(zip(new_lines, new_detections)):
        track_id = matched_ids.get(idx, None)
        
        if track_id:
            # 有匹配的Track ID，添加到行尾
            parts = new_line.split()
            # 添加置信度(默认为1.0)和Track ID
            output_line = f"{new_line} 1.000000 {track_id}"
        else:
            # 没有匹配，保持原样
            output_line = new_line
        
        output_lines.append(output_line)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    return len(original_detections), len(new_detections), len(matched_ids)

def main():
    original_labels_dir = r'd:\Project\大数据课设原始\labels'
    new_labels_dir = r'd:\Project\大数据课设\labels'
    output_labels_dir = r'd:\Project\大数据课设原始\labels_merged'
    
    # 创建输出目录
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 获取所有文件
    files = sorted([f for f in os.listdir(new_labels_dir) if f.endswith('.txt')])
    
    print("=" * 60)
    print("🔄 Track ID 迁移工具")
    print("=" * 60)
    print(f"\n📂 原始labels: {original_labels_dir}")
    print(f"📂 新labels: {new_labels_dir}")
    print(f"📂 输出目录: {output_labels_dir}")
    print(f"\n📊 共 {len(files)} 个文件需要处理\n")
    
    total_orig = 0
    total_new = 0
    total_matched = 0
    
    for filename in files:
        original_file = os.path.join(original_labels_dir, filename)
        new_file = os.path.join(new_labels_dir, filename)
        output_file = os.path.join(output_labels_dir, filename)
        
        orig_count, new_count, matched_count = process_single_file(
            original_file, new_file, output_file
        )
        
        total_orig += orig_count
        total_new += new_count
        total_matched += matched_count
        
        match_rate = (matched_count / new_count * 100) if new_count > 0 else 0
        print(f"✅ {filename}: 原始{orig_count}个, 新{new_count}个, 匹配{matched_count}个 ({match_rate:.1f}%)")
    
    print("\n" + "=" * 60)
    print("📊 统计总结")
    print("=" * 60)
    print(f"总原始检测数: {total_orig}")
    print(f"总新检测数: {total_new}")
    print(f"总匹配数: {total_matched}")
    print(f"总匹配率: {(total_matched/total_new*100) if total_new > 0 else 0:.1f}%")
    print(f"\n💾 结果保存在: {output_labels_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
