"""
为标签添加缺失的confidence值（安全版）
读取 labels/ 目录，处理后保存到 labels_merged/ 目录
不会修改原始 labels/ 目录
"""
import os
import glob
import shutil

def process_label_file(file_path):
    """处理单个标签文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append('')
            continue
        
        parts = line.split()
        
        # 检查格式：class x1 y1 x2 y2 x3 y3 x4 y4 [confidence] [track_id]
        if len(parts) == 9:
            # 只有坐标，没有confidence和track_id
            # 添加confidence=0.8
            new_line = f"{line} 0.800000"
            new_lines.append(new_line)
            modified = True
        elif len(parts) == 10:
            # 有confidence但没有track_id，或者有track_id但没有confidence
            # 检查第10个字段是否是数字（confidence）还是字符串（track_id）
            try:
                float(parts[9])
                # 是数字，说明已有confidence，保持不变
                new_lines.append(line)
            except ValueError:
                # 是字符串，说明是track_id，需要在前面插入confidence
                track_id = parts[9]
                coords_part = ' '.join(parts[:9])
                new_line = f"{coords_part} 0.800000 {track_id}"
                new_lines.append(new_line)
                modified = True
        else:
            # 其他情况，保持不变
            new_lines.append(line)
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + '\n')
        return True
    return False

def main():
    # 源目录：只读，不修改
    source_labels_dir = r'd:\Project\大数据课设\labels'
    
    # 目标目录：保存处理后的结果
    target_labels_dir = r'd:\Project\大数据课设原始\labels_merged'
    
    print("=" * 60)
    print("🔧 为标签添加缺失的confidence值（安全版）")
    print("=" * 60)
    print(f"\n📂 源目录（只读）: {source_labels_dir}")
    print(f"📂 目标目录（写入）: {target_labels_dir}\n")
    
    # 确保目标目录存在
    if not os.path.exists(target_labels_dir):
        os.makedirs(target_labels_dir)
        print(f"✅ 创建目标目录: {target_labels_dir}\n")
    
    # 获取所有txt文件
    txt_files = sorted(glob.glob(os.path.join(source_labels_dir, '*.txt')))
    
    print(f"📊 找到 {len(txt_files)} 个标签文件\n")
    
    modified_count = 0
    skipped_count = 0
    
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        target_file_path = os.path.join(target_labels_dir, filename)
        
        # 先复制文件到目标目录
        shutil.copy2(file_path, target_file_path)
        
        # 然后处理目标目录中的文件
        if process_label_file(target_file_path):
            print(f"✅ {filename}: 已添加confidence")
            modified_count += 1
        else:
            print(f"⏭️  {filename}: 无需修改")
            skipped_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 完成！共处理 {len(txt_files)} 个文件")
    print(f"   - 修改: {modified_count} 个文件")
    print(f"   - 跳过: {skipped_count} 个文件")
    print(f"\n🔒 原始 labels/ 目录未被修改，保持完整")
    print(f"💾 处理结果保存在: {target_labels_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
