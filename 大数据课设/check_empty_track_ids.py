"""
检查labels目录中标注文件的规范性
包括：Track ID缺失、字段数量、格式验证等
"""
import os
import glob
import re

def check_label_file(file_path):
    """检查单个标注文件的规范性"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = {
        'total_lines': 0,
        'empty_track_id': 0,
        'wrong_field_count': 0,
        'invalid_class': 0,
        'invalid_confidence': 0,
        'invalid_coordinate': 0,
        'invalid_track_id_format': 0,  # 新增：Track ID格式错误
        'valid_lines': 0,
        'issues': []  # 详细问题列表
    }
    
    valid_classes = {0, 1, 2}  # car=0, pedestrian=1, two-wheeler=2
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        results['total_lines'] += 1
        parts = line.split()
        has_issue = False
        
        # 检查1: 字段数量
        # 标准格式: class x1 y1 x2 y2 x3 y3 x4 y4 confidence [track_id]
        # 应该是10个字段(无track_id)或11个字段(有track_id)
        if len(parts) < 10 or len(parts) > 11:
            results['wrong_field_count'] += 1
            results['issues'].append(f"行{line_num}: 字段数错误({len(parts)}个)，应为10或11个")
            has_issue = True
            continue
        
        # 检查2: 类别ID有效性
        try:
            class_id = int(parts[0])
            if class_id not in valid_classes:
                results['invalid_class'] += 1
                results['issues'].append(f"行{line_num}: 无效类别ID {class_id}")
                has_issue = True
        except ValueError:
            results['invalid_class'] += 1
            results['issues'].append(f"行{line_num}: 类别ID不是整数 '{parts[0]}'")
            has_issue = True
        
        # 检查3: 坐标值有效性 (8个坐标点)
        coord_valid = True
        for i in range(1, 9):
            try:
                coord = float(parts[i])
                if coord < 0 or coord > 1:
                    results['invalid_coordinate'] += 1
                    results['issues'].append(f"行{line_num}: 坐标{i}超出范围 [{coord}]")
                    coord_valid = False
                    has_issue = True
                    break
            except ValueError:
                results['invalid_coordinate'] += 1
                results['issues'].append(f"行{line_num}: 坐标{i}不是数字 '{parts[i]}'")
                coord_valid = False
                has_issue = True
                break
        
        if not coord_valid:
            continue
        
        # 检查4: confidence值有效性
        try:
            confidence = float(parts[9])
            if confidence < 0 or confidence > 1:
                results['invalid_confidence'] += 1
                results['issues'].append(f"行{line_num}: confidence超出范围 [{confidence}]")
                has_issue = True
        except ValueError:
            results['invalid_confidence'] += 1
            results['issues'].append(f"行{line_num}: confidence不是数字 '{parts[9]}'")
            has_issue = True
        
        # 检查5: Track ID是否存在且有效
        if len(parts) == 10:
            # 没有Track ID
            results['empty_track_id'] += 1
            results['issues'].append(f"行{line_num}: 缺少Track ID")
            has_issue = True
        elif len(parts) == 11:
            track_id = parts[10]
            if not track_id or track_id == 'None' or track_id == '':
                results['empty_track_id'] += 1
                results['issues'].append(f"行{line_num}: Track ID为空")
                has_issue = True
            else:
                # 检查Track ID格式是否符合规范 (必须是: 类别_数字，如 pedestrian_12345)
                # 不能是纯数字
                if re.match(r'^\d+$', track_id):
                    results['invalid_track_id_format'] += 1
                    results['issues'].append(f"行{line_num}: Track ID格式错误，不能是纯数字 '{track_id}'，应为 '类别_数字' 格式")
                    has_issue = True
                elif not re.match(r'^(car|pedestrian|two_wheeler)_\d+$', track_id):
                    results['invalid_track_id_format'] += 1
                    results['issues'].append(f"行{line_num}: Track ID格式不规范 '{track_id}'，应为 '类别_数字' 格式（如 pedestrian_12345）")
                    has_issue = True
        
        if not has_issue:
            results['valid_lines'] += 1
    
    return results

def main():
    labels_dir = r'd:\Project\大数据课设\labels'
    
    print("=" * 80)
    print("🔍 检查labels目录中标注文件的规范性")
    print("=" * 80)
    print(f"\n📂 检查目录: {labels_dir}\n")
    
    # 获取所有txt文件
    txt_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
    
    print(f"📊 共 {len(txt_files)} 个文件\n")
    
    # 汇总统计
    grand_total = {
        'total_lines': 0,
        'empty_track_id': 0,
        'wrong_field_count': 0,
        'invalid_class': 0,
        'invalid_confidence': 0,
        'invalid_coordinate': 0,
        'invalid_track_id_format': 0,
        'valid_lines': 0,
        'files_with_issues': 0
    }
    
    all_issues = {}  # 存储所有文件的问题
    
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        results = check_label_file(file_path)
        
        # 累加统计
        for key in grand_total:
            if key != 'files_with_issues':
                grand_total[key] += results[key]
        
        # 记录有问题的文件
        if (results['empty_track_id'] > 0 or 
            results['wrong_field_count'] > 0 or
            results['invalid_class'] > 0 or
            results['invalid_confidence'] > 0 or
            results['invalid_coordinate'] > 0 or
            results['invalid_track_id_format'] > 0):
            grand_total['files_with_issues'] += 1
            all_issues[filename] = results
    
    # 打印总体统计
    print("=" * 80)
    print("📊 总体统计")
    print("=" * 80)
    print(f"总行数:              {grand_total['total_lines']}")
    print(f"✅ 完全规范的行数:   {grand_total['valid_lines']} ({grand_total['valid_lines']/grand_total['total_lines']*100:.2f}%)")
    print(f"❌ 缺少Track ID:     {grand_total['empty_track_id']} ({grand_total['empty_track_id']/grand_total['total_lines']*100:.2f}%)")
    print(f"⚠️  Track ID格式错误: {grand_total['invalid_track_id_format']} ({grand_total['invalid_track_id_format']/grand_total['total_lines']*100:.2f}%)")
    print(f"⚠️  字段数错误:       {grand_total['wrong_field_count']}")
    print(f"⚠️  类别ID错误:       {grand_total['invalid_class']}")
    print(f"⚠️  Confidence错误:   {grand_total['invalid_confidence']}")
    print(f"⚠️  坐标值错误:       {grand_total['invalid_coordinate']}")
    print(f"\n有问题的文件数:      {grand_total['files_with_issues']}/{len(txt_files)}")
    print("=" * 80)
    
    # 打印有问题的文件详情
    if all_issues:
        print("\n" + "=" * 80)
        print("📋 有问题的文件详情")
        print("=" * 80)
        
        for filename, results in sorted(all_issues.items()):
            print(f"\n📄 {filename}:")
            print(f"   总行数: {results['total_lines']}")
            print(f"   缺少Track ID: {results['empty_track_id']} 行")
            if results['wrong_field_count'] > 0:
                print(f"   字段数错误: {results['wrong_field_count']} 行")
            if results['invalid_class'] > 0:
                print(f"   类别ID错误: {results['invalid_class']} 行")
            if results['invalid_confidence'] > 0:
                print(f"   Confidence错误: {results['invalid_confidence']} 行")
            if results['invalid_coordinate'] > 0:
                print(f"   坐标值错误: {results['invalid_coordinate']} 行")
            if results['invalid_track_id_format'] > 0:
                print(f"   Track ID格式错误: {results['invalid_track_id_format']} 行")
            
            # 显示所有问题（不再限制为5条）
            if results['issues']:
                print(f"   具体问题:")
                for issue in results['issues']:
                    print(f"      - {issue}")
    
    print("\n" + "=" * 80)
    print("💡 建议:")
    if grand_total['empty_track_id'] > 0:
        print(f"   1. 使用标注工具补全 {grand_total['empty_track_id']} 个缺失的Track ID")
    if grand_total['invalid_track_id_format'] > 0:
        print(f"   2. 修正 {grand_total['invalid_track_id_format']} 个Track ID格式错误（应为 '类别_数字' 格式，如 pedestrian_12345）")
    if grand_total['wrong_field_count'] > 0:
        print(f"   3. 检查并修复 {grand_total['wrong_field_count']} 个字段数错误的行")
    if grand_total['invalid_class'] > 0:
        print(f"   4. 修正 {grand_total['invalid_class']} 个无效的类别ID")
    if grand_total['invalid_confidence'] > 0:
        print(f"   5. 修正 {grand_total['invalid_confidence']} 个无效的confidence值")
    if grand_total['invalid_coordinate'] > 0:
        print(f"   6. 修正 {grand_total['invalid_coordinate']} 个超出范围的坐标值")
    if grand_total['valid_lines'] == grand_total['total_lines']:
        print("   ✅ 所有标注文件都符合规范！")
    print("=" * 80)

if __name__ == '__main__':
    main()
