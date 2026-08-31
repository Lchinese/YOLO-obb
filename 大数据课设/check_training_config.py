"""
训练配置合规性检查脚本
严格按照实验指导书_yolo_obb训练步骤.md要求进行检查
"""
import os
import yaml

print("=" * 70)
print("🔍 YOLO OBB 训练配置合规性检查")
print("=" * 70)
print("\n参考文档：实验指导书_yolo_obb训练步骤.md")
print("-" * 70)

# 检查结果统计
checks_passed = 0
checks_total = 0

def check(condition, description, reference_line):
    """检查项"""
    global checks_passed, checks_total
    checks_total += 1
    
    if condition:
        print(f"✅ [{checks_total}] {description}")
        print(f"   📖 参考：第{reference_line}行")
        checks_passed += 1
        return True
    else:
        print(f"❌ [{checks_total}] {description}")
        print(f"   📖 参考：第{reference_line}行")
        return False

# ==================== 检查 yolo_test.yaml ====================
print("\n📄 检查 yolo_test.yaml 配置文件")
print("-" * 70)

yaml_file = 'yolo_test.yaml'
if not os.path.exists(yaml_file):
    print(f"❌ 文件不存在: {yaml_file}")
    exit(1)

with open(yaml_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 1. 检查path配置
check(
    'path' in config and config['path'],
    "数据集根目录 path 已配置",
    "106-108"
)

# 2. 检查train/val路径
check(
    'train' in config and 'val' in config,
    "训练集和验证集路径 train/val 已配置",
    "110-112"
)

# 3. 检查类别数
check(
    'nc' in config and config['nc'] == 3,
    "类别数 nc=3 正确",
    "114"
)

# 4. 检查类别名称格式（应该是列表）
names = config.get('names', {})
if isinstance(names, list):
    check(
        len(names) == 3 and names[0] == 'car' and names[1] == 'pedestrian' and names[2] == 'two-wheeler',
        "类别名称 names 使用列表格式且内容正确",
        "116-122"
    )
else:
    check(False, "类别名称 names 应使用列表格式（- car），而非字典格式（0: car）", "116-122")

# 5. 检查task配置（关键！）
check(
    'task' in config and config['task'] == 'obb',
    "任务类型 task=obb 已配置（旋转目标检测必需）",
    "124"
)

# 6. 检查obb配置（关键！）
check(
    'obb' in config and config['obb'] == True,
    "OBB参数 obb=true 已配置（旋转边界框必需）",
    "126"
)

# ==================== 检查 train.py ====================
print("\n📄 检查 train.py 训练脚本")
print("-" * 70)

train_file = 'train.py'
if not os.path.exists(train_file):
    print(f"❌ 文件不存在: {train_file}")
    exit(1)

with open(train_file, 'r', encoding='utf-8') as f:
    train_content = f.read()

# 7. 检查预训练权重
check(
    'yolo11n-obb.pt' in train_content,
    "使用yolo11n-obb.pt预训练权重（OBB检测推荐）",
    "150"
)

# 8. 检查data参数
check(
    "data='yolo_test.yaml'" in train_content or 'data="yolo_test.yaml"' in train_content,
    "数据配置文件指向yolo_test.yaml",
    "154"
)

# 9. 检查epochs参数
check(
    'epochs=' in train_content,
    "训练轮数 epochs 参数已配置",
    "156"
)

# 10. 检查batch参数
check(
    'batch=' in train_content,
    "批次大小 batch 参数已配置",
    "158"
)

# 11. 检查imgsz参数
check(
    'imgsz=' in train_content,
    "图片尺寸 imgsz 参数已配置",
    "160"
)

# 12. 检查device参数
check(
    'device=' in train_content,
    "设备 device 参数已配置（GPU或CPU）",
    "162"
)

# 13. 检查patience参数（防止过拟合）
check(
    'patience=' in train_content,
    "早停耐心值 patience 参数已配置（防止过拟合）",
    "188"
)

# 14. 检查pretrained参数
check(
    'pretrained=' in train_content,
    "预训练权重 pretrained 参数已配置",
    "172"
)

# ==================== 检查数据集结构 ====================
print("\n📄 检查数据集目录结构")
print("-" * 70)

dataset_dir = 'dataset'
check(
    os.path.exists(dataset_dir),
    "数据集根目录 dataset/ 存在",
    "85-89"
)

check(
    os.path.exists(os.path.join(dataset_dir, 'images', 'train')),
    "训练集图片目录 dataset/images/train/ 存在",
    "85-89"
)

check(
    os.path.exists(os.path.join(dataset_dir, 'images', 'val')),
    "验证集图片目录 dataset/images/val/ 存在",
    "85-89"
)

check(
    os.path.exists(os.path.join(dataset_dir, 'labels', 'train')),
    "训练集标注目录 dataset/labels/train/ 存在",
    "85-89"
)

check(
    os.path.exists(os.path.join(dataset_dir, 'labels', 'val')),
    "验证集标注目录 dataset/labels/val/ 存在",
    "85-89"
)

# 统计训练集和验证集数量
train_images = len([f for f in os.listdir(os.path.join(dataset_dir, 'images', 'train')) if f.endswith('.jpg')])
val_images = len([f for f in os.listdir(os.path.join(dataset_dir, 'images', 'val')) if f.endswith('.jpg')])
total = train_images + val_images

check(
    total == 39 and train_images == 31 and val_images == 8,
    f"数据集划分正确：训练集{train_images}张 + 验证集{val_images}张 = {total}张（8:2比例）",
    "91-97"
)

# ==================== 总结 ====================
print("\n" + "=" * 70)
print(f"📊 检查结果汇总")
print("=" * 70)
print(f"✅ 通过: {checks_passed}/{checks_total}")
print(f"❌ 失败: {checks_total - checks_passed}/{checks_total}")

if checks_passed == checks_total:
    print("\n🎉 所有检查项全部通过！配置严格符合实验指导书要求！")
    print("\n💡 下一步：运行 python train.py 开始训练")
else:
    print(f"\n⚠️  有 {checks_total - checks_passed} 项检查未通过，请修正后重新检查")
    print("\n📖 请参考：实验指导书_yolo_obb训练步骤.md")

print("=" * 70)
