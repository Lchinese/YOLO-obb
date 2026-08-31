"""
使用训练好的YOLO OBB模型进行推理测试
"""
from ultralytics import YOLO

# 加载训练好的模型
print("=" * 60)
print("🔍 YOLO OBB 模型推理测试")
print("=" * 60)

model_path = 'runs/obb/yolo_obb_training/weights/best.pt'
print(f"\n📦 加载模型: {model_path}")
model = YOLO(model_path)

# 推理配置
conf_threshold = 0.25  # 置信度阈值
iou_threshold = 0.7    # IoU阈值

print(f"\n⚙️  推理参数:")
print(f"   - 置信度阈值: {conf_threshold}")
print(f"   - IoU阈值: {iou_threshold}")
print("\n" + "=" * 60)

# 推理整个images文件夹
print("\n🚀 开始推理 images/ 目录下的所有图片...")
results = model.predict(
    source='images/',      # 推理源
    save=True,             # 保存结果
    conf=conf_threshold,   # 置信度阈值
    iou=iou_threshold,     # IoU阈值
    show=False,            # 不显示窗口（后台运行）
    save_txt=False,        # 不保存txt标注
    save_conf=True,        # 保存置信度
)

print("\n" + "=" * 60)
print("✅ 推理完成！")
print("=" * 60)
print(f"\n📊 推理结果保存在: runs/obb/predict/")
print(f"📈 共处理 {len(results)} 张图片")

# 打印每张图片的检测结果
print("\n📋 检测统计:")
for i, result in enumerate(results):
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        print(f"  - {result.path}: {len(boxes)} 个目标")
    else:
        print(f"  - {result.path}: 无目标")

print("\n💡 提示: 可视化结果已保存在 runs/obb/predict/ 目录")
