from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    # 使用绝对路径，避免在不同工作目录下运行时报找不到文件
    root = Path(__file__).resolve().parent
    data_cfg = root / "yolo_test.yaml"

    if not data_cfg.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {data_cfg}")

    # OBB 检测建议使用 -obb 预训练权重，使用 small 模型提升精度
    model = YOLO("yolo11s-obb.pt")

    model.train(
        data=str(data_cfg),  # 数据集配置文件
        epochs=100,  # 增加训练轮数：50 → 100
        batch=8,  # 降低batch：16 → 8，适配yolo11s+800分辨率的显存需求
        imgsz=800,  # 增大图像尺寸：640 → 800，提升小目标检测
        device=0,  # 使用GPU训练
        workers=4,  # 数据加载workers
        project=str(root / "runs" / "obb"),
        name="yolo_obb_training",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        # 优化参数
        augment=True,  # 启用数据增强
        mosaic=1.0,  # Mosaic增强概率
        mixup=0.1,  # Mixup增强
        copy_paste=0.1,  # Copy-Paste增强
        patience=50,  # 早停耐心值（epochs无改善则停止）
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率比例
        momentum=0.937,  # SGD动量/Adam beta1
        weight_decay=0.0005,  # 权重衰减
        close_mosaic=10,  # 最后10个epoch关闭mosaic
    )


if __name__ == "__main__":
    main()
