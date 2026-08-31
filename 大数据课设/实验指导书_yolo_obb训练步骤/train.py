from ultralytics import YOLO

# 1. 载入模型，yolo26n  项目没有的话会自动下载
model = YOLO("yolo26n.pt")

# 2. 训练
model.train(
    data="ultralytics\\cfg\\datasets\\yolo_test.yaml",  # 数据集配置文件  上一步yaml文件的路径
    epochs=50,        # 你需要的轮数
    batch=8,         # CPU上一般小点 8/16/32
    imgsz=640,
    device="cpu",     # 强制cpu
    workers=2         # 可调
)