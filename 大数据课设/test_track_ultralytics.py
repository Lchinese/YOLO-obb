import os
import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm

# ===================== 1. 路径配置 =====================
MODEL_PATH = r"runs/obb/yolo_obb_training/weights/best.pt"          # 已训练好的检测模型
VIDEO_PATH = r"video.mp4"        # 待预测视频
OUT_DIR = r"runs/obb/track_ultralytics"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "traj_png"), exist_ok=True)

# ===================== 2. 加载模型 =====================
print("=" * 60)
print("🎯 Ultralytics YOLO OBB 视频跟踪")
print("=" * 60)
print(f"\n📦 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"📹 打开视频: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"   视频信息: {width}x{height}  {fps:.2f}fps  共{total_frames}帧")

vis_video_path = os.path.join(OUT_DIR, "tracking_result.mp4")
writer = cv2.VideoWriter(
    vis_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 25,
    (width, height)
)

# 逐帧坐标表
records = []
# 每个 track_id 的中心点序列
track_history = defaultdict(list)

frame_idx = 0
t0 = time.time()
pbar = tqdm(total=total_frames, desc="处理帧", unit="帧", dynamic_ncols=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # persist=True 表示在连续帧之间保留跟踪状态
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    result = results[0]

    # 1) 保存带框视频
    plotted = result.plot()
    writer.write(plotted)

    # 2) 读取旋转框(OBB)、类别、ID
    det_count = 0
    if result.obb is not None and result.obb.xywhr is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()       # cx, cy, w, h, angle(rad)
        cls_ids = result.obb.cls.cpu().numpy() if result.obb.cls is not None else []
        track_ids = result.obb.id.int().cpu().tolist() if result.obb.id is not None else [-1] * len(xywhr)
        det_count = len(xywhr)

        for xywhr_row, cls_id, track_id in zip(xywhr, cls_ids, track_ids):
            cx, cy, w, h, angle = xywhr_row.tolist()

            records.append({
                "frame": frame_idx,
                "track_id": int(track_id),
                "class_id": int(cls_id),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "angle_rad": float(angle)
            })

            if track_id != -1:
                track_history[int(track_id)].append((cx, cy))

    elapsed = time.time() - t0
    avg_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
    pbar.set_postfix({"检测数": det_count, "速度": f"{avg_fps:.1f}f/s", "轨迹ID数": len(track_history)})
    pbar.update(1)
    frame_idx += 1

pbar.close()

cap.release()
writer.release()

# ===================== 3. 导出逐帧坐标表 =====================
df = pd.DataFrame(records)
csv_path = os.path.join(OUT_DIR, "coordinates.txt")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存跟踪框坐标表: {csv_path}")
print(f"✅ 已保存可视化视频: {vis_video_path}")

# ===================== 3.5 生成 tracks.json =====================
import json
print("\n💾 正在生成 tracks.json...")

# 将 records 转换为 tracks 格式 (track_id -> [{frame, x, y, w, h, angle, class}])
tracks_dict = defaultdict(list)
for rec in records:
    track_id = rec['track_id']
    if track_id != -1:
        tracks_dict[str(track_id)].append({
            'frame': rec['frame'],
            'x': rec['cx'],
            'y': rec['cy'],
            'w': rec['w'],
            'h': rec['h'],
            'angle': rec['angle_rad'],
            'class': rec['class_id']
        })

# 保存为 JSON
tracks_json_path = os.path.join(OUT_DIR, "tracks.json")
with open(tracks_json_path, 'w', encoding='utf-8') as f:
    json.dump(dict(tracks_dict), f, indent=2, ensure_ascii=False)
print(f"✅ 已保存轨迹数据: {tracks_json_path}")
print(f"   📊 共 {len(tracks_dict)} 条轨迹")

# ===================== 4. 绘制整体轨迹路线图 =====================
print("\n📊 正在绘制轨迹路线图...")
plt.figure(figsize=(10, 8))
for track_id, points in track_history.items():
    if len(points) < 2:
        continue
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.plot(xs, ys, linewidth=1.5, label=f"ID {track_id}")

plt.gca().invert_yaxis()  # 图像坐标系左上角为原点
plt.xlabel("x / pixel")
plt.ylabel("y / pixel")
plt.title("Object Trajectories")
plt.tight_layout()
traj_png_path = os.path.join(OUT_DIR, "trajectory_map.png")
plt.savefig(traj_png_path, dpi=200)
plt.close()
print(f"✅ 已保存轨迹路线图: {traj_png_path}")

print("\n" + "=" * 60)
print("🎉 视频跟踪完成！")
print("=" * 60)
print(f"\n📁 结果保存在: {OUT_DIR}/")
print(f"   - tracking_result.mp4: 可视化视频")
print(f"   - coordinates.txt: 逐帧坐标")
print(f"   - trajectory_map.png: 轨迹路线图")
print(f"\n📊 统计:")
print(f"   - 总帧数: {frame_idx}")
print(f"   - 总检测数: {len(records)}")
print(f"   - 轨迹ID数: {len(track_history)}")
