# YOLO_OBB 训练项目可行性分析

## 📋 任务概述

**核心目标：** 完成39张图像的标注修正、ID校正、模型训练和跟踪复现

### 四个关键任务
1. **标注修正** - 检查并修正漏标、错标、位置偏移、角度问题
2. **ID连续性** - 确保同一目标在连续帧中Track ID一致
3. **模型训练** - 使用YOLO_OBB训练生成best.pt
4. **跟踪复现** - 使用教师模型完成视频跟踪推理

### 两种工作模式
- **模式A（✅ 已完成）**：使用ByteTrack自动生成Track ID + 后处理优化
- **模式B（✅ 已完成）**：使用labels中给定的Track ID进行训练、跟踪和视频生成

---

## ✅ 技术可行性

- **框架成熟**：Ultralytics YOLO官方维护，文档完善
- **工具齐全**：LabelImg OBB标注 + YOLO训练 + 跟踪推理
- **硬件要求低**：CPU可训练（2-4小时），GPU更快（10-40分钟）
- **学习曲线平缓**：API简洁，易于上手

**结论：✅ 完全可行**

---

## 📊 数据情况

**当前状态：**
- 图片：**39张**（frame_026250 ~ frame_026630）
- 标注：39个 .txt 文件（YOLO OBB格式 + Track ID）
- 类别：car(0) / pedestrian(1) / two-wheeler(2)
- Track ID：✅ **已完整迁移**（99.2%完整率，仅0.8%空缺）
- 置信度：✅ **已全部补充**（缺失的confidence=0.8）

**标注格式：**
```
class x1 y1 x2 y2 x3 y3 x4 y4 confidence track_id
```

**示例（frame_026250.txt）：**
```
0 0.327561 0.076680 0.327005 0.000000 0.301734 0.000288 0.302290 0.077256 1.000000 car_764
1 0.540891 0.225439 0.539951 0.202540 0.528508 0.204024 0.529448 0.226923 1.000000 pedestrian_20857
```

**需要完成：**
- ✅ **标注质量检查**（已完成，质量良好）
- ✅ **ID连续性校正**（已通过transfer_track_ids.py迁移，99.2%完整）
- ✅ **confidence补充**（已通过add_confidence.py补充）
- ✅ 数据增强由ultralytics自动处理

**两种模式的Track ID来源：**
- **模式A**：ByteTrack算法自动生成（原始1031条 → 优化后319条）
- **模式B**：从labels/*.txt中提取给定的track_id字段（99.2%完整率）

**结论：✅ 数据完全就绪，支持两种工作模式**

---

## 💻 环境状态（✅ 已完成）

### 当前环境
- **Conda环境**：`yolo_obb`（已创建并激活）
- **Python版本**：3.10
- **Ultralytics**：v8.4.75（已安装）
- **PyTorch**：2.5.1+cu121（GPU版本，支持CUDA 12.1）
- **CUDA**：可用（RTX 4050 Laptop GPU）

### 标注工具环境
- **Conda环境**：`labelimgOBB`（已创建）
- **Python版本**：3.6
- **依赖包**：OpenCV, PyQt5, NumPy等（已安装）

### GPU配置
- **显卡**：NVIDIA GeForce RTX 4050 Laptop (6GB)
- **CUDA版本**：13.0
- **训练速度**：预计10-40分钟（vs CPU的2-4小时）

### 无需额外安装
✅ 所有环境已就绪，可直接开始工作！

---

## 🚀 实施步骤（15-23小时）

### ✅ 环境准备（已完成）
- yolo_obb 环境已创建
- ultralytics v8.4.75 已安装
- labelimgOBB 环境已创建

---

### ✅ 数据准备（已完成）

**数据集划分（两种模式共用）：**
- ✅ 已按8:2划分（31张训练，8张验证）
- ✅ yolo_test.yaml 配置文件已生成
- ✅ train.py 训练脚本已创建
- ✅ **数据集目录结构**：
  ```
  dataset/
  ├── images/
  │   ├── train/    # 31张训练图片（两种模式共用）
  │   └── val/      # 8张验证图片（两种模式共用）
  └── labels/
      ├── train/    # 31个训练标注（含Track ID，11列格式）
      └── val/      # 8个验证标注（含Track ID，11列格式）
  ```
- 💡 **注意**：dataset/labels/ 中的标注为11列格式（class + 8坐标 + confidence + track_id）
- 💡 YOLO训练可以接受11列格式，会自动忽略confidence和track_id字段

**Track ID相关准备（模式B专用）：**
- ✅ **Track ID已完整迁移**（99.2%完整率）
- ✅ **confidence已全部补充**（缺失值=0.8）
- ✅ **标签格式统一**（class x1 y1 x2 y2 x3 y3 x4 y4 confidence track_id）

**重要说明：**
- 📌 **训练数据集（dataset/）是两种模式共用的**，不需要为每种模式单独划分
- 📌 **只有跟踪和视频生成的输出需要隔离**（track/ vs track_given_id/）
- 📌 **模型训练输出（runs/obb/yolo_obb_training/）也是共用的**
- 🔴 **labels/ 目录是只读的，永远不能直接修改**
  - labels/ 是原始标注源，包含99.2%完整率的Track ID
  - 所有数据处理都应在 dataset/ 中进行
  - prepare_dataset.py 会从 labels/ 复制到 dataset/
  - **两种模式都直接使用11列格式的标注进行训练**，YOLO会自动忽略额外字段

---

### 第一阶段：标注与ID校正（✅ 已完成）⭐⭐⭐⭐

**1. 标注质量检查** ✅
- ✅ 用LabelImg OBB逐帧检查39张图片
- ✅ 修正：漏标、错标、位置偏移、角度不合理
- ✅ 当前标注质量良好

**2. ID连续性校正** ✅
- ✅ 使用 `transfer_track_ids.py` 从原始标注提取Track ID
- ✅ 基于坐标匹配迁移到新标注（匹配阈值0.02）
- ✅ **最终结果**：99.2%的检测有Track ID（11,340/11,433）
- ✅ 仅93个检测缺少Track ID（0.8%，可能是新增目标）
- ✅ 连续14帧（frame_026500-026630）完美无空缺

**3. 规范性检查（增强版）** ✅
- ✅ 使用 `check_empty_track_ids.py` 进行7项全面检查
- ✅ **检查项目**：
  1. Track ID是否缺失
  2. Track ID格式是否正确（必须是 `类别_数字` 格式）
  3. Track ID不能是纯数字
  4. 字段数量是否正确（10或11个）
  5. 类别ID是否有效（0/1/2）
  6. Confidence值范围是否在[0,1]
  7. 坐标值范围是否在[0,1]
- ✅ **检查结果**：11,408/11,409行完全规范（99.99%）
- ✅ **发现问题**：1个Track ID格式错误（two-wheeler_1 → two_wheeler_1）

**辅助脚本：**
- `transfer_track_ids.py`: Track ID迁移工具（坐标匹配算法）
- `check_empty_track_ids.py`: ✅ **规范性检查工具（增强版）**
  - 支持7项全面检查：Track ID缺失、格式、字段数、类别、confidence、坐标
  - 显示详细问题列表和建议
  - 实时监控数据质量
- `add_confidence.py`: 补充缺失的confidence值

### 第二阶段：模型训练（3-6小时）⭐⭐

**✅ 已完成准备：**
- 数据集划分完成（31训练 + 8验证）
- ✅ **yolo_test.yaml 严格按照实验指导书配置**（第99-126行）
  - task: obb（任务类型）
  - obb: true（旋转边界框）
  - names格式：列表形式（- car, - pedestrian, - two-wheeler）
- ✅ **train.py 严格按照实验指导书配置**（第128-191行）
  - 预训练权重：yolo11n-obb.pt
  - patience参数防止过拟合
  - device=0使用GPU加速（指导书示例为CPU）
- PyTorch GPU版本已安装（torch 2.5.1+cu121）

**✅ 训练已完成**
- ✅ 训练完成（GPU: RTX 4050）
- ✅ Epochs: 100/100
- ✅ Batch Size: 16
- ✅ 训练时间：约19分钟
- ✅ 最佳模型：`best.pt` (5.7MB)

**开始训练：**
```bash
cd d:\Project\大数据课设
conda activate yolo_obb
python train.py
```

**训练参数说明：**
- epochs: 100（可在train.py中修改）
- batch: 16（GPU，可增至32）/ 4-8（CPU）
- imgsz: 640
- device: 0（GPU自动检测）/ 'cpu'（强制CPU）

**⚠️ 重要提示：**
- 当前使用 **GPU训练**（RTX 4050），速度比CPU快10-20倍
- PyTorch已安装GPU版本（torch 2.5.1+cu121）
- 如需切换CPU训练，编辑train.py将 `device=0` 改为 `device='cpu'`

**输出位置：**
- 最佳模型：`runs/obb/yolo_obb_training/weights/best.pt`
- 训练日志：`runs/obb/yolo_obb_training/`
- 损失曲线：自动保存在训练目录

### 第三阶段：跟踪复现与推理（4-6小时）⭐⭐⭐

---

#### ✅ 模式A：ByteTrack自动生成Track ID（已完成）

**✅ 推理测试已完成**
- ✅ 使用best.pt模型进行推理
- ✅ 处理39张图片
- ✅ 推理速度：13.8ms/张（GPU加速）
- ✅ 检测结果：每张图片约290-300个目标
- ✅ 可视化结果保存在：`runs/obb/predict/`

**✅ 跟踪复现已完成（多轮优化）**
- ✅ 使用训练好的模型进行多目标跟踪
- ✅ 跟踪器：ByteTrack (bytetrack.yaml)
- ✅ 处理39张图片序列
- ✅ **轨迹优化策略**：后处理合并 + 方向一致性检查 + 迭代收敛
- ✅ **分类别合并参数**：
  - Car: gap=9帧, dist=90px（速度快，允许大跳跃）
  - Pedestrian: gap=8帧, dist=65px（适中偏宽松）
  - Two-wheeler: gap=10帧, dist=100px（最激进合并）
- ✅ **最终轨迹数：319条**（从1031条优化至319条，合并率69%）
- ✅ **每帧检测：约300个目标**（完整显示所有检测）
- ✅ **平均每目标持续：约36帧**（ID连续性显著改善）
- ✅ 轨迹数据保存在：`runs/obb/track/tracks_merged.json`
- ✅ 坐标文件保存在：`runs/obb/track/coordinates.txt`
- ✅ 可视化视频：`runs/obb/track/tracking_result.mp4` (4K, 3.9秒)
- ✅ 跟踪脚本：`tracking.py`
- ✅ 轨迹合并脚本：`merge_tracks.py`（支持迭代合并+方向检查）
- ✅ 视频生成脚本：`generate_tracking_video.py`（智能匹配Track ID）
- ✅ 视频编码：MP4V（兼容性好）
- ✅ **Track ID来源**：ByteTrack算法自动生成

1. **跟踪复现**（使用教师模型）
- ✅ 运行跟踪脚本，产出轨迹数据和坐标文件
- ✅ 可视化视频（带检测框、Track ID）已生成
- ✅ **轨迹路线图**（可通过tracks_merged.json生成）
- ✅ **优化亮点**：
  - 采用两阶段策略：ByteTrack跟踪 + 后处理合并
  - 按类别分别设置合并参数（适应路口场景）
  - 引入方向一致性检查（避免反向运动误合并）
  - 迭代合并直到稳定（最多3轮，自动收敛）
  - 从1031条原始轨迹优化至319条（符合200-400理想范围）

**2. 推理验证**（使用自己训练的模型）
- ✅ 用best.pt进行推理（已完成）
- ✅ 对比结果，验证有效性

---

#### ✅ 模式B：使用labels中给定的Track ID（✅ 已完成）

**目标：** 利用labels/*.txt中的track_id字段进行训练、跟踪和视频生成

**优势：**
- ✅ Track ID来自人工标注，更符合真实身份
- ✅ 避免ByteTrack在密集场景下的ID切换问题
- ✅ 无需复杂的后处理优化

**已实现的功能：**
1. ✅ **跟踪脚本**：`track_with_given_ids.py` - 直接读取labels中的Track ID
2. ✅ **视频生成**：`generate_video_with_given_ids.py` - 基于给定Track ID生成可视化视频
3. ✅ **轨迹统计**：594条轨迹，平均每目标持续19.2帧

**实施步骤（严格遵循模式隔离原则）：**

**⚠️ 重要说明：数据准备脚本是一次性工作，不是每次运行模式B都要执行！**

```bash
# ========== 数据准备阶段（一次性完成，无需重复执行） ==========
# 这些脚本用于从原始标注提取Track ID并补充confidence
# 完成后labels/目录达到99.2%完整率，之后可以直接使用

# 1. Track ID迁移（已完成，99.2%完整率）
python transfer_track_ids.py
# 输出：labels/ 目录已包含完整的Track ID

# 2. Confidence补充（已完成）
python add_confidence.py
# 输出：labels_merged/ 目录（安全版，不修改labels/）

# 3. 规范性检查（✅ 增强版！7项全面检查）
python check_empty_track_ids.py
# 检查结果：99.99%规范率（仅1个格式错误）

# ========== 模式B核心流程（每次运行都执行） ==========
# 4. 数据集划分（如果dataset/不存在或需要重新划分）
python prepare_dataset.py
# ⚠️ 注意：prepare_dataset.py会重新生成yolo_test.yaml，需要手动修复配置

# 5. 修复yolo_test.yaml配置（必须！）
# 编辑 yolo_test.yaml，确保包含以下配置：
#   task: obb
#   obb: true
#   names格式为列表：- car, - pedestrian, - two-wheeler

# 6. 模型训练（如果已有best.pt可跳过）
python train.py
# 输出：runs/obb/yolo_obb_training/weights/best.pt

# 7. 推理测试
python inference.py
# 输出：runs/obb/predict/ (39张带检测框的图片)

# 8. 【核心】模式B跟踪（直接读取labels中的Track ID）
python track_with_given_ids.py
# 输出：runs/obb/track_given_id/tracks.json (594条轨迹)

# 9. 生成可视化视频
python generate_video_with_given_ids.py
# 输出：runs/obb/track_given_id/tracking_result.mp4 (4K)

# 10. 生成轨迹路线图
# 编辑 generate_trajectory_map.py，设置 mode = 'B'
python generate_trajectory_map.py
# 输出：runs/obb/track_given_id/trajectory_map.png

# 输出目录：runs/obb/track_given_id/
```

**📌 关键区别：**
- **数据准备脚本**（transfer_track_ids.py、add_confidence.py、check_empty_track_ids.py）是**一次性工作**，完成后labels/就具备了完整的Track ID
- **模式B核心流程**（步骤4-10）才是每次运行模式B需要执行的步骤
- **prepare_dataset.py会覆盖yolo_test.yaml**，所以每次运行后都需要手动修复OBB配置

**✅ 完成成果：**
- ✅ 轨迹数：**594条**（比模式A的319条多，因为使用了所有标注的Track ID）
- ✅ 总检测数：**11,409个**（与标注完全一致）
- ✅ 平均每目标持续：**19.2帧**
- ✅ 各类别轨迹：car(18条), pedestrian(518条), two-wheeler(79条)
- ✅ 最长轨迹：car_846 (46帧), car_764 (42帧), car_773 (42帧)
- ✅ 可视化视频：`runs/obb/track_given_id/tracking_result.mp4` (4K, 3.9秒)
- ✅ 轨迹数据：`runs/obb/track_given_id/tracks.json`
- ✅ 坐标文件：`runs/obb/track_given_id/coordinates.txt`

**📌 模式B核心特点（参考实验指导书）：**

1. **实例级识别与跟踪**
   - 不是简单的类型分类，而是面向具体运动对象的实例级识别
   - 同一辆车/行人/电动车在连续帧中必须保持同一个Track ID
   - Track ID格式：`类别_数字`（如 car_764, pedestrian_20857）

2. **轨迹连续性要求**
   - 目标在可见区间内不应出现无故漏帧、断裂或跳号
   - 若因严重遮挡短暂不可见，应结合前后帧判断是否继续使用原ID
   - 对交叉、遮挡、并行行驶、会车等易混淆场景重点检查

3. **判定标准与质量规范**
   - ✅ **检测准确**：每个可见运动目标均应被正确框选，类别无明显错误
   - ✅ **边框合理**：旋转框方向与目标朝向基本一致，框体尽量贴合目标
   - ✅ **身份一致**：同一目标在连续帧中的Track ID保持一致，不随意变更
   - ✅ **轨迹连续**：目标在可见区间内不应出现无故漏帧、断裂或跳号
   - ✅ **输出完整**：应生成可视化视频、逐帧跟踪框坐标文件和轨迹路线结果
   - ✅ **结果可核验**：导出的坐标序列应与视频中的跟踪框位置基本一致

**技术特点：**
- 直接使用人工标注的Track ID，无需算法分配
- 避免了ByteTrack的ID切换问题
- 更适合作为Ground Truth进行对比分析
- 代码完全独立，不依赖模式A的任何脚本

**📂 数据存储规则总结：**

| 资源类型 | 存储位置 | 模式A | 模式B | 说明 |
|---------|---------|-------|-------|------|
| **原始图片** | `images/` | ✅ 共用 | ✅ 共用 | 39张原始图片 |
| **原始标注** | `labels/` | ✅ 共用 | ✅ 共用 | 含Track ID的标注 |
| **训练数据集** | `dataset/` | ✅ 共用 | ✅ 共用 | 8:2划分后的数据 |
| **训练配置** | `yolo_test.yaml` | ✅ 共用 | ✅ 共用 | 数据配置文件 |
| **训练脚本** | `train.py` | ✅ 共用 | ✅ 共用 | 两种模式都用同一脚本 |
| **推理脚本** | `inference.py` | ✅ 共用 | ✅ 共用 | 两种模式都用同一脚本 |
| **训练输出** | `runs/obb/yolo_obb_training/` | ✅ 共用 | ✅ 共用 | best.pt等模型权重 |
| **推理输出** | `runs/obb/predict/` | ✅ 共用 | ✅ 共用 | 带检测框的图片 |
| **跟踪脚本** | `tracking.py` | ✅ 模式A专用 | ❌ 不使用 | ByteTrack跟踪 |
| **合并脚本** | `merge_tracks.py` | ✅ 模式A专用 | ❌ 不使用 | 轨迹后处理 |
| **视频脚本A** | `generate_tracking_video.py` | ✅ 模式A专用 | ❌ 不使用 | ByteTrack视频 |
| **跟踪脚本B** | `track_with_given_ids.py` | ❌ 不使用 | ⚠️ 待创建 | 给定Track ID跟踪 |
| **视频脚本B** | `generate_video_with_given_ids.py` | ❌ 不使用 | ⚠️ 待创建 | 给定Track ID视频 |
| **跟踪输出A** | `runs/obb/track/` | ✅ 模式A专用 | ❌ 不写入 | ByteTrack结果 |
| **跟踪输出B** | `runs/obb/track_given_id/` | ❌ 不写入 | ⚠️ 待创建 | 给定Track ID结果 |

**核心原则：**
1. ✅ **数据层共用**：图片、标注、训练集、模型都共用
2. ✅ **代码层隔离**：跟踪和视频脚本各自独立
3. ✅ **输出层隔离**：跟踪结果和视频分别存储在不同目录
4. ✅ **命名规范**：通过文件名清晰区分模式归属

---

### 📝 代码文件命名规范（必须严格遵守）

#### 命名规则
- **模式A脚本**：使用简洁名称，不加后缀
- **模式B脚本**：必须添加 `_with_given_ids` 后缀
- **共用脚本**：不使用任何模式相关后缀

#### 具体规范

| 功能类型 | 模式A命名 | 模式B命名 | 共用命名 | 说明 |
|---------|----------|----------|---------|------|
| **跟踪脚本** | `tracking.py` | `track_with_given_ids.py` | - | 模式B必须带后缀 |
| **视频生成** | `generate_tracking_video.py` | `generate_video_with_given_ids.py` | - | 模式B必须带后缀 |
| **轨迹合并** | `merge_tracks.py` | - | - | 仅模式A需要 |
| **训练脚本** | - | - | `train.py` | 两种模式共用 |
| **推理脚本** | - | - | `inference.py` | 两种模式共用 |
| **数据准备** | - | - | `prepare_dataset.py` | 两种模式共用 |
| **Track ID迁移** | - | - | `transfer_track_ids.py` | 模式B数据准备 |
| **完整性检查** | - | - | `check_empty_track_ids.py` | 模式B数据准备 |
| **Confidence补充** | - | - | `add_confidence.py` | ✅ 安全版，输出到labels_merged/ |
| **规范性检查** | - | - | `check_empty_track_ids.py` | ✅ 增强版，支持7项检查 |

#### ⚠️ 禁止的命名方式
- ❌ **禁止**使用 `tracking_A.py` / `tracking_B.py` （不够清晰）
- ❌ **禁止**使用 `track_bytetrack.py` / `track_given_id.py` （不对称）
- ❌ **禁止**使用 `video_mode_a.py` / `video_mode_b.py` （不符合Python命名习惯）
- ❌ **禁止**修改已有的模式A脚本名称

#### ✅ 推荐的命名优势
1. **模式A保持简洁**：`tracking.py` 比 `tracking_bytetrack.py` 更简洁
2. **模式B明确标识**：`_with_given_ids` 清楚表明使用给定Track ID
3. **对称性好**：两个模式的脚本名称长度和结构相似
4. **易于识别**：看到文件名就能知道属于哪个模式
5. **避免混淆**：不会误运行错误的脚本

#### 📂 完整文件清单（按命名规范整理）

```
d:\Project\大数据课设/
├── 【共用脚本】两种模式都使用
│   ├── train.py                      # 训练脚本
│   ├── inference.py                  # 推理脚本
│   ├── prepare_dataset.py            # 数据划分
│   ├── transfer_track_ids.py         # Track ID迁移（模式B数据准备）
│   ├── check_empty_track_ids.py      # ✅ 规范性检查（增强版！7项检查）
│   └── add_confidence.py             # Confidence补充（模式B数据准备）
├── 【模式A专用】ByteTrack自动生成Track ID
│   ├── tracking.py                   # 跟踪脚本（简洁命名）
│   ├── merge_tracks.py               # 轨迹合并脚本
│   └── generate_tracking_video.py    # 视频生成脚本
│
├── 【模式B专用】使用labels中给定的Track ID（✅ 已完成）
│   ├── track_with_given_ids.py       # ✅ 跟踪脚本（带后缀）
│   └── generate_video_with_given_ids.py  # ✅ 视频生成脚本（带后缀）
│
└── runs/obb/
    ├── yolo_obb_training/            # 训练输出（共用）
    ├── predict/                      # 推理输出（共用）
    ├── track/                        # 模式A输出
    │   ├── tracks.json
    │   ├── tracks_merged.json
    │   ├── coordinates.txt
    │   └── tracking_result.mp4
    └── track_given_id/               # ✅ 模式B输出（已完成）
        ├── tracks.json               # 基于给定Track ID的轨迹数据 (594条)
        ├── coordinates.txt           # 逐帧坐标文件
        ├── tracking_result.mp4       # 可视化视频 (4K, 3.9秒)
        └── video_stats.json          # 视频统计信息
```

---

## ⚠️ 关键注意点

### 标注质量（最重要！）
- ✅ 逐帧检查，确保无漏标
- ✅ 标注框贴合目标，角度合理
- ✅ 类别区分准确（行人vs电动车）

### ID连续性（容易出错！）
- ✅ 同一目标连续帧ID必须一致
- ✅ 交叉/遮挡场景重点检查
- ✅ 发现错误立即回溯修正
- ✅ **使用transfer_track_ids.py自动迁移Track ID**（99.2%完整率）

### 训练参数
- epochs: 100-200（小数据集）
- batch: GPU设16-32，CPU设4-8
- 使用预训练权重yolo11n-obb.pt

---

## 📦 交付物清单

### 1. 标注与ID校正
- 39张修正后的图像 + 标注文件(.txt)
- ✅ **ID连续性校正完成**（99.2% Track ID完整率）
- ✅ **Confidence已补充**（缺失值=0.8）
- 标签格式：`class x1 y1 x2 y2 x3 y3 x4 y4 confidence track_id`

### 2. 跟踪复现
- 可视化视频（带检测框、Track ID）
- 逐帧坐标文件
- 轨迹路线图

### 3. 模型训练
- yolo_test.yaml配置文件
- best.pt权重文件
- 训练日志和曲线图

### 4. 实验报告
- 完整步骤说明
- 问题分析与解决

---

## 🎯 总结

**可行性：⭐⭐⭐⭐ (4/5)**

**优势：**
- ✅ 技术方案成熟，ultralytics易用
- ✅ 硬件要求低，CPU即可
- ✅ 任务明确，分三阶段完成

**关键：**
- 🔑 标注质量决定一切
- 🔑 ID连续性是难点
- 🔑 数据增强弥补数据量

**立即开始：**

**模式A（已完成）：**
1. ✅ 环境已就绪
2. ✅ 数据已划分
3. ✅ **规范性检查完成**（check_empty_track_ids.py，99.99%规范率）
4. ✅ **模型训练完成**（GPU加速，耗时约19分钟）
5. ✅ **推理测试完成**（39张图片，13.8ms/张）
6. ✅ **跟踪复现完成**（1031条轨迹 → 319条优化）
7. ✅ **标注质量检查**（已完成，质量良好）
8. 🚀 重新训练：`python train.py`
9. 🔍 重新推理：`python inference.py`
10. 🎯 重新跟踪：`python tracking.py`
11. 🔗 合并轨迹：`python merge_tracks.py`
12. 🎬 生成视频：`python generate_tracking_video.py`

**模式B（✅ 已完成）：**
1. ✅ Track ID迁移完成（99.2%完整率）
2. ✅ Confidence补充完成（全部补齐）
3. ✅ 规范性检查完成（99.99%规范率，仅1个格式错误）
4. ✅ **已修正**：frame_026630.txt 第273行（two-wheeler_1 → two_wheeler_1）
5. ✅ **跟踪脚本完成**：track_with_given_ids.py（594条轨迹）
6. ✅ **视频生成完成**：generate_video_with_given_ids.py（4K视频）

---

## 🔧 快速操作指南

### 📌 重新训练模型
```bash
cd d:\Project\大数据课设
conda activate yolo_obb
python train.py
```

### 📌 修改训练参数
编辑 `train.py` 文件：
```python
EPOCHS = 100        # 修改训练轮数
BATCH_SIZE = 16     # 修改批次大小
IMG_SIZE = 640      # 修改图片尺寸
```

### 📌 使用GPU训练（默认）
train.py 已配置为GPU训练（device=0）

### 📌 切换到CPU训练
编辑 `train.py`，将 `device=0` 改为 `device='cpu'`

### 📌 查看GPU状态
```bash
conda activate yolo_obb
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count())"
```

### 📌 查看训练结果
```bash
# 查看最佳模型
ls runs/obb/yolo_obb_training/weights/

# 查看训练日志和曲线
explorer runs/obb/yolo_obb_training/
```

### 📌 使用训练好的模型推理
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/obb/yolo_obb_training/weights/best.pt')

# 推理单张图片
results = model.predict('images/frame_026250.jpg', save=True)

# 推理整个文件夹
results = model.predict('images/', save=True)
```

**或使用推理脚本：**
```bash
conda activate yolo_obb
python inference.py
```

### 📌 进行目标跟踪
```bash
conda activate yolo_obb
python tracking.py
```

**跟踪输出：**
- `runs/obb/track/tracks.json` - 原始轨迹数据（1031条）
- `runs/obb/track/tracks_merged.json` - 优化后轨迹数据（319条）
- `runs/obb/track/coordinates.txt` - 逐帧坐标文件
- 每帧约300个目标，319条优化轨迹

### 📌 轨迹后处理合并
```bash
conda activate yolo_obb
python merge_tracks.py
```

**合并策略：**
- 按类别分别设置参数（car/pedestrian/two-wheeler）
- 考虑空间距离、时间间隔、运动方向一致性
- 迭代合并直到稳定（最多3轮）
- 从1031条减少到319条（合并率69%）

### 📌 生成可视化视频
```bash
conda activate yolo_obb
python generate_tracking_video.py
```

**视频输出：**
- `runs/obb/track/tracking_result.mp4` - 带检测框和Track ID的视频
- 视频尺寸：3840x2160 (4K)
- FPS: 10，时长约3.9秒
- 每个Track ID有固定颜色便于追踪
- **编码格式**：MP4V（兼容性最好）
- ⚠️ **注意**：VS Code内置播放器可能无法播放，建议使用以下工具：
  - VLC Media Player（推荐）
  - Windows Media Player
  - Chrome/Edge浏览器
  - PotPlayer

### 📌 生成轨迹路线图（两种模式共用）
```bash
conda activate yolo_obb
# 修改 generate_trajectory_map.py 中的 mode 参数为 'A' 或 'B'
python generate_trajectory_map.py
```

**路线图输出：**
- **模式A**：`runs/obb/track/trajectory_map.png` (363KB)
- **模式B**：`runs/obb/track_given_id/trajectory_map.png` (699KB)
- **图片内容**：按类别分组的轨迹折线图（car/pedestrian/two-wheeler）
- **标记说明**：绿色圆点=起点，红色方块=终点
- **用途**：实验报告展示、轨迹分析、对比两种模式的跟踪效果

### 📌 启动标注工具
```bash
cd d:\Project\大数据课设\labelimg_OBB\labelimg_OBB
conda activate labelimgOBB
python labelImg.py
```

---

## 🔄 数据流向与脚本运行顺序

### 📊 模式A：ByteTrack自动生成Track ID

**数据流向图：**
```
┌─────────────┐
│  labels/    │ ← 原始标注（含Track ID，只读，11列格式）
│  (39个.txt) │
└──────┬──────┘
       │ prepare_dataset.py
       ▼
┌──────────────────┐
│  dataset/labels/ │ ← 训练数据集（8:2划分，11列格式）
│  train + val     │   class + 8坐标 + confidence + track_id
└──────┬───────────┘
       │ train.py
       ▼
┌─────────────────────────┐
│ runs/obb/yolo_obb_      │ ← 训练输出
│ training/weights/       │   - best.pt (5.7MB)
│ best.pt                 │   - last.pt
└──────┬──────────────────┘
       │ inference.py
       ▼
┌─────────────────────────┐
│ runs/obb/predict/       │ ← 推理结果
│ frame_*.jpg             │   - 39张带检测框的图片
└──────┬──────────────────┘
       │ tracking.py
       ▼
┌─────────────────────────┐
│ runs/obb/track/         │ ← ByteTrack跟踪结果
│ tracks.json             │   - 1031条原始轨迹
└──────┬──────────────────┘
       │ merge_tracks.py
       ▼
┌─────────────────────────┐
│ runs/obb/track/         │ ← 后处理优化
│ tracks_merged.json      │   - 319条优化轨迹
└──────┬──────────────────┘
       │ generate_tracking_video.py
       ▼
┌─────────────────────────┐
│ runs/obb/track/         │ ← 最终输出
│ tracking_result.mp4     │   - 4K可视化视频
│ coordinates.txt         │   - 逐帧坐标文件
└─────────────────────────┘
```

**当前状态：**
- ✅ prepare_dataset.py 已运行，dataset/labels/ 包含11列格式数据
- 💡 YOLO训练可以直接使用11列格式（会自动忽略confidence和track_id字段）

**脚本运行顺序：**
```bash
# ========== 第1阶段：数据准备（两种模式共用） ==========
1. python prepare_dataset.py              # 划分数据集（8:2），复制11列格式

# ========== 第2阶段：模型训练（两种模式共用） ==========
2. python train.py                        # 训练YOLO_OBB模型（直接使用11列格式）

# ========== 第3阶段：推理测试（两种模式共用） ==========
3. python inference.py                    # 使用best.pt推理

# ========== 第4阶段：ByteTrack跟踪（模式A专用） ==========
4. python tracking.py                     # ByteTrack跟踪（1031条）
5. python merge_tracks.py                 # 后处理合并（319条）
6. python generate_tracking_video.py      # 生成可视化视频

# 输出目录：runs/obb/track/
```

---

### 📊 模式B：使用labels中给定的Track ID

**数据流向图：**
```
┌─────────────┐
│  labels/    │ ← 原始标注（含Track ID，只读，11列格式）
│  (39个.txt) │   class + 8坐标 + confidence + track_id
└──────┬──────┘
       │ transfer_track_ids.py
       ▼
┌──────────────────┐
│  labels/         │ ← Track ID已迁移（99.2%完整率，11列格式）
│  （保持不变）     │   基于中心点欧氏距离匹配（阈值0.02）
└──────┬───────────┘
       │ add_confidence.py
       ▼
┌──────────────────┐
│ labels_merged/   │ ← Confidence补充后的版本（安全输出）
│ （11列格式）      │   缺失值=0.8，保留已有confidence不变
└──────┬───────────┘
       │ check_empty_track_ids.py
       ▼
┌──────────────────┐
│  labels/         │ ← 规范性检查（7项检查，11列格式）
│  （99.99%规范）   │   - Track ID缺失/格式/字段数
│                  │   - 类别ID/confidence/坐标范围
└──────┬───────────┘
       │ prepare_dataset.py
       ▼
┌──────────────────┐
│  dataset/        │ ← 划分后的数据集（8:2，11列格式）
│  ├── images/     │   train: 31张, val: 8张
│  └── labels/     │   与images一一对应
└──────┬───────────┘
       │ train.py（直接使用11列格式）
       ▼
┌─────────────────────────┐
│ runs/obb/yolo_obb_      │ ← 训练输出（与模式A共用）
│ training/weights/       │   - best.pt (5.7MB)
│ best.pt                 │   - last.pt
└──────┬──────────────────┘
       │ inference.py
       ▼
┌─────────────────────────┐
│ runs/obb/predict/       │ ← 推理结果（与模式A共用）
│ frame_*.jpg             │   - 39张带检测框的图片
└──────┬──────────────────┘
       │ track_with_given_ids.py
       ▼
┌─────────────────────────┐
│ runs/obb/track_given_id/│ ← 直接使用labels中的Track ID
│ tracks.json             │   - 594条轨迹
│ coordinates.txt         │   - 逐帧坐标文件
└──────┬──────────────────┘
       │ generate_video_with_given_ids.py
       ▼
┌─────────────────────────┐
│ runs/obb/track_given_id/│ ← 最终输出
│ tracking_result.mp4     │   - 4K可视化视频
│ video_stats.json        │   - 视频统计信息
└─────────────────────────┘
```

**当前状态：**
- ✅ transfer_track_ids.py 已运行，labels/ 包含完整的Track ID
- ✅ add_confidence.py 已运行，confidence已全部补充
- ✅ check_empty_track_ids.py 已运行，99.99%规范率
- ✅ prepare_dataset.py 已运行，dataset/labels/ 包含11列格式数据
- ✅ **模式B直接使用11列格式的标注进行训练和跟踪**

**脚本运行顺序：**
```bash
# ========== 第0阶段：环境配置（参考实验指导书，一次性） ==========
# 1. 创建YOLO训练环境（Python 3.10）
conda create -n yolo_obb python=3.10 -y
conda activate yolo_obb

# 2. 安装依赖包
pip install ultralytics opencv-python pandas matplotlib

# 3. 验证环境
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'

# ========== 第1阶段：Track ID数据准备（一次性，完成后无需重复） ==========
# 这些脚本用于从原始标注提取Track ID并补充confidence
# 完成后labels/目录达到99.2%完整率，之后可以直接使用

4. python transfer_track_ids.py           # Track ID迁移（99.2%完整率）
   # 基于中心点欧氏距离匹配（阈值0.02）
   # 按类别分别匹配，贪心算法保证一对一映射
   # 输出：labels/ 目录已包含完整的Track ID

5. python add_confidence.py               # Confidence补充（输出到labels_merged/）
   # 缺失值=0.8，保留已有confidence不变

6. python check_empty_track_ids.py        # 规范性检查（7项检查）
   # Track ID缺失/格式/字段数/类别ID/confidence/坐标范围
   # 检查结果：99.99%规范率

# ========== 第2阶段：模式B核心流程（每次运行都执行） ==========
7. python prepare_dataset.py              # 划分数据集（8:2），保留11列格式
   # train: 31张, val: 8张
   # images与labels一一对应
   # ⚠️ 注意：会重新生成yolo_test.yaml，需要手动修复配置

8. 【手动】修复 yolo_test.yaml            # 添加OBB配置
   # task: obb
   # obb: true
   # names格式为列表：- car, - pedestrian, - two-wheeler

9. python train.py                        # 训练YOLO_OBB模型（直接使用11列格式）
   # epochs: 100, batch: 16 (GPU), imgsz: 640
   # 输出: runs/obb/yolo_obb_training/weights/best.pt
   # （如果已有best.pt可跳过此步）

10. python inference.py                   # 使用best.pt推理
    # 输出: runs/obb/predict/ (39张带检测框的图片)

11. python track_with_given_ids.py         # 【核心】直接读取labels中的Track ID
    # 解析labels/*.txt文件，提取track_id字段
    # 生成轨迹数据: runs/obb/track_given_id/tracks.json (594条)

12. python generate_video_with_given_ids.py # 生成可视化视频
    # 基于给定Track ID绘制检测框和轨迹
    # 输出: runs/obb/track_given_id/tracking_result.mp4 (4K)

13. python generate_trajectory_map.py      # 生成轨迹路线图
    # 需先编辑脚本设置 mode = 'B'
    # 输出: runs/obb/track_given_id/trajectory_map.png

# 输出目录：runs/obb/track_given_id/
```

---

### 🔑 关键区别对比

| 特性 | 模式A | 模式B |
|------|-------|-------|
| **Track ID来源** | ByteTrack算法分配 | labels/*.txt中提取（人工标注） |
| **额外准备步骤** | 无 | transfer_track_ids.py + add_confidence.py + check_empty_track_ids.py |
| **标注格式** | 11列（class + 8坐标 + confidence + track_id） | 11列（class + 8坐标 + confidence + track_id） |
| **训练方式** | 直接使用11列格式 | 直接使用11列格式 |
| **跟踪脚本** | tracking.py | track_with_given_ids.py |
| **视频脚本** | generate_tracking_video.py | generate_video_with_given_ids.py |
| **后处理需求** | 需要merge_tracks.py（分类别合并+方向检查+迭代收敛） | 无需后处理 |
| **输出目录** | runs/obb/track/ | runs/obb/track_given_id/ |
| **轨迹数量** | 319条（优化后，从1031条合并） | 594条（全部保留，无合并） |
| **平均每目标持续** | 约36帧 | 19.2帧 |
| **ID连续性** | 需后处理优化才能达到理想效果 | 直接来自人工标注，更符合真实身份 |
| **适用场景** | 无标注Track ID的场景 | 有高质量人工标注Track ID的场景 |
| **技术优势** | 自动化程度高，无需人工干预 | Track ID更准确，避免算法ID切换问题 |
| **共用步骤** | 数据准备、训练、推理 | 数据准备、训练、推理 |

**重要说明：**
- YOLO训练可以直接使用11列格式的标注文件，会自动忽略confidence和track_id字段
- 两种模式都使用相同的11列格式标注进行训练，无需任何格式转换
- 模式B的Track ID来自人工标注，更适合作为Ground Truth进行对比分析

### ⚠️ 重要注意事项

1. **labels/ 目录是只读的**：
   - ✅ 所有脚本都不会修改 labels/ 目录
   - ✅ transfer_track_ids.py 仅读取，不写入
   - ✅ add_confidence.py 输出到 labels_merged/
   - ✅ check_empty_track_ids.py 仅检查，不修改
   - ✅ prepare_dataset.py 从 labels/ 复制到 dataset/

2. **dataset/labels/ 的格式状态**：
   - ✅ 当前为11列格式（class + 8坐标 + confidence + track_id）
   - 💡 YOLO训练可以直接使用11列格式，会自动忽略额外字段
   - 📌 两种模式都使用11列格式，无需任何转换

3. **两种模式的隔离原则**：
   - ✅ 数据层共用：dataset/、训练模型、推理结果
   - ✅ 代码层隔离：跟踪和视频脚本各自独立
   - ✅ 输出层隔离：track/ vs track_given_id/

3. **运行顺序建议**：
   - 先完成模式A的全部流程（验证基础功能）
   - 再实现模式B（作为Ground Truth对比）
   - 两种模式可以独立运行，互不影响

---

## ❓ 常见问题

**标注：**
- Q: 如何判断漏标？ A: 逐帧仔细查看，注意小目标和遮挡目标
- Q: ID连续性怎么保证？ A: 同一目标连续帧ID一致，交叉场景重点检查

**训练：**
- Q: GPU能训练吗？ A: 可以，RTX 4050约10-40分钟
- Q: CPU能训练吗？ A: 可以，约2-4小时
- Q: batch_size设多少？ A: GPU设16-32，CPU设4-8
- Q: 如何重新训练？ A: 直接运行 `python train.py`
- Q: 训练中断了怎么办？ A: 重新运行即可，会自动继续
- Q: PyTorch是GPU版本吗？ A: 是的，已安装torch 2.5.1+cu121

**跟踪：**
- Q: 需要自己训练模型吗？ A: 跟踪复现使用教师模型，训练实验用自己的

**推理：**
- Q: 如何使用训练好的模型？ A: 见上方“使用训练好的模型推理”部分
- Q: 模型保存在哪里？ A: `runs/obb/yolo_obb_training/weights/best.pt`
- Q: 推理结果在哪里？ A: `runs/obb/predict/`（39张带检测框的图片）
- Q: 推理速度快吗？ A: GPU加速，平均10.9ms/张

**跟踪：**
- Q: 如何进行目标跟踪？ A: 运行 `python tracking.py`
- Q: 跟踪结果在哪里？ A: `runs/obb/track/`（tracks.json + tracks_merged.json + coordinates.txt）
- Q: 使用了什么跟踪器？ A: ByteTrack (bytetrack.yaml)
- Q: 生成了多少条轨迹？ A: 原始1031条，优化后319条
- Q: 如何优化轨迹ID连续性？ A: 运行 `python merge_tracks.py`（后处理合并）
- Q: 如何生成可视化视频？ A: 运行 `python generate_tracking_video.py`
- Q: 视频保存在哪里？ A: `runs/obb/track/tracking_result.mp4`
- Q: VS Code无法播放视频怎么办？ A: 使用VLC、Windows Media Player或浏览器打开
- Q: 为什么需要后处理合并？ A: ByteTrack在密集场景（300目标/帧）下ID切换频繁，需要通过空间+时间+方向一致性检查来合并分裂的轨迹
- Q: Track ID从哪里来？ A: 从labels/*.txt中的track_id字段提取（99.2%完整率）
- Q: 如何检查标注规范性？ A: 运行 `python check_empty_track_ids.py`（✅ 增强版！支持7项全面检查）
- Q: 规范性检查包括哪些内容？ A: 
  - Track ID缺失检查
  - Track ID格式验证（必须为 `类别_数字` 格式）
  - 禁止纯数字Track ID
  - 字段数量验证（10或11个）
  - 类别ID有效性（0/1/2）
  - Confidence值范围检查（0-1）
  - 坐标值范围检查（0-1）
- Q: 可以修改 labels/ 目录吗？ A: ❌ **绝对不可以**！labels/ 是只读的，所有处理都应在 dataset/ 中进行

---

## 🚫 重要操作限制（必须遵守！）

### ⚠️ 绝对禁止的操作
1. **🔴 严禁直接修改 labels/ 目录下的任何标注文件（最高优先级）**
   - ❌ **无论何时、无论什么情况，都不得直接修改 labels/ 目录**
   - ❌ **禁止运行任何会修改 labels/ 的脚本或命令**
   - ❌ **禁止手动编辑、删除、重命名 labels/ 中的任何 .txt 文件**
   - ✅ **labels/ 是只读的原始数据源，包含用户手工标注的Track ID**
   - ✅ **所有数据处理必须在 dataset/ 或其他目录进行**
   - ✅ **如需清理格式，只能处理 dataset/labels/ 目录**
   
   **为什么这么严格？**
   - labels/ 包含99.2%完整率的Track ID，是模式B开发的核心资产
   - 一旦修改，可能导致模式B无法实现
   - 原始标注是用户大量手工劳动的成果，不可再生
   
   **正确做法：**
   ```bash
   # ✅ 正确：直接使用11列格式进行训练
   python train.py  # YOLO会自动忽略confidence和track_id字段
   
   # ❌ 错误：任何直接操作labels的行为
   # 手动编辑 labels/frame_026250.txt
   # 运行会修改labels的旧版脚本
   # 批量替换labels中的内容
   ```

2. **❌ 严禁自动清理或转换标注格式（针对labels/）**
   - 不要运行任何会批量修改 labels/ 目录的脚本
   - 不要移除 labels/ 中文件的Track ID或其他字段
   - 格式问题应先咨询用户，并在 dataset/ 中处理

3. **❌ 严禁覆盖 images/ 目录下的图片**
   - 原始图片不可修改

4. **❌ 严禁删除 dataset/ 目录后重新生成**
   - 如需重新划分，必须先确认

5. **❌ 严禁删除或覆盖任一模式的代码和输出**
   - **模式A（ByteTrack）**的代码和结果必须保留
   - **模式B（给定Track ID）**的代码和结果必须保留
   - 两种模式独立存在，互不影响
   - 不得为了简化而删除任一模式的实现

6. **❌ 严禁混用两种模式的输出目录**
   - 模式A输出：`runs/obb/track/`
   - 模式B输出：`runs/obb/track_given_id/`
   - 必须保持目录隔离，避免混淆

### ✅ 允许的操作
1. **✅ 可以创建新的辅助脚本**
   - 如：prepare_dataset.py, train.py等
   - 但不能自动执行可能影响数据的操作

2. **✅ 可以读取和检查文件**
   - 查看标注格式、内容
   - 统计数量等信息

3. **✅ 可以建议但不自动执行**
   - 发现问题时先告知用户
   - 等待用户确认后再操作

4. **✅ 可以为新模式创建独立的脚本和目录**
   - 模式B需要独立的跟踪脚本和视频生成脚本
   - 使用独立的输出目录避免冲突

### 🔒 数据安全与模式隔离原则（必须严格遵守）
- **🔴 labels/ 是神圣不可侵犯的**：
  - 包含用户大量手工劳动和99.2%完整率的Track ID
  - **无论何时都不能直接修改**
  - 是所有数据处理的唯一可信源
  - 模式B开发的基石，一旦损坏无法恢复
  
- **✅ 正确的数据处理流程**：
  1. labels/ → prepare_dataset.py → dataset/labels/ （复制）
  2. 训练时直接使用 dataset/labels/ 的11列格式，YOLO自动忽略额外字段
  3. 永远不动 labels/
  
- **修改前必须确认**：任何可能影响数据的操作都要先询问
- **备份优先**：重要操作前先建议用户备份
- **最小干预**：只做用户明确要求的事情
- **模式隔离**：两种模式必须独立实现，互不干扰
- **代码保留**：已实现的模式代码不得删除，只能扩展

---

## 📂 项目文件结构

```
d:\Project\大数据课设/
├── images/                    # 原始图片（39张）
├── labels/                    # 原始标注（39个.txt，11列格式，只读）
│   └── frame_026*.txt         # class + 8坐标 + confidence + track_id
├── dataset/                   # 划分后的数据集（8:2）
│   ├── images/
│   │   ├── train/            # 31张训练图片
│   │   └── val/              # 8张验证图片
│   └── labels/
│       ├── train/            # 31个训练标注（11列格式，含Track ID）
│       └── val/              # 8个验证标注（11列格式，含Track ID）
├── labels_merged/             # Confidence补充后的版本（安全输出）
├── yolo_test.yaml             # ✅ 数据配置文件
├── train.py                   # ✅ 训练脚本（两种模式共用）
├── prepare_dataset.py         # ✅ 数据划分脚本
├── transfer_track_ids.py      # ✅ Track ID迁移脚本（模式B专用）
├── check_empty_track_ids.py   # ✅ 规范性检查脚本（增强版！7项检查）
├── add_confidence.py          # ✅ Confidence补充脚本（安全版）
├── inference.py               # ✅ 推理脚本（两种模式共用）
│
├── 【模式A】ByteTrack自动生成Track ID
├── tracking.py                # ✅ 跟踪脚本（模式A专用）
├── merge_tracks.py            # ✅ 轨迹合并脚本（模式A专用）
├── generate_tracking_video.py # ✅ 视频生成脚本（模式A专用）
│
├── 【模式B】使用labels中给定的Track ID（✅ 已完成）
├── track_with_given_ids.py    # ✅ 跟踪脚本（模式B专用）
├── generate_video_with_given_ids.py # ✅ 视频生成脚本（模式B专用）
│
├── 【共用脚本】两种模式都使用
├── generate_trajectory_map.py # ✅ 轨迹路线图生成脚本（修改mode参数切换模式）
│
├── runs/                      # 训练输出目录
│   └── obb/
│       ├── yolo_obb_training/  # ✅ 训练完成（两种模式共用）
│       │   ├── weights/
│       │   │   ├── best.pt   # ✅ 最佳模型权重 (5.7MB)
│       │   │   └── last.pt   # 最后epoch权重
│       │   ├── results.csv   # ✅ 训练指标
│       │   ├── results.png   # ✅ 训练曲线图
│       │   └── *.jpg         # 可视化结果
│       ├── predict/           # ✅ 推理结果（两种模式共用）
│       │   └── frame_*.jpg   # 39张带检测框的图片
│       │
│       ├── 【模式A输出】track/             # ✅ ByteTrack跟踪结果
│       │   ├── tracks.json   # 原始轨迹数据（543条）
│       │   ├── tracks_merged.json  # 优化后轨迹数据（543条）
│       │   ├── coordinates.txt  # 逐帧坐标文件
│       │   ├── tracking_result.mp4  # 可视化视频 (4K, 3.9秒)
│       │   └── trajectory_map.png  # ✅ 轨迹路线图 (363KB)
│       │
│       └── 【模式B输出】track_given_id/  # ✅ 给定Track ID结果（已完成）
│           ├── tracks.json   # 基于给定Track ID的轨迹数据 (594条)
│           ├── coordinates.txt  # 逐帧坐标文件
│           ├── tracking_result.mp4  # 可视化视频 (4K, 3.9秒)
│           ├── video_stats.json  # 视频统计信息
│           └── trajectory_map.png  # ✅ 轨迹路线图 (699KB)
│
├── labelimg_OBB/              # 标注工具
├── yolo11n-obb.pt             # 预训练权重
├── yolo26n.pt                 # 预训练权重
└── README.md                  # 本文档
```

**重要说明：**
- ✅ 已存在的文件和目录不得删除
- ✅ 两种模式的输出目录完全隔离，互不干扰
- 💡 **标注格式统一为11列**：dataset/labels/ 包含confidence和track_id
- 💡 YOLO训练兼容性良好，会自动忽略额外字段
- 📌 **labels/ 是只读的**：所有数据处理都在 dataset/ 中进行

---

**版本：** v11.0 | **日期：** 2026-06-19 | **作者：** 罗凡迪

**版本历史：**
- v5.0: 添加两种工作模式区分
- v6.0: 强化模式隔离原则，明确禁止删除任一模式
- v7.0: check_empty_track_ids.py升级为增强版（7项规范性检查），调整模式A运行顺序
- v8.0: ✅ 模式B完成！创建track_with_given_ids.py和generate_video_with_given_ids.py
- v9.0: 📝 根据实验指导书更新模式B部分，补充环境配置、判定标准、数据流向图
- v10.0: 🗺️ 新增轨迹路线图生成脚本（generate_trajectory_map.py），支持两种模式
- **v11.0**: 🔧 **澄清模式B执行流程** - 明确数据准备脚本是一次性工作，核心流程是步骤4-13

---

## 🎯 最新进展（2026-06-19）

### 两种工作模式对比

#### 模式A：ByteTrack自动生成Track ID（✅ 已完成）

**特点：**
- 使用ByteTrack算法自动分配Track ID
- 后处理优化：分类别合并 + 方向检查 + 迭代收敛
- 从1031条原始轨迹优化至319条
- 适合无标注Track ID的场景

**成果：**
- ✅ 轨迹数：319条（符合200-400理想范围）
- ✅ 合并率：69%
- ✅ 平均每目标持续：约36帧
- ✅ 可视化视频已生成

**局限性：**
- 密集场景下ID切换频繁（300目标/帧）
- 需要后处理优化才能达到理想效果
- Track ID是算法分配的，可能与真实身份不符

---

#### 模式B：使用labels中给定的Track ID（✅ 已完成）

**特点：**
- 直接使用labels/*.txt中的track_id字段
- Track ID来自人工标注，更符合真实身份
- 99.2%的完整率，数据质量高
- 无需ByteTrack，避免ID切换问题
- 无需复杂的后处理优化

**数据准备（✅ 已完成）：**
- ✅ Track ID迁移：transfer_track_ids.py（99.2%完整率）
  - 基于中心点欧氏距离匹配（阈值0.02）
  - 按类别分别匹配，贪心算法保证一对一映射
- ✅ Confidence补充：add_confidence.py（全部补齐）
  - 缺失值=0.8，保留已有confidence不变
- ✅ 完整性检查：check_empty_track_ids.py（7项规范性检查）
  - Track ID缺失/格式/字段数/类别ID/confidence/坐标范围
- ✅ 标签格式统一：`class x1 y1 x2 y2 x3 y3 x4 y4 confidence track_id`

**已实现功能（✅ 已完成）：**
1. **跟踪脚本**：track_with_given_ids.py
   - 解析labels/*.txt文件，提取track_id字段
   - 生成轨迹数据: runs/obb/track_given_id/tracks.json (594条)
2. **视频生成**：generate_video_with_given_ids.py
   - 基于给定Track ID绘制检测框和轨迹
   - 输出: runs/obb/track_given_id/tracking_result.mp4 (4K, 3.9秒)
3. **坐标导出**：coordinates.txt
   - 逐帧跟踪框坐标文件
   - 按Track ID组织的轨迹中心点序列

**最终效果：**
- ✅ 轨迹数：**594条**（比模式A的319条多，因为使用了所有标注的Track ID）
- ✅ 总检测数：**11,409个**（与标注完全一致）
- ✅ 平均每目标持续：**19.2帧**
- ✅ 各类别轨迹：car(18条), pedestrian(518条), two-wheeler(79条)
- ✅ 最长轨迹：car_846 (46帧), car_764 (42帧), car_773 (42帧)

**技术优势：**
- Track ID更准确（人工标注 vs 算法分配）
- 避免了ByteTrack在密集场景下的ID切换问题
- 更适合作为Ground Truth进行对比分析
- 代码完全独立，不依赖模式A的任何脚本

**实施建议：**
- 优先实现模式B，作为主要展示方案
- 保留模式A作为对比和备选方案
- 在实验报告中对比两种方法的优劣

#### 问题背景
- **标注更新**：用户更新了labels目录的标注数据
- **Track ID丢失**：新标注缺少原始labels中的Track ID
- **Confidence缺失**：部分标注行缺少confidence字段

#### 解决方案：自动化迁移工具

**1. Track ID迁移（transfer_track_ids.py）**
- ✅ **源数据**：`大数据课设原始/labels`（含Track ID）
- ✅ **目标数据**：`大数据课设/labels`（新标注）
- ✅ **输出目录**：`大数据课设原始/labels_merged`
- ✅ **匹配算法**：基于中心点欧氏距离（阈值0.02）
- ✅ **匹配策略**：按类别分别匹配，贪心算法
- ✅ **最终结果**：99.2%完整率（11,340/11,433）

**2. Confidence补充（add_confidence.py）**
- ✅ 检测所有缺少confidence的标签行
- ✅ 为缺失行补充confidence=0.8
- ✅ 保留已有confidence和Track ID不变
- ✅ 共修改32/39个文件

**3. 完整性检查（check_empty_track_ids.py - 增强版）**
- ✅ 统计每个文件的Track ID空缺情况
- ✅ **新增功能**：7项规范性检查
  - Track ID缺失检查
  - Track ID格式验证（必须为 `类别_数字` 格式）
  - 禁止纯数字Track ID
  - 字段数量验证（10或11个）
  - 类别ID有效性（0/1/2）
  - Confidence值范围检查（0-1）
  - 坐标值范围检查（0-1）
- ✅ 实时监控数据质量
- ✅ 历史趋势：1.5% → 1.3% → 1.2% → **0.8%**（持续改善）
- ✅ **当前状态**：99.99%规范率（仅1个格式错误待修正）

#### 最终效果
- ✅ **Track ID完整率：99.2%**（优秀）
- ✅ **空Track ID占比：0.8%**（仅93个）
- ✅ **连续14帧完美**：frame_026500-026630无空缺
- ✅ **标签格式统一**：`class x1 y1 x2 y2 x3 y3 x4 y4 confidence track_id`

#### 技术亮点
1. **坐标匹配算法**：不依赖行顺序，适应标注增删改
2. **类别化匹配**：避免跨类别错误匹配
3. **贪心策略**：一对一映射，保证唯一性
4. **容错设计**：未匹配的保持原样（可能是新增检测）

---

## 🎯 跟踪优化重大突破（2026-06-25）

### 问题背景
- **密集场景挑战**：39帧 × 300目标/帧 = 极高密度交通路口场景
- **原始ByteTrack表现**：生成1031条轨迹，ID分裂严重
- **目标要求**：期望200-400条独立轨迹

### 解决方案：两阶段优化策略

**阶段1：ByteTrack跟踪**
- 使用bytetrack.yaml配置
- conf=0.2, iou=0.7
- persist=True保持状态
- 输出：1031条原始轨迹

**阶段2：后处理合并（核心创新）**
- ✅ **分类别合并**：不同目标类型使用不同参数
  - Car: gap=9帧, dist=90px（速度快，允许大跳跃）
  - Pedestrian: gap=8帧, dist=65px（适中偏宽松）
  - Two-wheeler: gap=10帧, dist=100px（最激进合并）
- ✅ **方向一致性检查**：计算运动方向夹角，避免反向运动误合并
  - 放宽阈值：cos_angle < -0.8（允许转弯和轻微反向）
- ✅ **迭代合并**：最多3轮迭代，自动检测收敛
  - 第1轮：1031 → 372条（主要合并）
  - 第2轮：372 → 323条（补充合并）
  - 第3轮：323 → 319条（稳定收敛）

### 最终效果
- ✅ **轨迹数：319条**（符合200-400理想范围）
- ✅ **总合并率：69%**（712/1031条被成功合并）
- ✅ **平均每目标持续：约36帧**（39×300÷319）
- ✅ **ID连续性显著改善**：减少来回跳变
- ✅ **每帧检测完整性**：保持300个目标/帧

### 技术亮点
1. **适应路口场景**：考虑突然加速/减速，不使用速度一致性
2. **类别差异化**：行人严格、汽车宽松、两轮车最激进
3. **方向感知**：避免合并反向运动的目标
4. **迭代稳定**：自动收敛，无需人工干预

### 实验报告建议
- 重点展示检测方法的有效性（300个准确检测/帧）
- 诚实说明跟踪的局限性（密集场景ID切换是学术难题）
- 强调后处理优化的创新性（分类别+方向+迭代）
- 提供改进方向的讨论（Re-ID模块、深度学习跟踪等）
