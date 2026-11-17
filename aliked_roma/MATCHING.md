# RoMa 固定关键点匹配

本阶段读取 ALIKED 或 DeDoDe 的固定关键点库，输出两端原始关键点 ID 的对应关系。不重新采样关键点、不修改坐标、不合并或重新编号，也不依赖旧 `roma_sfm/src` 程序。

检测器从 `--features-dir` 的元数据自动识别，无需另设匹配模型类型，也不重新加载 ALIKED/DeDoDe。匹配查询全部已保存点，不使用检测分数再次过滤；`--min-confidence` 始终针对 RoMa certainty。点库来源及完整提取设置写入匹配目录的 `feature_bank.json`，现有 ALIKED 匹配缓存身份规则保持不变。

DeDoDe 运行示例与已完成的 12 张图、24 对匹配验证见 [DEDODE.md](DEDODE.md)。这 12 张图来自四个相距较远的局部拍摄段，使用明确的 `--pairs-file`，每组三张图内部全配对，不按子集名称顺序跨组配对。

流程：两次顺序 RoMa 推理（A→B、B→A）→ 查询全部固定关键点 → 原图像素距离分配 → 双向 ID 一致性和前后向 warp 循环检查 → 固定坐标上的 MAGSAC F 验证 → 原子保存 NPZ。

## 运行

在项目根目录和 `aliked_roma` 子目录都可启动。以下以子目录为例，使用绝对路径避免输出位置随工作目录变化：

```bash
cd /home/nvidia/src/roma/aliked_roma
source /home/nvidia/venvs/2dgs_env/bin/activate

# 如果尚未建立关键点库，先提取三张图；已有库则直接复用。
python extract.py \
  --images-dir /home/nvidia/colmap_data/DjiData_1367/images \
  --output-dir /home/nvidia/src/roma/outputs/fixed_aliked \
  --masks-dir /home/nvidia/colmap_data/DjiData_1367/masks \
  --max-images 3 --device cuda

# 检查配对、关键点文件和原始图像是否存在；不加载 RoMa、不创建输出。
python match.py \
  --features-dir /home/nvidia/src/roma/outputs/fixed_aliked \
  --output-dir /home/nvidia/src/roma/outputs/roma_matches \
  --max-pairs 3 --dry-run

# 实际匹配；images-dir 默认读取关键点库内记录的原始图像目录。
python match.py \
  --features-dir /home/nvidia/src/roma/outputs/fixed_aliked \
  --output-dir /home/nvidia/src/roma/outputs/roma_matches \
  --weights-dir /home/nvidia/models/RoMa \
  --roma-repo-dir /home/nvidia/src/RoMa \
  --max-pairs 3 --device cuda
```

从项目根目录运行时，入口改为 `python -m aliked_roma.match`，其他参数相同。提取时可以用其他 `--output-dir`；匹配时将 `--features-dir` 指向那个实际目录。DeDoDe 点库和不同选点策略需要各自独立的匹配输出目录。

默认在关键点库索引中的图像间按名称排序，使用连续配对 `--pairing-mode linear --overlap 5`。支持三种候选配对模式：

| 模式 | `--overlap 5` 时的前向索引间隔 | `overlap=K` 的含义 |
|---|---|---|
| `linear`（默认） | 1、2、3、4、5 | 连续窗口 1 到 K |
| `exponential` | 1、2、4、8、16 | K 个指数层级，2^0 到 2^(K-1) |
| `hybrid` | 1、2、3、4、5、8、16 | 上述两者合并并去重 |

指数模式对应本机及当前 COLMAP 的 `quadratic_overlap=true` 配对语义。也可直接使用别名 `--quadratic-overlap`，等价于 `--pairing-mode exponential`；两种模式参数不能同时填写。这里的间隔是名称排序后的图像索引差，不是时间戳差。超出图像数量的间隔会被截断，不会回绕、自匹配或重复匹配。

例如 K=10 时，指数模式的候选间隔为 1、2、4、8、16、32、64、128、256、512：

```bash
python match.py \
  --features-dir /home/nvidia/src/roma/outputs/fixed_aliked \
  --output-dir /home/nvidia/src/roma/outputs/roma_matches \
  --pairing-mode exponential --overlap 10 \
  --max-pairs 0 --device cuda
```

先在同一命令末尾加 `--dry-run` 可以检查当前关键点库实际支持的间隔和配对数量。需要保留所有近邻时，将模式改为 `hybrid`。

`--pairs-file FILE` 可指定配对，每行两个相对图像名；指定后以文件为准，不再增加任何模式生成的图像对。反向重复会合并，自配对和缺少固定关键点的图像会报错。`--max-pairs` 按去重并排序后的配对取前 N 对；设为 0 或省略则处理全部。

模式、overlap、max-pairs 只控制选择哪些图像对，不改变某一对的匹配算法。因此可在同一个匹配输出目录切换模式，已有对应对会复用，只计算新增图像对。`pairing_config.json` 单独记录本次配对模式与间隔；它不进入不可变的匹配配置校验。`index.json` 累积保留完成的配对，`pairs_used.json` 和 `association_stats.json` 对应本次选择的范围。先完成更多图像的关键点提取后，指数间隔才会实际扩展到更远的图像。

实现参照：[COLMAP SequentialPairGenerator](https://github.com/colmap/colmap/blob/main/src/colmap/controllers/pairing.cc)。

只使用本地 `roma_outdoor.pth` 和 `dinov2_vitl14_pretrain.pth`，不会下载权重或安装/替换 PyTorch。`romatch` 未安装时，可通过 `--roma-repo-dir` 加载已有官方仓库。

## 过滤与坐标约定

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--coarse-res` | 448 | RoMa 粗分辨率，须为 14 的倍数 |
| `--upsample-res` | 672 | RoMa 上采样分辨率 |
| `--max-assignment-distance` | 3 | 预测位置到目标固定关键点的距离，原图像素 |
| `--nn-ratio` | 0.8 | 最近与次近关键点的空间距离比，拒绝有歧义的分配 |
| `--min-confidence` | 0.2 | 来回路径上 RoMa certainty 的最低值 |
| `--max-cycle-error` | 3 | 未吸附到关键点的 warp 来回误差，原图像素 |
| `--ransac-threshold` | 2 | 在两端固定关键点坐标上估计 F 的像素阈值 |
| `--min-pair-inliers` | 15 | 判为 `ok` 的最低内点数；不足的配对仍保存并标记 |

最近邻搜索在原始图像像素坐标中进行，因此不会把非方形图像的 x/y 距离当成同尺度归一化距离。两方向都要满足距离、置信度和循环误差约束，并且 A:i 指向 B:j 时，B:j 必须指回 A:i，因此输出是单对一对一。

循环检查使用真实反向 RoMa warp，并检查未吸附的预测位置。它与“最近邻互选”是两个独立条件。RoMa certainty 保留原始概率，取对应来回查询的最小值；没有调用会改写置信度的 `sample()`。

固定关键点已经采用原图尺度的 COLMAP/RoMa 坐标，左上像素中心为 `(0.5,0.5)`。查询采用 `2*points/[W,H]-1` 与 `align_corners=False`；保存的是固定关键点 ID，不是 flow 预测坐标。后续写 COLMAP 不应再次加 0.5。

## 输出

```text
roma_matches/
  run_config.json          # 关键点库、模型权重/源码、过滤参数与 run_id
  feature_bank.json        # 点库来源、检测器设置、检测分数与匹配分数的语义
  pairs_used.json          # 本次选择的规范方向配对，名称按字典序排列
  pairing_config.json      # 本次配对模式、overlap、实际索引间隔
  index.json              # 累积已完成配对索引及质量分类
  statistics.csv          # 候选/互选/内点数量与状态
  association_stats.json  # 本次所选配对的 F 内点图，统计关键点复用与孤立双视图比例
  last_run.json
  matching.log
  matches/<SHA256>.npz
```

每份 NPZ 包含：

- `matches`：uint32 `[M,2]`，两端固定关键点的原始行号，未重编号。
- `scores`：float32 `[M]`，该对应的来回查询最低置信度。
- `assignment_error_px`：float32 `[M,2]`，两方向预测到固定关键点的距离。
- `cycle_error_px`：float32 `[M,2]`，两方向未吸附预测的循环误差。
- `inliers`：bool `[M]`，初始 F 验证结果；输出保留全部互选候选，使用此掩码选择内点。
- `fundamental`：可选 float64 `[3,3]`，没有估计成功时不保存该字段。
- `metadata_json`：图像名、两端 `feature_id`、`feature_set_id`、关键点总数、匹配 `run_id` 和候选统计。
- `schema_version`、`content_id`：格式版本及结果内容校验值。

```python
from pathlib import Path
from aliked_roma import load_features
from aliked_roma.match_store import load_matches

features = Path('/home/nvidia/src/roma/outputs/fixed_aliked')
matches = Path('/home/nvidia/src/roma/outputs/roma_matches')
a, b = '1789799406000.jpg', '1789799408000.jpg'
pair = load_matches(matches, a, b)  # 始终返回图像名按字典序排列的规范方向
a, b = pair.metadata['images']
f0, f1 = load_features(features, a), load_features(features, b)
pair.bind(f0, f1)                 # 检查确实引用这两份固定关键点
ids = pair.matches[pair.inliers]
points0 = f0.keypoints[ids[:, 0]]  # 使用原始 ID 读取，不能对子集重新编号
points1 = f1.keypoints[ids[:, 1]]
```

掩膜过滤已固定在关键点库中，匹配阶段不重新过滤或编号。每次读取源图像都会检查内容哈希，确保 RoMa 看到的是提取关键点时的那份图像。

重跑会检查配置、结果内容校验值和两端关键点身份，然后复用完整 NPZ，不重新加载模型推理。参数、权重、RoMa 源码或关键点内容变化时，必须使用新的匹配输出目录。中断后可恢复已经原子保存的配对；损坏或不兼容的缓存会报错，不会静默重算覆盖。空关键点库、少于 8 条对应或几何失败会保存空/无内点结果并标记 `insufficient_inliers`。

`association_stats.json` 的统计范围仅是本次选择的配对，少量图像的双视图比例不能直接代表全量数据。它统计共享固定 ID 的情况，不证明关联一定属于同一个物理点。

COLMAP 数据库导出已实现，入口为 `python -m aliked_roma.export_colmap`，完整命令与格式见 [COLMAP.md](COLMAP.md)。保存的 F 内点掩码不能直接充当 COLMAP 的标定两视图几何；导出后仍需执行 COLMAP 的 E/F/H 验证，再进行 mapper。当前尚未实现全局轨迹冲突过滤。

实现参考：[RoMa 官方 matcher](https://github.com/Parskatt/RoMa/blob/main/romatch/models/matcher.py)。本程序采用原图像素最近邻和显式反向 warp 检查，因此没有直接使用默认归一化距离阈值的 `match_keypoints()`。

## 匹配可视化

无需 GPU，直接读取已有关键点和匹配文件，不重新匹配或修改固定 ID：

```bash
cd /home/nvidia/src/roma/aliked_roma
python visualize.py \
  --features-dir /home/nvidia/src/roma/outputs/fixed_aliked \
  --matches-dir /home/nvidia/src/roma/outputs/roma_matches \
  --max-pairs 3 --max-lines 100
```

项目根目录也可使用 `python -m aliked_roma.visualize`。输出默认保存到匹配目录下的 `visualizations/`，可用 `--output-dir` 修改。

- `matches_overview.jpg`：最多六对的总览。每对内点按空间分布抽样，颜色表示 RoMa 置信度；红色显示 F 验证剔除的对应。
- `pair_*.png`：各图像对的独立高清图。
- `three_view_correspondences.png`：存在三图闭环时，显示一组空间分散的对应。同色同编号表示同一组三图对应，最多显示 `--max-tracks 24` 组。
- `three_view_correspondences.csv`：所选三图中全部闭环对应及原始点库 ID；图中的显示编号只是本表行号，不会改变关键点 ID。
- `visualization_manifest.json`：实际总数、抽样数及输出文件。

可用 `--pair IMAGE0 IMAGE1` 指定某一对。图中连线经过抽样，不能据显示线数推断全部匹配数量；总数在标题和清单中给出。
