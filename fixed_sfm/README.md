# ALIKED 固定关键点：独立实现的第一阶段

本项目位于 `/home/nvidia/src/roma`，独立于 `/home/nvidia/src/roma_sfm` 中的旧程序，不导入、不修改旧匹配代码。后续开发均在本项目目录进行。本阶段为每张图建立一份固定关键点库；后续 RoMa 匹配必须引用这份库中的 ID。RoMa 固定 ID 匹配已实现，见 [MATCHING.md](MATCHING.md)；COLMAP 数据库导出、原生几何验证与建图命令见 [COLMAP.md](COLMAP.md)。全局轨迹冲突过滤尚未实现。

每张图只提取一次。ID 是最终 NPZ 的行号，从 0 开始；所有配对共用这份索引。参数、权重、图像或掩膜变化时，必须建立新的输出目录。本程序没有覆盖旧关键点的选项。

## 环境与运行

使用已有兼容环境，不安装或替换 Jetson 的 PyTorch：

```bash
cd /home/nvidia/src/roma
source /home/nvidia/venvs/2dgs_env/bin/activate

# 仅列出输入，不加载 torch，不创建输出
python -m fixed_sfm.extract \
  --images-dir /home/nvidia/colmap_data/DjiData_1367/images \
  --max-images 3 --dry-run

# 先处理 3 张；默认 aliked-n16、最多 4096 点、阈值 0.2
python -m fixed_sfm.extract \
  --images-dir /home/nvidia/colmap_data/DjiData_1367/images \
  --output-dir /home/nvidia/colmap_data/DjiData_1367/fixed_aliked \
  --max-images 3 --resize-max 1024 --device cuda \
  --masks-dir /home/nvidia/colmap_data/DjiData_1367/masks

# 去掉 max-images 即处理全部；前 3 张校验后直接复用
python -m fixed_sfm.extract \
  --images-dir /home/nvidia/colmap_data/DjiData_1367/images \
  --output-dir /home/nvidia/colmap_data/DjiData_1367/fixed_aliked \
  --resize-max 1024 --device cuda \
  --masks-dir /home/nvidia/colmap_data/DjiData_1367/masks
```

也支持在程序子目录直接运行同一入口：

```bash
cd /home/nvidia/src/roma/fixed_sfm
python extract.py \
  --images-dir /home/nvidia/colmap_data/DjiData_1367/images \
  --output-dir /home/nvidia/src/roma/outputs/fixed_aliked \
  --masks-dir /home/nvidia/colmap_data/DjiData_1367/masks \
  --max-images 3 --device cuda
```

相对的 `--output-dir` 以执行命令时的工作目录为基准。例如，在 `fixed_sfm/` 中使用 `outputs/fixed_aliked` 会写到 `fixed_sfm/outputs/fixed_aliked`。使用绝对路径可让两种启动方式共享同一输出位置。

`--masks-dir` 可省略，省略后不做区域过滤。指定时要求每张图都有掩膜，白色保留、黑色排除；尺寸必须与原图一致。支持 `sub/image.png`、`sub/image.jpg.png`、`sub/image.jpg`，按此顺序查找。

权重默认读取 PyTorch Hub 的 `checkpoints/aliked-n16.pth`，本机已有 `/home/nvidia/.cache/torch/hub/checkpoints/aliked-n16.pth`。不自动下载；自定义缓存使用 `--torch-hub-dir DIR`，文件位于 `DIR/checkpoints/<model>.pth`。其他模型需提前提供对应官方权重。

`--device cpu` 可用于验证。CPU 和 CUDA 缓存使用不同配置身份，请放在不同输出目录。程序没有静默回退；CUDA 不可用时会报错。

`--image-list FILE` 可指定子集：UTF-8 文本，每行一个相对于图像根目录的名字。所有输入按文件名排序；`--max-images` 仅控制本次选择，不影响关键点库配置，可以从小批量继续到全量。

## 网格均匀提取

增加 `--grid-size 256` 启用 V2：整图只计算一次 ALIKED 特征图/分数图，然后在原图掩膜内按网格分配整图预算。不足配额不补给其他格，最终统一保存固定 ID。默认 `--grid-size 0` 保留原整图模式。V2 必须使用新的特征目录并重新匹配；V1 的 `--grid-margin` 参数保留兼容但已忽略。

完整命令、配额与坐标规则见 [GRID.md](GRID.md)。

## 输出与读取

```text
fixed_aliked/
  run_config.json               # 不变的配置、权重/实现哈希与 feature_set_id
  index.json                    # 累积已完成图像的索引
  statistics.csv                # 逐图关键点数量与 feature_id
  last_run.json                 # 本次状态与 extracted/cached 数量
  extraction.log
  features/sub/image.jpg.npz    # 保留完整图像文件名，避免 jpg/png 同名冲突
```

NPZ 数据：

| 字段 | 含义 |
|---|---|
| `image_name` | 相对于图像根目录的唯一名称 |
| `image_size` | int64 `[width,height]`，原始图像尺寸 |
| `keypoint_ids` | uint32 `[N]`，与行号相同，固定不变 |
| `keypoints` | float32 `[N,2]`，原图尺度，COLMAP/RoMa 坐标 |
| `scores` | float32 `[N]`，ALIKED 检测分数 |
| `descriptors` | 可选 float32 `[N,D]`，仅 `--save-descriptors` 时保存 |
| `feature_set_id` | 提取配置及权重、依赖实现的身份 |
| `feature_id` | 此图关键点文件内容的身份，供未来匹配文件绑定 |
| `source_sha256` / `mask_sha256` | 实际图像、掩膜内容哈希 |
| `coordinates` / `schema_version` | 明确的坐标约定、格式版本 |

默认不保存描述子，因为 RoMa 不需要它们。ALIKED 内部仍会计算描述子；如需后续 LightGlue 对照，可在第一次提取时加 `--save-descriptors`。启用该参数会改变配置，不能与已有缓存混用。

```python
from pathlib import Path
from fixed_sfm import load_features

bank = load_features(
    Path('/home/nvidia/colmap_data/DjiData_1367/fixed_aliked'),
    '1789799406000.jpg',
)
print(bank.image_size, bank.keypoints.shape, bank.feature_id)
print(bank.keypoint_ids[:5], bank.keypoints[:5], bank.scores[:5])
points_for_roma = bank.roma_keypoints()      # [-1,1]，ID 仍是原始行号
points_for_aliked = bank.aliked_keypoints()  # 如以后需要 LightGlue 输入
```

读取会校验字段、坐标、ID 及内容校验和，返回只读数组。未来匹配文件应保存两端 `feature_id`，并直接保存 `keypoint_ids` 对应；不能对过滤后的子数组重新编号，也不能在不同配对中更新固定点坐标。

## 坐标、过滤和复现约定

- 使用原始图像存储方向，不自动按 EXIF 旋转；掩膜、相机内参使用同一原图栅格。网格 V2 在整图推理后筛选，输出恢复到整图坐标。
- 默认只将长边大于 1024 的图下采样，小图不放大；`--resize-max 0` 使用原尺寸。LightGlue 的 `extract()` 将位置正确恢复到原图，保留实际横纵缩放比例和半像素映射。
- ALIKED 输出的左上像素中心为 `(0,0)`。写入库前统一加 `0.5`，使左上像素中心为 `(0.5,0.5)`，与 COLMAP 和 RoMa 的像素约定一致。后续直接写 COLMAP 时不要再次加 `0.5`；输入 RoMa 归一化坐标为 `2 * keypoints / [W,H] - 1`。
- 对 ALIKED 输出再次执行检测阈值与有效性检查，再过滤掩膜，按分数降序、y/x 升序排序，最后分配固定 ID。网格模式在分配 ID 前完成每格配额和跨格近邻处理。
- `--max-keypoints` 是整图数量上限：整图模式同时约束 ALIKED 候选；网格模式在掩膜内筛选后分配各格配额。不会用低分点或规则网格强行补足，极低纹理或全黑掩膜允许空库。
- 文件原子发布，已存在的文件不会覆盖；中断后再次运行会校验并复用。索引未及时写入时，可以从本次选中图像的完整 NPZ 恢复。单个输出目录只允许一个提取进程写入。
- 固定 ID 的保证来自复用同一份缓存，不承诺跨硬件、跨依赖版本重新推理得到逐位相同结果。图像/掩膜哈希、权重与实现哈希、设备及运行参数防止无意混用。

实现依据：[LightGlue ALIKED](https://github.com/cvg/LightGlue/blob/main/lightglue/aliked.py)、[原图坐标恢复](https://github.com/cvg/LightGlue/blob/main/lightglue/utils.py)、[HLoc 的 COLMAP 坐标转换](https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/triangulation.py)。

## 验证

```bash
# 无需 torch/GPU，验证稳定 ID、坐标、缓存完整性、续跑、配置/源文件变更、写入互斥
python -m unittest discover -s fixed_sfm/tests -v

# 用实际图像和已有权重进行 CPU 验证；与正式 CUDA 输出分开
python -m fixed_sfm.extract \
  --images-dir /home/nvidia/colmap_data/DjiData_1367/images \
  --output-dir outputs/aliked_smoke_cpu \
  --max-images 3 --resize-max 512 --max-keypoints 1024 \
  --device cpu --save-descriptors \
  --masks-dir /home/nvidia/colmap_data/DjiData_1367/masks
```
