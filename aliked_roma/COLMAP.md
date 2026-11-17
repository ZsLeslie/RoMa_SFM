# 固定关键点与 RoMa 匹配导出 COLMAP

入口：项目根目录运行 `python -m fixed_sfm.export_colmap`，或进入 `fixed_sfm/` 后运行 `python export_colmap.py`。使用已有的 `2dgs_env`，无需 GPU，也不重新提取或匹配。

## 本数据集的完整命令

本次输入目录为 `outputs/aliked`、`outputs/roma_matches`。以下 PINHOLE 参数读取自 `/home/nvidia/colmap_data/DjiData_1367/sparse/0/cameras.bin`：原图 1920×1439，参数依次为 `fx,fy,cx,cy`。换数据集、裁剪或缩放图像时，必须换成对应内参。

```bash
cd /home/nvidia/src/roma
source /home/nvidia/venvs/2dgs_env/bin/activate

# 可选：全量校验并统计，不写输出
python -m fixed_sfm.export_colmap \
  --features-dir outputs/aliked \
  --matches-dir outputs/roma_matches \
  --output-dir outputs/colmap_fixed \
  --camera-model PINHOLE \
  --camera-params 1350.287273,1350.713931,962.552727,716.328089 \
  --dry-run

# 正式导出；output-dir 必须是尚不存在的新目录
python -m fixed_sfm.export_colmap \
  --features-dir outputs/aliked \
  --matches-dir outputs/roma_matches \
  --output-dir outputs/colmap_fixed \
  --camera-model PINHOLE \
  --camera-params 1350.287273,1350.713931,962.552727,716.328089

# COLMAP 4.0.4 对数据库中的原始对应重新做标定两视图验证
colmap geometric_verifier \
  --database_path outputs/colmap_fixed/database.db \
  --TwoViewGeometry.max_error 2 \
  --TwoViewGeometry.min_num_inliers 15 \
  --TwoViewGeometry.compute_relative_pose 1 \
  --num_threads 4

# 几何验证完成后建图
mkdir -p outputs/colmap_fixed/sparse
colmap mapper \
  --database_path outputs/colmap_fixed/database.db \
  --image_path /home/nvidia/colmap_data/DjiData_1367/images \
  --output_path outputs/colmap_fixed/sparse \
  --Mapper.num_threads 4
```

如果 `outputs/colmap_fixed` 已经导出，直接从几何验证继续。重新导出请选择新目录，例如 `outputs/colmap_fixed_v2`。程序不覆盖已有数据库，避免丢失几何验证或 mapper 状态。

导出时 `two_view_geometries` 为空，必须运行 `geometric_verifier` 再运行 mapper。RoMa 的 F 掩码仅用来初筛，不能代替 COLMAP 的 E/F/H 分类与相对位姿估计。无需再运行 `feature_extractor` 或任何描述子 matcher。上述选项依据本机 COLMAP 4.0.4 的 `--help` 和源码；其他版本先检查此子命令是否可用。

若内参已经可靠标定，需要固定焦距和畸变，可给 mapper 增加 `--Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_params 0`。默认遵循 COLMAP 的优化设置。没有满足几何条件的图像可能无法注册，导出不承诺全部图像建图成功。

## 主要参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--features-dir` | 必填 | 固定 ALIKED 库，含配置、索引和 NPZ |
| `--matches-dir` | 必填 | 已完成的 RoMa 输出 |
| `--output-dir` | 必填 | 新的导出目录，内含 `database.db` |
| `--images-dir` | 特征配置中的目录 | 源图路径；搬移后可指定，内容必须一致 |
| `--camera-model` | 无 | 全部图像共享的 COLMAP 相机模型 |
| `--camera-params` | 无 | 该模型的原图内参，逗号分隔 |
| `--camera-config` | 无 | 多相机 JSON，与共享相机参数互斥 |
| `--match-selection` | `inliers` | `inliers` 初始 F 内点；`all` 全部双向互检对应 |
| `--min-matches` | `15` | 选择后不足该数量的配对跳过，至少为 8 |
| `--pair-scope` | `selected` | 最近一次选定配对；`completed` 为累计完成索引 |
| `--dry-run` | 关闭 | 相同输入校验和统计，不建输出目录或数据库 |

若希望由 COLMAP 完整重估全部互检对应，使用 `--match-selection all`，选择新的输出目录。两种选择都保留原始关键点 ID，都必须运行原生几何验证。

默认要求 `pairs_used.json` 中每一对都有完成索引和有效 NPZ；缺失或损坏即报错，不静默生成残缺数据库。`--pair-scope completed` 显式导出累计索引，可能包含以前配对模式留下的结果。

## 多相机配置

共享相机时使用上述 CLI。尺寸不同、相机不同时改用 `--camera-config cameras.json`，示例：

```json
{
  "cameras": [
    {"id": 1, "model": "PINHOLE", "width": 1920, "height": 1439,
     "params": [1350.287273, 1350.713931, 962.552727, 716.328089]},
    {"id": 2, "model": "SIMPLE_PINHOLE", "width": 1280, "height": 960,
     "params": [900, 640, 480]}
  ],
  "default_camera_id": 1,
  "image_cameras": {"second_camera/image.jpg": 2}
}
```

参数仅示范格式。每幅图必须获得明确的相机分配；尺寸必须与该图固定关键点库一致。程序检查模型参数数量、数值、焦距及相机 ID，不允许未知图像、重复或未使用相机。

## 输出与不变约束

```text
colmap_fixed/
  database.db           # 原生 pycolmap 创建的数据库
  export_report.json    # 导出时计数、相机参数、输入身份、未关联图像
  image_ids.json        # 图像名 -> image_id/camera_id/feature_id/关键点数量
  pair_statistics.csv   # 每对互检数、内点数、实际导出数、状态、内容身份
  pairs.txt             # 实际写入数据库的配对
```

- 图像按相对名称字典序分配从 1 开始的 `image_id`。关键点 ID 仍为各图 NPZ 从 0 开始的原始行号，不因过滤或图像 ID 改变而重排。
- 全部已索引图像及全部固定关键点均写入，包括空关键点图像和没有合格配对的图像；未匹配关键点不会删除。
- NPZ 已是原图 COLMAP 坐标，直接写入，不再加 `0.5`，不缩放、不合并、不用 RoMa 预测位置覆盖。
- 校验配置与 NPZ 内容哈希、匹配两端特征身份、索引计数、关键点数量及匹配 ID。逐张校验源图像 SHA256 和存储方向下的尺寸。
- 使用原生 pycolmap 数据库 API，创建每相机一个独立 rig、每图一个 frame，适配本机 COLMAP 4.x。这里不表示设备具有已知的多相机同步 rig。
- 不写描述子，不将 ALIKED 描述子伪装为 SIFT。原生几何验证和 mapper 使用已保存的关键点与对应。
- 临时目录内建库，通过 SQLite 完整性、外键与计数检查后发布。失败不留下正式数据库；输入特征与匹配保持不变。
- `export_report.json` 记录导出当时的状态。后续几何验证写入数据库后，报告的 `two_view_geometries=0` 仍是导出时计数。

原生集成测试覆盖非连续匹配 ID、坐标不重复转换、空关键点、多相机、输入损坏、未完成配对和已有输出，还验证无描述子数据库经过原生几何验证后可被 mapper 的 `DatabaseCache` 加载。

```bash
python -m unittest discover -s fixed_sfm/tests -v
```
