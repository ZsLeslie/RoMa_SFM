# ALIKED 网格提取 V2：整图特征图与空间配额

流程：**整图计算一次 ALIKED 特征图/分数图 → 全图 NMS 与亚像素候选 → 原图掩膜筛选 → 每格配额 → 固定 ID**。

V2 不再对 RGB 图像逐格裁块推理，保留整图网络上下文、填充与特征图对齐。`--grid-size 0` 保留原整图模式；大于 0 时启用 V2。

## 运行命令

使用已有环境和离线权重，选择新的特征目录，先提取少量图像：

```bash
cd /home/nvidia/src/roma
source /home/nvidia/venvs/2dgs_env/bin/activate

python -m fixed_sfm.extract \
  --images-dir /home/nvidia/colmap_data/DjiData/images \
  --masks-dir /home/nvidia/colmap_data/DjiData/masks \
  --output-dir outputs/DjiData_aliked_grid_v2 \
  --grid-size 256 \
  --max-keypoints 4096 --detection-threshold 0.05 \
  --resize-max 1280 --device cuda \
  --max-images 3
```

去掉 `--max-images 3` 即处理全部图像，同一配置下已完成的文件会校验并复用。也支持从 `fixed_sfm/` 目录运行 `python extract.py`，相对路径以当前工作目录为准。

无掩膜的叶片腔体数据示例：

```bash
python -m fixed_sfm.extract \
  --images-dir /home/nvidia/colmap_data/456_叶片A_主梁腔体_202607281631/images_undistorted \
  --output-dir outputs/task_456_aliked_grid_v2 \
  --grid-size 320 \
  --max-keypoints 4096 --detection-threshold 0.2 \
  --resize-max 1280 --device cuda \
  --max-images 3
```

阈值沿用各自数据的已有配置，仅作为起点。`--dry-run` 只列输入，不加载模型、建输出或执行逐图预算检查。

## 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--grid-size` | 0 | 原图像素中的网格边长；0 为原整图模式，启用时至少 32 |
| `--grid-margin` | 64 | 仅兼容 V1 命令；V2 忽略，不参与缓存身份，日志会说明 |
| `--max-keypoints` | 4096 | 整图最终上限，分配给有效格，不是全图候选截断数量 |
| `--resize-max` | 1024 | 整图推理长边上限；小图不放大；0 为原尺寸 |
| `--nms-radius` | 2 | 推理栅格中的全图 NMS 半径及最终跨格近邻半径 |
| `--detection-threshold` | 0.2 | 最终质量门槛，不自动降低 |

## 选择规则

1. 图像解码、统一缩放后，对整图执行一次 `extract_dense_map`。RGB 输入不涂黑，不逐格重新计算 CNN。
2. 使用本机 ALIKED 的 DKD 完成全图 NMS、亚像素定位和分数采样。其 `n_limit` 临时设为分数图像素数，调用后恢复。因此掩膜和配额前没有全图 Top-K 截断，原有 20000 候选限制不会让密集区域先占完名额。
3. 候选恢复至原图 COLMAP 坐标并确定唯一网格归属。过滤无效坐标、未过阈值及掩膜外候选，再执行配额和最终跨格近邻处理。
4. 无掩膜时所有格有效；有掩膜时，格内至少一个像素通过即为有效。右侧和底部不完整格与其他有效格配额相同。
5. 有效格数为 M、总预算为 N，各格先分 `N//M`，余数按 y/x 顺序分配。有效格数大于预算时报错；全部无效时保存空库并跳过 CNN。
6. 候选按分数降序、y/x 升序处理，达到配额后停止接收该格的点。被跳过的候选不会抑制邻格。空格/弱纹理格的余额不转给其他格。
7. 开启描述子保存时，复用同一次整图特征图，只描述最终入选点；无效描述子对应点会删除，不补足。关闭时不运行描述子头。
8. 入选集合不超过整图预算，交给原写入器排序、分配固定 ID。NPZ 仍为每图一份，格式不变。

例如 1920×1080、网格 256 时共 40 格。若有 19 个有效格、预算 4096，则前 11 格各 216，其余有效格各 215。`active=19` 不代表 19 格都存在合格角点；实际保存至少一个点的格才计入 `occupied`。

## 坐标、缓存与后续匹配

使用统一缩放的真实横纵比例和半像素映射恢复原图坐标，NPZ 左上像素中心仍为 `(0.5,0.5)`。不会重复增加偏移，不改变原图尺寸或相机内参。

新版本标识为 `full_image_aliked_masked_grid_v2`。版本、网格大小、预算规则、候选策略、描述子策略、适配器与网格实现 SHA256 进入配置身份。V1 目录不能作为 V2 输出目录，必须选择新的特征目录，并在新的匹配目录重新运行 RoMa。原来的特征与匹配保持不变。

`--grid-margin` 不再影响 V2 结果或身份，旧命令带该参数也可运行。`--grid-size 0` 保留原来的提取行为与配置格式。

## 日志与每格统计

日志会说明 `Grid V2: one full-image dense forward`，逐图输出 `cells/active/occupied` 和点数。`index.json` 各图的 `grid` 字段包含：

- `cells`、`active_cells`、`occupied_cells`：全部、掩膜有效、最终有点的格数。
- `quotas`：y/x 顺序的每格配额，无效格为 0。
- `selected_per_cell`：各格最终保存数量，均不超过对应配额。

续跑或索引恢复会根据已存关键点与掩膜重建统计。固定 ID 仍为 NPZ 行号，RoMa 和 COLMAP 导出接口可沿用。

V2 修复 V1 独立裁块可能导致的检测响应下降，仍受原图纹理、推理尺度和质量门槛限制，不会强行填满所有格，也不直接保证重建连通性。整图 CNN 显存需求高于单块；默认不保存描述子时提前释放不再需要的特征图。
