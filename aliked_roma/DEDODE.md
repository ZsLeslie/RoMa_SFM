# DeDoDe v2-L 固定点提取与整图匹配对照

当前源码包名是 `aliked_roma`。新增检测器使用已有 Kornia 0.8.2 的 DeDoDe 实现，严格加载官方 `dedode_detector_L_v2.pth`。不构建描述子、不下载权重、不替换 Jetson PyTorch。

```bash
cd /home/nvidia/src/roma
/home/nvidia/venvs/2dgs_env/bin/python -m aliked_roma.extract \
  --extractor dedode \
  --dedode-weights models/dedode_detector_L_v2.pth \
  --images-dir /home/nvidia/colmap_data/DjiData/images \
  --masks-dir /home/nvidia/colmap_data/DjiData/masks \
  --image-list outputs/roma_detector_comparison_20261117/image_list.txt \
  --output-dir outputs/dedode_blade_topk_20261117 \
  --max-keypoints 4096 --resize-max 0 --grid-size 0 --device cuda
```

同一命令使用 `--grid-size 256` 和新的输出目录可提取网格版。本次网格版位于 `outputs/dedode_blade_grid256_20261117`。也支持进入 `aliked_roma/` 后运行 `python extract.py`；相对路径仍相对当前工作目录。

`--extractor aliked` 为默认行为，原有 ALIKED 参数默认值保持不变。DeDoDe 默认模型 `dedode-L-v2`，检测阈值为 0，采用覆盖度校正后的原始分数排序，不套用 ALIKED 的 0.05/0.2 阈值。不额外执行推理 NMS，`--nms-radius` 只接受 0；第一版不支持 `--save-descriptors`。

整图 RGB 只前向一次，使用 ImageNet 归一化。`--resize-max 0` 使用原图；其他值只在必要时等比例缩小长边，记录实际取整后的 x/y 尺度。掩膜在原图坐标中检查，并在最终 top-K 前施加。网格版按原图 keep-mask 非空格均分点预算，余数按 y/x 次序分配，不跨格补配额。网格版与原生版均可能选入弱点，检测数量不代表可靠匹配。

输出仍使用原有固定点 NPZ 格式。适配器将 DeDoDe 像素中心转换为公共写入器要求的零中心坐标，写入器只加一次 0.5；不改变原图尺寸和内参约定。最终过滤和排序后才分配行 ID。配置身份包含权重、适配器、Kornia 源码、公共写入器、选点策略及运行环境；更改这些内容需要新输出目录。

当前只比较同一组 12 张图、12 个图像对的整图匹配信息，不使用此前的人工叶片区域。对比指标为初始 F 内点、唯一匹配观测、多邻图复用、三图闭环，以及整图均匀格覆盖率。

```bash
# 使用已完成的匹配缓存比较，不重新推理，不读取人工标注。
/home/nvidia/venvs/2dgs_env/bin/python -m aliked_roma.compare_matches \
  --baseline-features outputs/DjiData_aliked \
  --baseline-matches outputs/DjiData_roma \
  --candidate 'DeDoDe native+RoMa' outputs/dedode_blade_topk_20261117 outputs/dedode_blade_topk_roma_20261117 \
  --candidate 'DeDoDe grid256+RoMa' outputs/dedode_blade_grid256_20261117 outputs/dedode_blade_grid256_roma_20261117 \
  --pairs-file outputs/roma_detector_comparison_20261117/pairs.txt \
  --output-dir outputs/roma_detector_comparison_20261117

/home/nvidia/venvs/2dgs_env/bin/python -m unittest discover -s validation -v
```

对照程序检查源图、掩膜、特征及匹配内容身份，确认除点库身份以外的全部匹配配置一致。均匀格覆盖率在整张图上统计，分别报告 32、64、128 像素格；新增和丢失覆盖也同时保留。

不同检测器的行 ID 不直接比较。坐标互补性以同一条参考对应在两端原图中均位于给定像素半径内为准，报告 1、3、5 像素三档。这只反映像素位置的差异，不能证明新增的独立物理点或三维信息。

RoMa 匹配已接入同一固定点接口。使用如下命令运行或复用本次原生版的 12 对结果：

```bash
/home/nvidia/venvs/2dgs_env/bin/python -m aliked_roma.match \
  --features-dir outputs/dedode_blade_topk_20261117 \
  --pairs-file outputs/roma_detector_comparison_20261117/pairs.txt \
  --output-dir outputs/dedode_blade_topk_roma_20261117 \
  --weights-dir /home/nvidia/models/RoMa \
  --roma-repo-dir /home/nvidia/src/RoMa --device cuda
```

网格版对应 `--features-dir outputs/dedode_blade_grid256_20261117`、`--output-dir outputs/dedode_blade_grid256_roma_20261117`，其他参数相同。默认 RoMa 分辨率为 448/672，原图吸附半径与循环误差上限均为 3 px，最近邻比值为 0.8，最低 RoMa certainty 为 0.2，F 阈值为 2 px。检测器的 0 阈值不进入匹配置信度筛选。

12 张图来自四个局部拍摄段，所以本次 `pairs.txt` 只包含每组三张图内部三对，共 12 对。全量连续数据仍可使用既有的 linear/exponential/hybrid 配对方式。

两版共完成 24 对 GPU 匹配，保留原始坐标及行 ID，绑定两端特征内容身份，未重提取特征。匹配结果是初始 F 内点，尚未执行 COLMAP 标定几何验证或三角化。

当前对照报告位于 `outputs/roma_detector_comparison_20261117/report.md`。旧复查入口也已转向整图匹配比较，不再读取或绘制人工区域：

```bash
/home/nvidia/venvs/2dgs_env/bin/python outputs/dedode_blade_matching_20261117/analyze.py
```

参考：[DeDoDe 官方仓库](https://github.com/Parskatt/DeDoDe)、[DeDoDe v2](https://arxiv.org/abs/2404.08928)。
