# ALIKED 固定关键点与 RoMa SfM

项目目录：`/home/nvidia/src/roma`。后续代码开发、测试和项目内输出均在此目录进行。

当前源码包名为 `aliked_roma`。已接入 DeDoDe v2-L 检测器和固定 ID 的 RoMa 匹配；离线权重、提取/匹配命令及 12 张样本的整图匹配对照见 [DEDODE.md](aliked_roma/DEDODE.md)。当前对照不使用人工区域标注。

当前已实现 ALIKED 固定关键点提取、基于固定 ID 的双向 RoMa 匹配、循环一致性过滤与初始 F 几何验证，以及固定关键点和匹配结果到原生 COLMAP 数据库的导出。原有 `/home/nvidia/src/roma_sfm` 保留为旧程序。

```bash
cd /home/nvidia/src/roma
/home/nvidia/venvs/2dgs_env/bin/python -m aliked_roma.extract --help
/home/nvidia/venvs/2dgs_env/bin/python -m unittest discover -s validation -v
```

也支持进入 `aliked_roma/` 后直接运行 `python extract.py --help`；相对输出路径以当前工作目录为准。

详细运行命令、数据格式与坐标约定见 [aliked_roma/README.md](aliked_roma/README.md)。

此前的实图验证数据已迁移至 `outputs/aliked_smoke_cpu`、`outputs/aliked_default_cpu`，坐标验证结果位于 `outputs/aliked_verification.json`。

RoMa 匹配入口：`python -m aliked_roma.match`，或进入 `aliked_roma/` 后运行 `python match.py`。命令和格式见 [MATCHING.md](aliked_roma/MATCHING.md)。

COLMAP 导出入口：`python -m aliked_roma.export_colmap`。导出、几何验证、mapper 的完整命令见 [COLMAP.md](aliked_roma/COLMAP.md)。

ALIKED 网格均匀提取 V2：增加 `--grid-size 256`，整图计算一次特征图/分数图后，在掩膜内按网格配额选点。说明见 [GRID.md](aliked_roma/GRID.md)。`--grid-size 0` 保留原整图模式；`--grid-margin` 仅兼容旧命令，V2 已忽略。使用新的特征与匹配目录。
