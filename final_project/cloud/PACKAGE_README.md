# M4 云端训练包（2026-10-26，git `cca94d2`）

只打包，未上传。生成脚本：`final_project/cloud/pack.py`（仅白名单文件）。

```
cloud_pkg/
├── README.md               本文件
├── exclusion_check.md      封存测试区排除检查（PASS）
├── data_extra/             补算的逐单元升高分位数（67 MB，见其中 README.md）
├── data/                   大，只传一次（共 32.9 GB）
│   ├── shard_000 … shard_015.tar.zst   16 个分片：zstd level 3，每片约 2 GB，合计 31.7 GB（原始 51.2 GB）
│   ├── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth   DINOv3 权重，1.21 GB，未压缩，Meta 许可：只放私有存储
│   ├── MANIFEST.csv        每个文件属于哪个分片、路径、字节数、sha256（47,169 个文件 + 权重）
│   ├── SHARDS.csv          每个分片的 sha256 和字节数
│   └── SHA256SUMS          分片 + 权重的校验和（sha256sum -c 格式）
└── code/                   小，改代码后可单独重传（249 KB）
    ├── code.tar.zst        git archive HEAD 中的代码：src/、scripts/、cloud/、pyproject.toml、uv.lock、
    │                       .python-version、README.md、M4_plan.md 等文档、queue_m4.sh（清单见 FILES.txt）
    ├── FILES.txt           归档里的文件列表和 git HEAD
    ├── setup_cloud.sh      云端一键准备（见下）
    ├── sync_back.txt       训练完要传回本地的文件
    ├── M4_plan.md          含修订 1 的 M4 计划
    ├── equivalence_check.md 缓存模式等价性检查（PASS）
    └── SHA256SUMS          code/ 里各文件的校验和
```

## data/ 里有什么（解包后路径都相对于 `final_project/`）

| 内容 | 路径 |
|---|---|
| 训练和验证图块（训练直接读取的 `.npy`：图像、nDSM、植被；稀疏年份另有补洞版 nDSM），8,988 个图块，37 个邮编，4 个年份 | `data/interim/height_npy_city/{ZIP}_{年}/` |
| 每组图块的索引 | `data/interim/height_city/{ZIP}_{年}/index.csv` |
| 6 个时段的真变化掩膜，共 12,736 对 | `data/interim/bitemporal_masks_city/` |
| 单元编号图块，以及每个邮编的单元面积统计 | `data/interim/unit_ids_city/`、`unit_ids_city/sizes/` |
| 逐单元 LiDAR 标签（6 个训练时段） | `data/interim/lidar_change_city/{ZIP}_{y0}_{y1}.parquet` |
| 逐单元变化统计：升高比例、降低比例、升高中位数、两期高度的中位数和 p90、覆盖率、面积等，列名后缀 `_y0` / `_y1` 表示时段起止年 | `data/interim/m4_unit_change_stats.parquet` |
| 图块索引表：编号、时段、范围（EPSG:2272 英尺）、邮编、1 km 方块、训练或验证、真变化像元数 | `data/interim/m4_pair_index.csv` |
| 图块选择（只含训练和验证） | `data/interim/m4_patches.csv` |
| 训练和验证区的 1 km 方块划分 | `data/splits/splits_citywide_trainval_blocks.csv` |

**升高幅度的 p75、p90 和 1.5 m / 2.5 m 楼面占比在 `data_extra/` 里**（2026-10-26 补算，2,747,718 行，训练和验证区 6 个时段，sha256 和列说明见 `data_extra/README.md`）。云上把 `m4_unit_dh_quantiles.parquet` 复制到 `final_project/data/interim/`。

**不包含**：8 个封存测试邮编的任何文件、全市栅格、原始 LAS、航拍缓存、单元边界（含地址）、检查页和标注结果。

## 在云上怎么用

需要的工具：`zstd`、`tar`、`sha256sum`、`curl`。`uv` 如果没有，脚本会自动安装。

```bash
# 1. 把 data/ 和 code/ 传到云主机，例如 /data/cloud_pkg/
# 2. 一键准备：校验 → 解包 → 安装依赖 → 打印环境 → 100 步冒烟测试
bash /data/cloud_pkg/code/setup_cloud.sh /data/cloud_pkg /workspace/m4
# 3. 训练（修订 1 的规则：8,000 步，每次验证检查长高 AP ≥ 0.15，备用规则写在脚本里）
cd /workspace/m4/final_project
POPUP_PREP_CACHE=1 nohup bash results/citywide/m4/queue_m4.sh > results/citywide/m4/queue_cloud.log 2>&1 &
```

- `POPUP_DATA=citywide` 已写在排队脚本里。`POPUP_PREP_CACHE=1` 让训练直接读取打包好的掩膜、单元编号和单元面积，不需要全市栅格。
- 依赖按 `uv.lock` 原样安装：Python 3.13.2，torch 2.11.0+cu128，来自 PyTorch cu128 源。锁文件里已有 Linux x86_64 的轮子，不需要改。云端驱动需要支持 CUDA 12.8。
- 建议配置：内存 ≥ 64 GB，NVMe 本地盘 ≥ 150 GB（解包后约 55 GB，加上检查点）。数据不要放在网络挂载盘上。
- 云上训练会从头开始：代码包里不含本地的训练日志，所以排队脚本不会跳过已经在本地跑完的种子。在哪台机器、用哪个种子训练，需要你决定，报告里要注明。
- 训练完按 `sync_back.txt` 把文件传回本地。
