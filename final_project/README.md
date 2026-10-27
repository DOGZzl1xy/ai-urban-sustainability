# Final Project

MUSA 6950 — AI for Urban Sustainability · Final Project

**Teaching Aerial Imagery to See Height: Tracking Philadelphia's Gentle Densification with LiDAR-Supervised Foundation Models**

- [ASSIGNMENT.md](ASSIGNMENT.md) — 课程作业要求和交付物清单
- `proposal/` — 2 页 proposal 和事实核查记录（`verify_log.md`）
- `src/`, `scripts/` — 代码；`docker/` — GPU 镜像（`popup-vision`：YOLO 车辆检测）
- `results/popup_feasibility/` — 第一阶段：LiDAR 变化检测的可行性检验（19146），见 [summary.md](results/popup_feasibility/summary.md)
- `results/height_model_pilot/` — 第二阶段：从航拍估高度的快速检验
- `results/bitemporal/` — 第三阶段：双时相模型 M1–M3b 和第一轮正式测试，见 [summary.md](results/bitemporal/summary.md)
- `results/citywide/` — 全市阶段，索引见 [results/citywide/README.md](results/citywide/README.md)：范围和划分、M4（最终 `m4s_s0`）、第二轮测试、第 6 阶段全区估计（[stage6_plan.md](results/citywide/stage6_plan.md)）、第 7 阶段 RQ2 和 RQ3（**[stage7/summary.md](results/citywide/stage7/summary.md)**）
- `cloud/` — 云上训练和推断：打包脚本、`setup_cloud.sh`、服务器上用过的辅助脚本（`server_scripts/`）

## 进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| 0 | 新分支 `popup-densification`；删除前两个想法的代码（git 历史里还有，数据移到 `data/_old_ideas/`）；提交可行性检验 | 完成 |
| 1 | 定稿 proposal（核对数字、核实事实、图 1、2 页 PDF）；发现 2025 年 LiDAR，改为"三期 LiDAR + 模型补年份" | 完成，见 `proposal/verify_log.md` |
| 2 | 从 v3 标记里新抽 50 个，验证精确率 | 完成：95.5% [88.6, 100] |
| 3 | 高度模型快速检验（2018 训练，2022 和 2025 测试） | 完成：DINOv3 + DPT 建筑级 MAE 1.17–1.60 m，新建 92% / 91%，加层 44% / 44% → 下一步做双时相模型，见 `results/height_model_pilot/summary.md` |
| 4 | 第三阶段：双时相模型 M1/M2 → 单元级分类 M3 / M3b | 完成：最终模型 M3b 种子 1。按事先定好的标准，加层没有达标（两个种子平均 AP 0.31、网格相关 0.79），作为局限写进报告。2022→2025 只测了一次：长高 AP 0.56、加层 AP 0.16；在 250 m 网格和 block group 上，长高和新建的相关为 0.83–0.88。见 `results/bitemporal/summary.md` |
| 5 | 新阶段：全市。研究范围掩膜、新的封存测试区（8 个邮编）、M4 配方 | 完成并冻结（2026-10-23）：`results/citywide/study_area.md`、`splits_citywide.md`、`M4_plan.md` |
| 6 | 全市 LiDAR 和航拍处理（先试跑 20 个瓦片） | 完成：LiDAR 2018/2022/2025 覆盖全研究区，2010/2015 覆盖训练+验证区，共 2,661 块，0 失败；航拍 9 个年份，0 缺失；训练+验证区 38 个 ZIP 逐地块 LiDAR 变化（7 个时段）。2018 年点密度在市中心以外只有约 15–20 点/m²，含 2018 的时段覆盖的地块较少（约 65%），规则不改 |
| 7 | 2025 植被修正：4/18 飞行时树已长叶，树冠被当成建筑；改为同时用点分类（3/4/5）去植被 | 完成（2026-10-26 06:52）：2025 重下 548 块，0 失败；2022→2025 新建 1,502→1,346、加层 807→762（拆除 626→713，需注意）；`change_summary_trainval.csv` 已重算。M4 训练不受影响；第一轮测试不重跑，加注说明。见 `results/citywide/veg2025_fix.md` |
| 8 | 25 m 高层标记（100 m、250 m 网格和 block group） | 训练+验证区完成：标记 100 m 格 731 个、250 m 格 159 个、block group 35 个（`highrise_trainval.csv`）；测试区在第二轮测试时补 |
| 9 | LiDAR 规则抽查（新区域，每类 20 个） | 样本已抽（2026-10-26，`spotcheck_items.csv`）：联排 20、独立住宅 10（全区只有 10 个检测，全部抽取）、混合 20，共 50 个；页面 `results/citywide/review_spotcheck.html`（只留本地）**等用户标注**，导出 `review_spotcheck_labels.csv` 后运行 `python scripts/spotcheck.py score` |
| 10 | M4：切全市训练图块、两个种子训练、按验证集选定 | 进行中。数据准备完成（8,988 块，10,559 训练对 / 329 验证对）；抽查通过（95.7%）。按公式放大到 41,000 步的种子 0（`m4_s0`）在第 500–2,500 步间崩溃（学习率曲线被拉长），已暂停并保留、诊断。**修订 1**（2026-10-26，用户批准）：改用 M3b 原节奏 8,000 步，每次验证都检查 grew AP ≥ 0.15，备用规则事先写定；新运行 `m4b_s*`。**完成并冻结（2026-10-27 00:53）**：6 个种子中 s1–s4 在前 1,500 步被保护规则停掉，s0 和 s5 跑完；按验证集加层 AP 选定 **`m4b_s0`**（0.123 vs 0.116，长高 AP 0.720）。见 `results/citywide/m4/m4_selection.md`。**修订 2**（2026-10-27，用户批准，在任何测试之前）：配置 S（正样本权重 18.45、batch 16、EMA、均值+最大值池化），云 GPU 训练，两个种子都没被停掉；按事先定的 block bootstrap 规则采用，**最终 M4 = `m4s_s0`**（验证集加层 AP 0.426、长高 AP 0.799）。见 `m4_selection.md` |
| 11 | 第二轮正式测试（只跑一次）：M4 和 M3b 种子 1，2022→2025，8 个封存邮编 | 完成（2026-10-27，本地，锁文件 `round2_test_log.csv`）：M4 长高 AP 0.683、加层 AP 0.150；250 m 网格 r：新建 0.96、长高 0.91、加层 0.42 → 推断模型 `m4s_s0`，**加层没过 0.8 门槛，中间年份不估计加层**。加层精确率下降主要因为测试期加层发生率只有验证期的约 1/5。见 `results/citywide/m4/round2_test.json` |
| 12 | 中间年份推断（第 6 阶段）和 RQ2 / RQ3（第 7 阶段） | 完成（2026-10-28）：云上推断 6,235 个图块；中间年份只报新建（长高超过自己的 LiDAR 终点，用户批准）。同一批单元上 2018–22 → 2022–25：新建 1.62 → 1.20、加层 1.06 → 0.71（每千单元每年）；前 10% 的 250 m 格占 94% 新建；离轨道站点 1.6 km 以上几乎没有；许可 2018–20 签发的新建 61% 到 2025 年可见；没有对应许可的变化 3.6–9.1%。见 `results/citywide/stage6_plan.md`、`stage7/summary.md` |
| 13 | 交互式地图（只用汇总格子，不标地址） | 待做 |
| 14 | 最终报告 | 待做 |

## 数据

`data/` 不提交 git。`data/raw/` 是原始数据，不修改：2016 和 2025 地块、2017 建筑轮廓、边界、L&I 许可、航拍瓦片缓存（`data/raw/ortho/tiles/`，9 个年份），以及流式处理前就在本地的 LiDAR 原始文件（`keep_raw_files.txt`）。全市 LiDAR 是流式处理的：下载 → 切到全市 0.5 m 网格（`data/interim/citywide_tiles/`）→ 删除原始文件。按 ZIP 拼接的栅格在 `data/interim/citywide_rasters/`，逐地块变化标签在 `data/interim/lidar_change_city/`。
第 7 阶段新增的公开数据：`data/raw/li/permits_studyarea_2014_2025.csv`（Carto 许可）、`data/raw/acs/`（ACS 汇总文件，不需要 API 密钥）、`data/raw/boundaries/tl_2020_42_bg.zip`、`data/raw/transit/gtfs_public.zip`（SEPTA）；来源写在各目录的 `SOURCE*.txt` 里。第 6 阶段的逐单元结果在 `data/interim/stage6_units.parquet`、`data/interim/stage6_scores/`。

**云服务器**（AutoDL，RTX 4080 SUPER 32 GB）一直开着：数据在 `/root/autodl-tmp/`，最终检查点备份在 `/root/autodl-fs/m4_final/`（含 SHA256SUMS）。本地已有 `unitcls_m4s_s0.pt` 和 `_grew.pt`；`m4s_s1` 只在云上。

带具体地址的文件（`review*.html`）只留在本地。
