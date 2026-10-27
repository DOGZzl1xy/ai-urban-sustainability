# 全市阶段（results/citywide）索引

按时间顺序。**加粗**的是冻结的规则文件：改动只能以"修订"的形式追加在文末，并在执行前提交。

## 1. 范围、划分和数据质量
| 文件 | 内容 |
|---|---|
| **[study_area.md](study_area.md)** | 研究范围（152.5 km² 的住宅、中低密度混合用途分区）、25 m 高层标记规则 |
| **[splits_citywide.md](splits_citywide.md)** | 训练、验证、封存测试的 ZIP 和 1 km 方块划分（`splits_citywide*.csv`） |
| [veg2025_fix.md](veg2025_fix.md) | 2025 年树冠误判为建筑的修正 |
| [spotcheck_plan.md](spotcheck_plan.md) | LiDAR 规则抽查（精确率 95.7%），`spotcheck_*`、`review_spotcheck_labels.csv` |
| `change_summary_trainval.csv`、`zip_area_types.csv`、`study_area_stats.csv` | 训练和验证区的变化汇总、ZIP 类型、范围统计 |
| `highrise_trainval.csv`、`highrise_all.csv` | 高层标记（训练加验证区；全部 46 个 ZIP，第二轮测试时合并重算） |

## 2. 模型 M4（m4/）
| 文件 | 内容 |
|---|---|
| **[M4_plan.md](M4_plan.md)** | 配方、修订 1、修订 2、第二轮测试的执行记录 |
| [m4/m4_selection.md](m4/m4_selection.md) | 选种子的记录：修订 1 选 `m4b_s0`；修订 2 改选 **`m4s_s0`**（bootstrap 判定见 `m4_compare.json`） |
| `m4/unitcls_<tag>_{train,steps}.csv`、`_val*.json`、`DEAD_*` | 每个种子的训练和验证日志（`m4_s0`、`m4b_s0–s5`、`m4s_s0–s1`） |
| `m4/queue_*.sh`、`queue_*.log` | 训练排队脚本（本地 `queue_a1`、云上 `queue_m4s`） |
| `m4/cloud_readiness.md`、`m4/equivalence_check.md` | 上云前的检查 |
| `m4/round2_test.json`、`round2_grid.csv`、`round2_test_log.csv` | **第二轮正式测试**（只运行一次；锁文件存在就拒绝再跑） |

## 3. 全研究区估计（stage6/）
| 文件 | 内容 |
|---|---|
| **[stage6_plan.md](stage6_plan.md)** | 方法，以及"中间年份只报新建"的决定 |
| `stage6/cells_{100m,250m,bg}.csv` | 每个格子或 block group：LiDAR 两个时段的计数，模型对 2019、2020、2023、2024 年的估计，两条锚点序列（A、B），覆盖率和高层标记 |
| `stage6/totals.csv`、`consistency.csv`、`meta.json` | 全区合计、一致性检查、阈值和覆盖率 |

## 4. 城市分析（stage7/）
| 文件 | 内容 |
|---|---|
| **[stage7_plan.md](stage7_plan.md)** | RQ2 和 RQ3 的方法 |
| [stage7/summary.md](stage7/summary.md) | **主要结果** |
| `stage7/rq2_*` | 发生率（全区、同一批单元、区域类型、ZIP、轨道距离、收入）、逐年新建、250 m 格子、负二项回归 |
| `stage7/rq3_*` | 许可到建设的比例、检测时间、没有对应许可的比例（全区和按 ZIP）、建筑面积和住房 |
| `stage7/figs/` | 地图、逐年和收入图、检测时间图 |

## 5. 过程记录
| 位置 | 内容 |
|---|---|
| `processing/` | LiDAR 下载处理记录 `stream_<年>.csv`、航拍瓦片缓存记录 `imagery_<年>.csv` |
| `logs/` | 本地长任务的日志 |
| `logs/cloud/` | 云服务器日志：环境、上传校验、解压、测速、冒烟测试、第 6 阶段推断 |
| `../../cloud/` | 打包脚本（`pack*.py`）、`setup_cloud.sh`、`server_scripts/`（服务器上用过的辅助脚本） |

带地址的 `review*.html` 只留在本地（`.gitignore`）。所有提交的表格只有内部单元编号或汇总后的格子，没有地址和地块编号。
