# 姿态—中心候选锁定与普通录制曝光审计

**当前可取得的 3EED 普通录制中，符合独立确认条件的录制为 0 段。** 审计覆盖官方发布包中的 199 段普通录制，全部可追溯到此前的项目评估或开发。三组配置已按原代码锁定，独立验证等新录制。现有主方法采用 `C_E_Qsize`，S／GS／GD 归档。[逐平台审计](results/ordinary_recordings.csv) · [证据回执](results/exposure_receipt.json) · [配置锁](CANDIDATE_LOCK.json)

## 录制核对

按平台与原始录制名取交集。普通 Validation 使用已评分的 18,620 条方向记录对应的封存清单。普通 TRAIN 使用 R159 的 3,072 条方向记录及源域模块的拟合和标定清单。对源域模块另核对实际生成的自然样本特征数量及清单哈希。初始 Qwen 96 帧和后续 416 帧的两个封存清单也纳入审计。对已曝光录制中的新帧，仍按同一录制处理。[审计实现](audit_recordings.py)

| 平台 | 普通 TRAIN 可取得／已曝光 | 普通 Validation 可取得／已曝光 | 可独立确认 |
|---|---:|---:|---:|
| Waymo | 94／94 | 93／93 | **0** |
| Drone | 3／3 | 2／2 | **0** |
| Quad | 4／4 | 3／3 | **0** |
| 合计 | **101／101** | **98／98** | **0** |

Waymo 的 94 段普通 TRAIN 已由 R159 覆盖。Drone 的 3 段中 2 段进入过 R159，剩下 1 段做过源域拟合。Quad 的 4 段中 2 段进入过 R159，剩下 2 段用作源域标定。前一次报告中的 224 帧只在最初 96 帧的选型录制之外，此后多轮已用来开发和评估。[录制用途计数](results/ordinary_recordings.csv)

官方 Waymo split 另外列出 6 段 TRAIN 和 4 段 Validation。直接核对发布包 ZIP 成员名后，这 10 段在包中均缺失。本地归档已通过完整 SHA 和 CRC 核验，与当前官方 LFS 资产标识一致。重下同一资产仍会缺少这 10 段。[官方仓库](https://github.com/worldbench/3EED/blob/main/README.md) · [官方数据包](https://huggingface.co/datasets/RRRong/3EED/blob/main/3eed_dataset.zip) · [包结构及核验回执](results/exposure_receipt.json)

## 已锁定的三组

| 配置 | 历史列 | 固定实现 |
|---|---|---|
| H | `P` | 原全点四尺度 PCA 补框 `C_E_Qsize` |
| 同划分固定 PCA | `C_pca` | 固定原 Qwen 尺寸，在拟合侧同一选区和地面上，按 PCA 朝向重估中心 |
| 双姿态射线选择 | `C_ray` | 与 `C_pca` 共用拟合侧选区、地面和尺寸。分别按 PCA、Qwen 朝向重估中心，再用独立选择回波评分 |

沿用 4×4 像素格的拟合、选择和审计划分。候选至少要有 5 条选择射线。原代价为相机光心首交面深度残差，经 Qwen 水平短边一半归一化，分母至少 0.1 米，漏击记 1。两候选代价并列或支持不足时保留固定 PCA。缺拟合节点时沿用旧 Qwen 中位深度回退。姿态、尺寸和阈值按旧发布源码固定。未来有合格录制时，对同一批帧在新标签打开前封存三组框，计算阈值、mIoU、逐平台救回与误改及录制级配对区间。[原候选源码](https://github.com/Zonglin-He/A/blob/8e00f0eff790f614a34cb5612c18ac6ce972795a/studies/3eed_pose_consistent_visible_surface_grounding_20261121/surface_grounding.py) · [配置锁](CANDIDATE_LOCK.json)

## 历史三组复算

下面的结果来自既有逐帧分数。Acc@50 与 mIoU 均为三个平台等权，救回与误改按帧计数。两批子集有交集。[全部 416 帧和逐平台复算](results/historical_three_arms.csv) · [旧评分](https://github.com/Zonglin-He/A/blob/8e00f0eff790f614a34cb5612c18ac6ce972795a/studies/3eed_pose_consistent_visible_surface_grounding_20261121/results/evaluation_receipt.json)

| 历史范围 | H Acc@50／mIoU | 同划分 PCA | 双姿态射线 | 双姿态对 H 救回／误改 | 双姿态对同划分 PCA 救回／误改 |
|---|---:|---:|---:|---:|---:|
| 普通 Validation，288 帧 | 29.51%／31.92% | 29.51%／30.67% | **32.29%**／31.54% | 19／11 | 10／2 |
| 与最初选型录制分离，224 帧 | 27.43%／30.48% | 29.51%／29.76% | **32.47%**／31.74% | 19／8 | 9／2 |

Validation 上，双姿态相对 H 的 Acc@50 增加 2.78 pp，mIoU 下降 0.38 pp。224 帧中，双姿态相对 H 的 Acc@50 增量为 +5.03 pp，录制重抽样区间为 [−0.35, +10.16] pp。相对同划分 PCA 的增量为 +2.95 pp，区间为 [+0.31, +5.79] pp。Quad 的三次净救回集中在一段录制，Drone 在该对照上救回和误改均为零。[原姿态实验](https://github.com/Zonglin-He/A/blob/8e00f0eff790f614a34cb5612c18ac6ce972795a/studies/3eed_pose_consistent_visible_surface_grounding_20261121/REPORT_ZH.md)

## 语义分支归档

241 帧的 S 和 GS 原始语义回答均符合约定格式，其中 240 帧类别与子类型一致。按全量封存结果，唯一一次 GS 相对 S 的成框变化发生在普通 TRAIN 扩展，288 个普通 Validation 帧均保持 S 的成框。两组的 Acc@50 成功帧 ID 相同。H 与 S 同时改变了图像上下文、提示和尺寸接口，S 的退化归于整个新读出流程。结束当前分类到尺寸的提示迭代，既有结果和失败案例已归档。[语义分支逐帧数据](https://github.com/Zonglin-He/A/blob/ba805be17e7a3000c645b151da5a3421283e5985/studies/3eed_qwen_observed_semantic_size_20261121/results/rows.csv) · [复算回执](results/recount_receipt.json)

本轮完成录制审计、配置锁定和历史指标复算。独立确认的输入缺口是有 RGB、LiDAR、标定和任务标注且可核实未参与开发的普通录制。新预测数和新 GT 评分数均为 0，FINAL 未访问。带原本地封存清单时可用 `python audit_recordings.py` 重算录制计数，`python recount_locked.py` 复算历史结果。[决定](DECISION.json)
