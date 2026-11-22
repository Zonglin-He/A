# 只缩顶部 · 单帧 3D 指代论文实验

在 9,310 条普通 Validation 上，**只缩顶部的 Acc@50 为 37.23%**，较冻结 `C_ray_simple` 高 **+2.82 pp**，救回 249 条、误改 7 条。在录制、RGB 和 LiDAR 都与该清单分离的 1,536 条复测上，Acc@50 为 **34.18%**，差值 **+2.67 pp**，救回 48 条、误改 7 条。两个队列均按三个目标平台等权。 [完整实验图](results/paper_figure.pdf) · [固定协议](PROTOCOL.json) · [独立核验](results/independent_recount.json)

## 主比较

同一帧 RGB、LiDAR、标定和指代表达使用同一套冻结 Qwen3-VL-4B-Instruct 回答。原生 Qwen 框按统一规则转成 LiDAR 坐标系的直立框。表中各格为 **Acc@25 / Acc@50 / mIoU（%）**，阈值采用严格 `IoU > 0.25` 和 `IoU > 0.50`。回退、格式失败与无框结果均计入分母。 [Validation](results/validation_metrics.csv) · [录制隔离复测](results/extension_metrics.csv)

| 方法 | 普通 Validation | 录制隔离复测 |
| --- | --- | --- |
| 原生 Qwen，统一成直立框 | 4.29 / 0.76 / 3.18 | 1.76 / 0.26 / 1.59 |
| Qwen＋窗口中位深度 | 48.62 / 21.14 / 25.97 | 47.46 / 15.82 / 24.59 |
| 单尺度＋补框 | 49.23 / 28.73 / 29.26 | 54.88 / 26.50 / 29.90 |
| H，四尺度＋补框 | 51.62 / 29.71 / 30.24 | 57.36 / 27.86 / 30.99 |
| 固定 PCA 姿态及同选区 | 51.17 / 27.41 / 29.22 | 56.51 / 26.89 / 30.00 |
| 全类别双姿态射线选择 | 53.41 / 33.10 / 31.83 | 57.81 / 30.21 / 31.83 |
| 冻结 C_ray_simple | 53.71 / 34.41 / 32.26 | 58.92 / 31.51 / 32.51 |
| 三轴收缩 | 54.15 / 37.47 / 34.01 | 59.11 / 34.44 / 34.03 |
| **只缩顶部** | 54.34 / 37.23 / 33.88 | 58.98 / 34.18 / 33.73 |

普通 Validation 有 98 段录制和 7,221 个物理时刻，沿用此前允许访问的 9,310 条描述。复测覆盖固定 Ordinary TRAIN 清单的 1,536 条描述和 98 段录制，其录制、RGB 文件和 LiDAR 文件均与此 Validation 清单分离。两者都曾用于项目研发。复测仅用于冻结评估，新增模型训练与目标端权重更新均为 0。 [Validation 清单回执](../3eed_qwen_full_ordinary_validation_20261121/results/input_seal.json) · [复测清单回执](results/extension_input_seal.json) · [历史录制曝光审计](../3eed_pose_recording_exposure_audit_20261121/REPORT_ZH.md)

## 平台、录制与类别

三平台各自先按全部描述计分，再平均。救回和误改是实际配对计数。95% 区间使用每个平台内 4,000 次配对录制重抽样。Drone 与 Quad 的录制数较少，区间反映现有录制的变化。逐个留掉一段录制时，主方法相对冻结版的三平台 Acc@50 差值范围为 [+1.60, +3.02] pp 和 [+0.61, +3.54] pp。 [Validation 逐录制配对](results/validation_recording_pairs.csv) · [复测逐录制配对](results/extension_recording_pairs.csv) · [Validation 留一录制](results/validation_leave_one_out.csv) · [复测留一录制](results/extension_leave_one_out.csv)

| 队列 | 目标平台 | 描述数 | 录制数 | 冻结 Acc@50 | 主方法 25 / 50 / mIoU | Δ50 (pp) | 救回 / 误改 | Δ50 95% 区间 (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 普通 Validation | Waymo | 3,782 | 93 | 58.30 | 80.49 / 59.04 / 50.20 | +0.74 | 30 / 2 | [+0.41, +1.17] |
| 普通 Validation | Drone | 2,792 | 2 | 21.20 | 34.20 / 25.97 / 23.38 | +4.76 | 136 / 3 | [+1.10, +5.02] |
| 普通 Validation | Quad | 2,736 | 3 | 23.72 | 48.32 / 26.68 / 28.06 | +2.96 | 83 / 2 | [+0.00, +4.10] |
| 普通 Validation | 三平台等权 | 9,310 | 98 | 34.41 | 54.34 / 37.23 / 33.88 | +2.82 | 249 / 7 | [+0.59, +3.25] |
| 录制隔离复测 | Waymo | 512 | 94 | 60.94 | 82.42 / 61.91 / 51.60 | +0.98 | 5 / 0 | [+0.20, +1.90] |
| 录制隔离复测 | Drone | 512 | 2 | 20.51 | 46.88 / 27.34 / 26.79 | +6.84 | 41 / 6 | [+0.66, +9.44] |
| 录制隔离复测 | Quad | 512 | 2 | 13.09 | 47.66 / 13.28 / 22.80 | +0.20 | 2 / 1 | [+0.00, +0.47] |
| 录制隔离复测 | 三平台等权 | 1,536 | 98 | 31.51 | 58.98 / 34.18 / 33.73 | +2.67 | 48 / 7 | [+0.41, +3.73] |

标注类别仅用于事后分层。此表逐描述加权。距离、点数、相机是否高于原顶面和可行区间宽度在附表分列。 [Validation 分层](results/validation_strata.csv) · [复测分层](results/extension_strata.csv)

| 队列 | 类别 | 描述数 | 冻结 Acc@50 | 主方法 Acc@50 | Δ50 (pp) |
| --- | --- | --- | --- | --- | --- |
| 普通 Validation | car | 7,082 | 41.57 | 44.97 | +3.40 |
| 普通 Validation | pedestrian | 2,072 | 20.13 | 20.13 | +0.00 |
| 普通 Validation | cyclist | 77 | 61.04 | 61.04 | +0.00 |
| 录制隔离复测 | car | 897 | 45.26 | 49.83 | +4.57 |
| 录制隔离复测 | pedestrian | 614 | 11.24 | 11.24 | +0.00 |
| 录制隔离复测 | cyclist | 20 | 35.00 | 35.00 | +0.00 |

## 高度消融与同输入对照

固定 `C_ray_simple` 给出的框底 $b$、水平足迹、朝向和选框射线。令 $P$ 为原框内实测支持点，$E$ 为原 Selection 射线命中的首交点。设原顶面为 $t_0$，可行区间的下端为 $m=\max_{w\in P\cup E} w_z$。只在水平足迹与底面固定的嵌套框族中移动顶面，$t\in[m,t_0]$ 恰好描述同时保留这些点和原射线首交的条件。收缩使框成为原框的子集，原本未命中的射线无法增加交点。对原本命中的射线，只要保留原首交点，新框的首交仍发生在同一位置。若顶面低于 $m$，至少移除一个必须保留的点。

设原生框轴方向为 $R$、尺寸为 $d_1,d_2,d_3$，外包直立化后的高度是 $\sum_k |R_{zk}|d_k$。物体长轴倾斜时也可能贡献到这项外包高度。读出改用最接近竖直的原生轴的尺寸 $h_Q$ 作为先验，选取 $t=\min(t_0,\max(b+h_Q,m+\varepsilon))$，其中 $\varepsilon=10^{-8}\,\mathrm{m}$。代码另有空首交和相机处于原框内部时的原框回退。完整约束的度量意义限定在实际用于选框的一组相机代理射线，Audit 留出射线另行检验。 [现有入口](../3eed_qwen_extent_height_only_20261122/live.py) · [公式实现](../3eed_qwen_extent_height_only_20261122/extent_clip.py) · [两队列实现](paper_eval.py)

| 方法 | 普通 Validation | 录制隔离复测 |
| --- | --- | --- |
| 只用原生高度 | 54.27 / 37.13 / 33.73 | 58.92 / 34.05 / 33.56 |
| 原生高度＋框内实测点 | 54.35 / 37.20 / 33.86 | 58.98 / 34.18 / 33.70 |
| 原生高度＋原首交点 | 54.31 / 37.21 / 33.85 | 58.98 / 34.05 / 33.70 |
| **完整只缩顶部** | 54.34 / 37.23 / 33.88 | 58.98 / 34.18 / 33.73 |
| 仅取可行区间下端 | 53.47 / 32.85 / 31.88 | 57.88 / 30.66 / 31.97 |
| 同平台打乱原生高度 | 54.11 / 37.23 / 33.68 | 59.18 / 34.24 / 33.52 |
| 同输入软射线搜索 | 54.34 / 37.20 / 33.84 | 58.98 / 34.18 / 33.70 |
| 固定 25% Selection 射线 | 54.35 / 37.21 / 33.87 | 58.98 / 34.18 / 33.73 |
| 固定 50% Selection 射线 | 54.34 / 37.22 / 33.88 | 58.98 / 34.18 / 33.74 |
| 对侧队列中位数常量高度（追加对照） | 54.10 / 37.01 / 33.60 | 58.79 / 34.11 / 33.46 |

软搜索控制保持相同候选框底、姿态、测量和原生高度区间，搜索 17 个等距高度及三个受约束高度。目标是命中射线的三维测距误差均值加 10 倍未包容点比例。这一控制移植了 OVM3D-Det 的搜索目标。原论文还有候选构建、伪 LiDAR 和后续自训练。打乱高度保持平台内的高度分布，切断物体与高度的配对。追加常量控制用另一队列冻结输出的原生高度中位数，分别为 1.720 m（用于 Validation）和 1.710 m（用于复测）。 [OVM3D-Det §3.2 和附录 E](https://arxiv.org/html/2411.15657v1#S3.SS2) · [常量对照封存](results/constant_prior_seal.json)

Validation 从冻结版到仅用原生高度已获得 **+2.72 pp**。完整约束相对保留实测点而去掉原首交的版本只增加 **+0.026 pp**，同时将 Selection 射线变化由 **15,246** 降至 **0**。逐物体高度相对平台内打乱高度的 Validation mIoU 差值为 **+0.202 pp**，Acc@50 差值为 **-0.001 pp**。与常量高度相比，两个队列的 Acc@50 只多 **+0.220 和 +0.065 pp**。结果支持尺寸先验提供绝大部分准确率增量，首交约束提供观测不变性。 [Validation 配对](results/validation_pairs.csv) · [复测配对](results/extension_pairs.csv) · [常量配对](results/constant_prior_pairs.csv)

## 保留了哪些几何证据

计数只汇总对应方案确实改框的记录。一个测量出现在多个描述时计作多次使用。完整方法在 Validation 上有 867 条收紧了仅实测点版本的高度下界，其中 345 条被约束到原顶面。复测分别为 114 和 43 条。降低选中射线数的两个控制完整保留各自所用子集，表中 Selection 变化全部来自未用于修正的原 Selection 射线。 [机制统计](results/mechanism.csv) · [Validation 射线审计](results/validation_geometry.csv) · [复测射线审计](results/extension_geometry.csv)

| 队列 | 只缩顶部的变体 | 改框数 | Selection 变化 / 射线数 | Audit 变化 / 射线数 | 丢失实测点 |
| --- | --- | --- | --- | --- | --- |
| 普通 Validation | 只用原生高度 | 5,904 | 23,987 / 687,409 | 23,551 / 687,828 | 24,498 |
| 普通 Validation | 原生高度＋框内实测点 | 5,904 | 15,246 / 687,409 | 14,942 / 687,828 | 0 |
| 普通 Validation | 原生高度＋原首交点 | 5,559 | 0 / 560,883 | 495 / 560,839 | 1,844 |
| 普通 Validation | **完整只缩顶部** | 5,559 | 0 / 560,883 | 411 / 560,839 | 0 |
| 普通 Validation | 仅取可行区间下端 | 5,559 | 0 / 560,883 | 2,227 / 560,839 | 0 |
| 普通 Validation | 同平台打乱原生高度 | 4,488 | 0 / 441,864 | 500 / 441,819 | 0 |
| 普通 Validation | 同输入软射线搜索 | 5,838 | 12,693 / 658,035 | 12,471 / 658,125 | 115 |
| 普通 Validation | 固定 25% Selection 射线 | 5,617 | 1,313 / 570,008 | 1,550 / 570,045 | 0 |
| 普通 Validation | 固定 50% Selection 射线 | 5,580 | 495 / 562,987 | 830 / 563,013 | 0 |
| 录制隔离复测 | 只用原生高度 | 693 | 4,210 / 88,080 | 4,262 / 88,227 | 4,139 |
| 录制隔离复测 | 原生高度＋框内实测点 | 693 | 2,574 / 88,080 | 2,599 / 88,227 | 0 |
| 录制隔离复测 | 原生高度＋原首交点 | 650 | 0 / 69,261 | 65 / 69,292 | 397 |
| 录制隔离复测 | **完整只缩顶部** | 650 | 0 / 69,261 | 53 / 69,292 | 0 |
| 录制隔离复测 | 仅取可行区间下端 | 650 | 0 / 69,261 | 317 / 69,292 | 0 |
| 录制隔离复测 | 同平台打乱原生高度 | 519 | 0 / 55,837 | 69 / 55,849 | 0 |
| 录制隔离复测 | 同输入软射线搜索 | 685 | 2,043 / 84,095 | 2,047 / 84,199 | 13 |
| 录制隔离复测 | 固定 25% Selection 射线 | 663 | 238 / 72,785 | 267 / 72,885 | 0 |
| 录制隔离复测 | 固定 50% Selection 射线 | 653 | 73 / 69,966 | 111 / 70,021 | 0 |

## 用真值计算顶面上限

设水平相交面积为 $A_\cap$，预测足迹面积为 $A$，标注底面、顶面和体积为 $g_b,g_t,V_g$。交叠体积为 $I(t)=A_\cap[\min(t,g_t)-\max(b,g_b)]_+$，IoU 为 $I(t)/(A(t-b)+V_g-I(t))$。存在正交叠时，IoU 在 $g_t$ 前非减，在其后非增。因此将标注顶面投影到实际的可行区间即可得到最优 IoU。对 151 条固定样本又用 33 个高度网格逐条验证最优性。真值在预测封存后才参与这项诊断。 [上限和失败类型](results/oracle_regimes.csv) · [成功、误改、约束锁定和失败示例](results/example_cases.csv)

| 队列 | 可收缩分支 | 原顶面 MAE (m) | 主方法 MAE (m) | 上限 MAE (m) | 主方法 Acc@50 | 上限 Acc@50 | Δ50 (pp) | 可再救回 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 普通 Validation | 5,904 | 0.348 | 0.223 | 0.093 | 37.23 | 38.63 | +1.40 | 123 |
| 录制隔离复测 | 693 | 0.338 | 0.192 | 0.073 | 34.18 | 35.35 | +1.17 | 18 |

Validation 可收缩分支有 1,358 条的标注顶面高于原顶面，其中 5 条发生阈值误改。原顶面构成收缩的上界。可行区间内的最优高度也继承原水平足迹和底面的误差。 [上限失败分解](results/oracle_regimes.csv)

## 3EED 已发表实现的历史参考

三份原单源模型各跨到另外两个平台。历史复跑与主表使用相同普通目标清单规模和有向 IoU 口径，分别汇总 18,620 个方向预测与 9,310 个单模型描述预测。官方 BBF 使用评测类名跨度，纯文本列只使用原句。原论文用单平台训练、跨平台测试构成迁移，本文用一个冻结通用 VLM 覆盖三个目标平台。历史表作为不同输入和训练条件的参考。 [3EED §4.1 与附录 B](https://arxiv.org/html/2511.01755v1#S4.SS1) · [已发布本地六方向复跑](https://github.com/Zonglin-He/A/blob/0c68f2c94c0d00cf91305f16df4e0884c97300e0/studies/3eed_official_cross_platform_validation_ablation_20261118/REPORT_ZH.md) · [方向原表](results/historical_3eed.csv)

| 历史方法 | 方向记录数 | 六方向等权 Acc@25 / Acc@50 / mIoU（%） |
| --- | --- | --- |
| 3EED 官方 BBF＋原几何，使用类别跨度 | 18,620 | 22.55 / 6.16 / 11.79 |
| 3EED 原始纯文本读出 | 18,620 | 22.48 / 6.15 / 11.75 |
| 3EED 官方 BBF＋PCA，使用类别跨度 | 18,620 | 25.51 / 9.94 / 13.55 |
| 3EED 原始纯文本读出＋PCA | 18,620 | 25.34 / 9.95 / 13.52 |
| 3EED 四尺度 v3 | 18,620 | 27.30 / 11.10 / 14.77 |

WeakM3D 和 OVM3D-Det 通过测量点到首交点的残差来优化框。OpenBox 的高度细化估计地面和底面。Segment, Lift and Fit 与 AmodalBox 则引入完整形状先验来求框。这里研究固定姿态与底面后的一维几何读出，原首交位置直接确定可行区间，随后才用高度先验选值。它允许报告每一帧的可收缩量和固定足迹下的精度上限。 [WeakM3D §3.3](https://arxiv.org/pdf/2203.08332) · [OpenBox 附录 C.2](https://arxiv.org/html/2512.01352v2#A3.SS2) · [SLF §3.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11297.pdf) · [AmodalBox §3.2](https://arxiv.org/html/2611.10821v1#S3.SS2)

## 运行成本与复验材料

单线程 Intel(R) Core(TM) i5-14600KF，96 条描述各做三次交叉配对，读取已经预热。完整几何中位时间从 32.20 ms 到 32.32 ms，单独的高度操作为 **0.227 ms**，第 90 百分位为 0.354 ms。Qwen 两个回答与冻结对照共用。普通 Validation 复用了封存回答。录制复测新增 **2,816** 个回答且复用 **256** 个，主要运行秒数为 1259.8，NVIDIA GeForce RTX 5090 峰值张量分配 11.73 GiB。此处 CPU 几何时延和 GPU 批量生成吞吐量分别测量。 [CPU 计时](results/timing.json) · [Qwen 复测回执](results/extension_qwen_seal.json)

在有锁定队列和只读模型权重的研究环境中运行以下命令。预测阶段写入私有载荷并封存，评分阶段核对散列后使用目标标注。两个队列的匿名数值、每段录制统计和两套 IoU 评分检查随结果发布。三种正式几何入口在 1,536 条复测上逐框复现。FINAL 访问次数为 0。 [正式入口复放](results/extension_live_replay.json)

```bash
study=studies/3eed_qwen_height_only_paper_20261123
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
crossplatform/env/bin/python "$study/transfer.py" register
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$study/transfer.py" infer --model "$QWEN_MODEL_DIR"
for cohort in validation extension; do
  crossplatform/env/bin/python "$study/paper_eval.py" propose "$cohort" --workers 8
  crossplatform/env/bin/python "$study/paper_eval.py" evaluate "$cohort" --workers 8
done
python "$study/audit_paper.py"
crossplatform/env/bin/python "$study/constant_prior.py" --workers 8
crossplatform/env/bin/python "$study/benchmark_cpu.py"
crossplatform/env/bin/python "$study/historical.py" --history "$VERIFIED_GIT_CHECKOUT"
crossplatform/env/bin/python "$study/replay_extension.py" --workers 8
crossplatform/env/bin/python "$study/make_figures.py"
python "$study/summarize.py"
```
