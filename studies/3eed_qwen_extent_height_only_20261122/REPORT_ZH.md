# 首交等价收缩的闭式改写与纯高度简化

**将只缩顶部登记为简化候选。** 9,310 条既有普通 Validation 的三平台等权 Acc@50 为 **37.2304%**。相对完整三轴版低 **0.2409 pp**，相对冻结 `C_ray_simple` 高 **2.8215 pp**，保留了原 Acc@50 增量的 **92.1%**。 [全量指标](results/metrics.csv) · [配对结果](results/paired.csv)

## 从原比例式到尺寸截断

固定旧框的姿态、地面接触面和原水平锚定方式。记 `D0` 为旧框尺寸、`Dint` 为 Qwen 原生尺寸、`Dreq` 为容纳保护点所需的最小尺寸。逐轴收缩可写为 `min(D0, max(Dint, Dreq))`。原来的边界余量保留。对 9,310 条逐框比较，最大坐标或尺寸差为 **3.55×10⁻¹⁵ m**，Acc@25 和 Acc@50 的逐例判定完全一致。 [实现](extent_clip.py) · [逐框核对](results/prediction_seal.json)

简化版只应用高度分量。设 `W` 为原框内实测目标点与旧 Selection 首交点的并集。

```text
h = min(h0, max(h_Q, max(w.z for w in W) - z_bottom + ε))
center_z = z_bottom + h / 2
```

四尺度选区和双姿态读出提供原框。高度更新只移动顶面，水平中心、长宽、底面和朝向沿用原框。新框包含保护点且位于原框内，因此每条原 Selection 射线保留同一首交点与命中状态。 [协议](PROTOCOL.json) · [推理入口](live.py)

## 配对结果

| 配置 | Acc@25 | Acc@50 | mIoU | Acc@50 对冻结版救回／误改 |
|---|---:|---:|---:|---:|
| 冻结 `C_ray_simple` | 53.7083% | 34.4089% | 32.2560% | — |
| 三轴尺寸截断 | 54.1542% | 37.4713% | **34.0130%** | 273／13 |
| 只缩顶部 | **54.3367%** | 37.2304% | 33.8785% | 249／7 |

只缩顶部与三轴版直接配对，Acc@50 救回 **14** 条、误改 **32** 条。平台内按录制重抽样的 Acc@50 差值区间为 **[−0.337, +0.076] pp**。mIoU 差值为 **−0.1346 pp**，区间 **[−0.174, −0.007] pp**。 [重抽样与封存回执](results/evaluation_receipt.json)

| 平台 | 描述数 | 冻结 Acc@50 | 三轴 Acc@50 | 只缩顶部 Acc@50 | 只缩顶部对三轴差值 |
|---|---:|---:|---:|---:|---:|
| Waymo | 3,782 | 58.30% | 58.86% | 59.04% | +0.19 pp |
| Drone | 2,792 | 21.20% | 26.25% | 25.97% | −0.29 pp |
| Quad | 2,736 | 23.72% | 27.30% | 26.68% | −0.62 pp |

## 射线与推理复现

只缩顶部修正 **5,559** 条描述，保留了这些框原本容纳的 **1,472,587** 个实测点实例，顶面收缩中位数 **0.190 m**。涉及 **560,883** 条 Selection 射线，首交状态变化为 **0**，共同命中的最大深度差为 **7.11×10⁻¹⁵ m**。留作检查的 **560,839** 条 Audit 射线中，**411** 条的命中或首交深度发生变化。首交保证对应选框实际使用的 Selection 射线。 [逐组射线统计](results/ray_checks.csv)

评测沿用已曝光的普通 Validation，全部回退计入分母。两种截断的候选先封存，再由原有 IoU 评分器评估。两种入口在 9,310 条上均逐框复现封存结果。CPU 复用冻结回答，新 Qwen 查询、训练与 FINAL 访问均为零。简化入口默认只缩顶部，`height_only=False` 可复现三轴结果。 [入口复放](results/live_replay.json) · [独立复算](results/verification.json)

在具备既有封存输入的实验环境中，用以下命令封存、评分、复放入口和独立复算。

```bash
study=studies/3eed_qwen_extent_height_only_20261122
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
crossplatform/env/bin/python "$study/run_study.py" propose --workers 8
crossplatform/env/bin/python "$study/run_study.py" evaluate --workers 6
crossplatform/env/bin/python "$study/replay_live.py" --workers 8
crossplatform/env/bin/python "$study/verify.py"
```
