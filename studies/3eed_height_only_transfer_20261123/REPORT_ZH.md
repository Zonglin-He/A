# 只缩顶部 · 同尺寸消融、冻结三维前端与外部场景

本轮延续原论文主方法与同一有向 IoU，增加同尺寸位置对照、冻结 WildDet3D 三维前端和 Talk2Car 外部自然语言场景。普通 Validation 与录制隔离复测仍分别为 **9,310** 和 **1,536** 条。完整主表与已有高度先验、软搜索、真值顶面上限和效率实验见第一轮报告。 [前一轮完整实验](../3eed_qwen_height_only_paper_20261123/REPORT_ZH.md) · [本轮协议](PROTOCOL.json) · [三维前端划分补充](WILDDET_SPLIT_ADDENDUM.json) · [外部校验补充](EXTERNAL_BRIDGE_ADDENDUM.json) · [补充图](results/transfer_figure.pdf)

固定最终框高和保留的几何证据后，Validation 上保留原底面比尽量对称收缩高 **2.12 pp Acc@50**。换成冻结 WildDet3D 三维输出后，顶缩的配对增量为 **+0.48 pp**。Talk2Car–mini 的配对变化为 **+0.00 pp**。

## 同样的框高，应该从哪一端收缩

原有选区、两个姿态候选、最终获选姿态及目标尺寸完全配对。设原底面为 $b$、原框高为 $h_0$、主方法得到的框高为 $h$。三个收缩方案共用同一个 $h$，底面分别上移 0、$(h_0-h)/2$ 和 $h_0-h$。第四项将对称上移投影到仍保留全部原框内实测点与原首交点的范围。

令 $W$ 为这些保护点。固定 $h$ 和水平足迹时，可行上移区间为

$$
0\le \delta \le \min\left\{h_0-h,\min_{w\in W} w_z-b\right\}.
$$

主方法的顶面已经容纳 $W$，因此再上移只会因底面略过最低保护点而失去证据。实现保留 $10^{-8}$ 米外向余量。各格为 **Acc@25 / Acc@50 / mIoU（%）**，两组均三平台等权。 [实现](anchor_control.py) · [逐录制对照](results/validation_anchor_recordings.csv)

| 位置对照 | 普通 Validation | 录制隔离复测 |
| --- | --- | --- |
| 冻结 C_ray_simple | 53.71 / 34.41 / 32.26 | 58.92 / 31.51 / 32.51 |
| 对称收缩 | 53.69 / 33.98 / 32.08 | 58.85 / 31.12 / 32.47 |
| 只缩底部 | 52.58 / 28.57 / 30.01 | 57.23 / 27.93 / 30.85 |
| 最近的证据等价对称位置 | 53.87 / 35.11 / 32.64 | 58.98 / 33.27 / 33.16 |
| **只缩顶部** | 54.34 / 37.23 / 33.88 | 58.98 / 34.18 / 33.73 |

| 队列 | 位置对照 | 相对只缩顶部的中心移动数 | Selection 射线变化 | 失去原框内实测点 | 相对冻结 ΔAcc@50 (pp) | 救回 / 误改 |
| --- | --- | --- | --- | --- | --- | --- |
| Validation | 对称收缩 | 5,559 | 6,000 | 20,650 | -0.43 | 12 / 51 |
| Validation | 只缩底部 | 5,559 | 27,628 | 88,577 | -5.83 | 7 / 530 |
| Validation | 最近的证据等价对称位置 | 5,406 | 0 | 0 | +0.70 | 85 / 24 |
| Validation | **只缩顶部** | 0 | 0 | 0 | +2.82 | 249 / 7 |
| 录制隔离复测 | 对称收缩 | 650 | 1,098 | 3,654 | -0.39 | 1 / 7 |
| 录制隔离复测 | 只缩底部 | 650 | 3,820 | 11,949 | -3.58 | 5 / 60 |
| 录制隔离复测 | 最近的证据等价对称位置 | 636 | 0 | 0 | +1.76 | 30 / 3 |
| 录制隔离复测 | **只缩顶部** | 0 | 0 | 0 | +2.67 | 48 / 7 |

Validation 上，保留原底面的 Acc@50 比同高对称收缩高 **3.25 pp**，比保留同样观测的对称位置高 **2.12 pp**，后一差值的录制重抽样区间为 **[+0.52, +2.49] pp**。已缩框样本中，固定底面的下表面 MAE 为 **0.200 m**，证据等价的对称位置为 **0.242 m**。保护点约束限制可行变动，原下表面提供定位锚点。 [Validation 核验](results/validation_anchor_evaluation.json) · [复测核验](results/extension_anchor_evaluation.json)

固定体积后，IoU 随竖直重叠长度单调增加。将标注中心投影到可行上移区间，求得同高上移上限。141 条固定样本以 21 个网格点复验。相对原底面，上限在 Validation 只多救回 **12** 条和 **+0.14 pp**，在复测多 **8** 条和 **+0.52 pp**。该上限只使用已封存后的标注。 [真值位置上限](ground_attribution.py) · [Validation 上限表](results/validation_fixed_height_oracle_pairs.csv) · [复测上限表](results/extension_fixed_height_oracle_pairs.csv)

## 同一指代对象，改用冻结 WildDet3D 的三维输出

在 **9,310** 条普通 Validation 上复用 Qwen 的 2D 窗口和类别路由。WildDet3D 的已发布训练配置使用 3EED 的检测和语言 TRAIN。在读取模型权重前，已按作者划分将本项比较从 Ordinary TRAIN 复测改到完整普通 Validation。 [作者训练数据](https://github.com/allenai/WildDet3D/blob/1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a/docs/TRAINING_DATA.md#3eed) · [划分补充](WILDDET_SPLIT_ADDENDUM.json)

作者的 `geometric` 模式在同一图像上按窗输出单个三维框。比较该模型的 RGB、RGB 加最近回波深度，以及从其纯 RGB 输出接入同一选区、双姿态和顶面修正。RGB 与深度共用相机内参，存在畸变时将图像和窗同时校正。光学深度来自该帧 LiDAR 投影，空像素保留为空。四元数框与 Qwen 框按同一规则转成直立框，以相同有向 IoU 计分。 [作者几何窗与深度接口](https://github.com/allenai/WildDet3D/blob/1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a/docs/INFERENCE.md) · [公开权重](https://huggingface.co/allenai/WildDet3D) · [推理与封存](wilddet_eval.py)

首轮联接沿用了接口文档的尺寸顺序。核对正式评分器后，按它的轴定义重新解释了 **18,590** 个封存框的尺寸。原来的中心、旋转、窗口和模型权重均用于重放。 [轴重放回执](results/wilddet3d_axis_replay.json) · [作者评分器](https://github.com/allenai/WildDet3D/blob/1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a/wilddet3d/eval/detect3d.py#L410)

| 三维提供者与读出 | Waymo | Drone | Quad | 三平台等权 |
| --- | --- | --- | --- | --- |
| Qwen 三维输出＋冻结读出 | 80.04 / 58.30 / 49.30 | 32.92 / 21.20 / 20.42 | 48.17 / 23.72 / 27.05 | 53.71 / 34.41 / 32.26 |
| **Qwen 三维输出＋只缩顶部** | 80.49 / 59.04 / 50.20 | 34.20 / 25.97 / 23.38 | 48.32 / 26.68 / 28.06 | 54.34 / 37.23 / 33.88 |
| WildDet3D 原生 RGB | 62.24 / 41.25 / 38.16 | 29.41 / 14.68 / 17.82 | 17.54 / 8.59 / 10.62 | 36.40 / 21.51 / 22.20 |
| WildDet3D 原生 RGB＋LiDAR | 62.16 / 41.59 / 38.31 | 28.47 / 13.57 / 17.08 | 6.43 / 1.35 / 4.59 | 32.36 / 18.84 / 19.99 |
| WildDet3D 三维输出＋冻结读出 | 88.97 / 69.17 / 58.01 | 40.65 / 32.41 / 28.41 | 51.50 / 30.67 / 30.34 | 60.37 / 44.08 / 38.92 |
| WildDet3D 三维输出＋只缩顶部 | 88.97 / 69.30 / 58.22 | 40.97 / 32.99 / 29.21 | 51.57 / 31.40 / 31.06 | 60.51 / 44.56 / 39.50 |

在 WildDet3D 三维输出上，只缩顶部对同前端冻结读出的 Acc@50 变化为 **+0.48 pp**，录制区间 **[+0.03, +0.59] pp**，救回／误改为 **48／7**。Qwen 主方法相对 WildDet3D RGB＋LiDAR 的直接输出为 **+18.39 pp**。各组共用 Qwen 的二维定位。

使用 WildDet3D 纯 RGB 三维输出，再接入物理选区和双姿态读出的组合，Acc@50 已达到 **44.08%**，加入顶缩后为 **44.56%**。两种三维前端的系统差距同时包含训练数据与预测差异。顶缩的增量由各前端内部的开关对照确定。

WildDet3D 三维输出有 **5,962** 条启用双姿态，**5,361** 个框实际收缩。已改框样本中，Selection 首交变化 **0／492,424**，额外 Audit 首交变化 **367／492,356**。 [逐条度量](results/wilddet3d_rows.csv.gz)

共获得 RGB 的 **9295** 个及带深度的 **9295** 个有效三维框。作者输出存储尺寸的顺序为 W、L、H，OpenCV 局部轴对应 L、H、W。角点与作者评分器的最大差为 1.42e-14 m。包含加载和封存的模型推理阶段累计耗时 47.8 分钟，NVIDIA GeForce RTX 5090 在主推理进程的峰值张量分配为 6.45 GiB。独立重算两阈值的配对数目。 [逐平台对照](results/wilddet3d_pairs.csv) · [权重和推理回执](results/wilddet3d_box_provider_seal.json) · [度量核验](results/wilddet3d_evaluation.json)

## 外部数据上的单帧语言复测

用官方 Talk2Car 命令与公开 nuScenes mini 的 CAM_FRONT 元数据精确匹配。作者加载器以相机 `sample_data` 读取该字段。每条命令只取一张原图、同录制内时间最近的一次 LIDAR_TOP，按各自时间的标定和车辆位姿配准。共 **113 条指令**、**69 个相机记录**、**6 段录制**、**9 条相机关键帧指令**。最大时差 **16.04 ms**。交集来自原 Talk2Car Train 划分，本实验只做冻结评估，交集内 Val 指令为 0。 [Talk2Car 作者加载器](https://github.com/talk2car/Talk2Car/blob/3fde52aee3341312196ab9b2f999337eff0693a0/talk2car.py) · [nuScenes 坐标与数据元](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html) · [样本封存](results/talk2car_input_seal.json)

| 冻结读出 | 全集 | 关键帧子集 | 扫帧子集 |
| --- | --- | --- | --- |
| 原生 Qwen 直立框 | 22.12 / 12.39 / 12.40 | 22.22 / 0.00 / 7.03 | 22.12 / 13.46 / 12.87 |
| Qwen＋窗口中位深度 | 20.35 / 13.27 / 11.52 | 11.11 / 11.11 / 6.34 | 21.15 / 13.46 / 11.97 |
| 冻结 C_ray_simple | 16.81 / 11.50 / 9.85 | 11.11 / 11.11 / 7.03 | 17.31 / 11.54 / 10.10 |
| **只缩顶部** | 16.81 / 11.50 / 9.77 | 11.11 / 11.11 / 6.11 | 17.31 / 11.54 / 10.09 |

只缩顶部相对冻结读出的 Acc@50 变化为 **+0.00 pp**，救回／误改 **0／0**，对应录制重抽样区间 **[+0.00, +0.00] pp**。这项外部测量使用作者发布的相机系三维标注并转到 LiDAR 系。在有二维框标注的 113 条上，Qwen 的二维 Acc@50 为 **29.20%**。几何评分输入完整保留无框和回退。冻结 Qwen 生成 226 个回答，推理阶段耗时 6.5 分钟。外部 3D 结果采用交集协议，Talk2Car 的官方排行榜按其二维任务计分。 [配对结果](results/talk2car_pairs.csv) · [外部核验](results/talk2car_evaluation.json) · [官方二维评价定义](https://github.com/talk2car/Talk2Car#evaluation)

锁定解析器获得 **58 / 113** 个可用三维框。几何读出有 **6** 条进入双姿态分支，只有 **5** 个框收缩。外部 mIoU 相对冻结版变化 **-0.08 pp**。顶缩作用取决于前端能否提供有效的三维框和物理支撑。9 条相机关键帧的三维标注转换与 nuScenes 原标注逐条比对。 [逐级覆盖](results/talk2car_frontend_diagnostics.json) · [标注坐标核验](results/talk2car_evaluation.json)

## 复验

以下命令在匹配的研究环境中重放实验。目录和权重散列见协议，冻结 Qwen checkpoint 与上一轮相同。模型生成、几何候选和评分各有单独封存。发布数据只包含匿名数字。新增源训练、目标端更新和 FINAL 访问均为 0。

```bash
study=studies/3eed_height_only_transfer_20261123
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for cohort in validation extension; do
  crossplatform/env/bin/python "$study/anchor_control.py" infer "$cohort"
  crossplatform/env/bin/python "$study/anchor_control.py" evaluate "$cohort"
  crossplatform/env/bin/python "$study/ground_attribution.py" seal "$cohort"
  crossplatform/env/bin/python "$study/ground_attribution.py" evaluate "$cohort"
done
python "$study/fetch_assets.py" wilddet3d "$WILDDET_WEIGHTS_DIRECTORY"
python "$study/lingbot_config.py" "$study/results/lingbot_config.json"
python "$study/wilddet_eval.py" infer --repository "$WILDDET_CHECKOUT" --weights "$WILDDET_WEIGHTS_FILE"
crossplatform/env/bin/python "$study/wilddet_eval.py" generate
crossplatform/env/bin/python "$study/wilddet_eval.py" evaluate
python "$study/fetch_assets.py" nuscenes "$NUSCENES_CACHE"
python "$study/fetch_assets.py" talk2car "$TALK2CAR_COMMANDS"
crossplatform/env/bin/python "$study/nuscenes_data.py" "$NUSCENES_ARCHIVE" "$TALK2CAR_COMMANDS"
python "$study/nuscenes_eval.py" infer --model "$QWEN_MODEL_DIR"
crossplatform/env/bin/python "$study/nuscenes_eval.py" generate
crossplatform/env/bin/python "$study/nuscenes_eval.py" evaluate --archive "$NUSCENES_ARCHIVE" --commands "$TALK2CAR_COMMANDS"
crossplatform/env/bin/python "$study/external_diagnostics.py"
crossplatform/env/bin/python "$study/report.py"
```
