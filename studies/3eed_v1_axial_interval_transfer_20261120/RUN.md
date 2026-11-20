# 运行与复核

在有普通数据和已登记缓存的本地研究工作区，研究目录应位于 `studies/3eed_v1_axial_interval_transfer_20261120`。入口使用现有 3EED Python 环境及 NumPy、SciPy。训练和推理都可在 CPU 完成。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
study=studies/3eed_v1_axial_interval_transfer_20261120
crossplatform/env/bin/python "$study/fit.py"
crossplatform/env/bin/python "$study/predict.py"
crossplatform/env/bin/python "$study/score.py"
crossplatform/env/bin/python "$study/verify.py"
```

源端需要历史源域实例实验的自然样本清单与冻结特征，以及固定实例几何实验中的源域原位 IoU。目标端需要原始普通点云、R183 普通清单与已封存的 v1 赢家。评分另需普通目标标注和既有 K16 逐候选 IoU。各依赖的位置由 `axial.py` 给出，输入摘要见 `PROTOCOL.json`。独立复核还读取原始源点云以重建缓存。公开的代码和标量结果可以用于审查；完整重跑需要这些本地数据及父实验缓存。

从空的研究输出目录运行会依序生成协议锁、12 个训练读出的封存记录、六方向的无标签预测封存和评分封存。已存在完整封存时，各生成阶段会停止以保护实测结果。对已完成的研究可直接执行 `verify.py`。冻结预测使用原 v1 输出，按源模型各自选取一个已训练读出。每个源的 4 个头都训练 1,024 次全批次梯度，各步均读取 3,840 个样本－起点－反射组合。训练期间采用固定解析平滑上界及相同每系数 L2，包含偏置。

在本次单线程运行中，源端取点、训练和留出评分合计约 14.7 秒，冻结目标预测约 8.4 秒，全部目标候选的两种 IoU 评分约 17.1 秒。以上计时覆盖对应脚本内部阶段，不含历史缓存生成、启动、独立复核和发布耗时。

`results/directions.csv` 使用六方向等权任务指标，`paired.csv` 给出所有预设配对。成绩单位是百分数，差值是百分点。救回表示基线 IoU 不超过阈值且该臂超过阈值，误改表示相反的穿越。`results/all_rows.csv.gz` 提供全部匿名方向记录，列有每臂 IoU、是否移位及 K16 外机会标记；行序与原样本顺序无关。`verify.py` 在封存后计算额外的位置错因分解。
