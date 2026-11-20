# 复现原生 seed 监督编码读出

使用含已登记 Waymo、Drone、Quad 单源权重、原始点云、源域真值、已编译的 3EED 算子、原生训练仓库以及前序封存 K16 的工作区。精确哈希列在 [PROTOCOL.json](PROTOCOL.json)。[上游 K16 和同批源域清单的准备过程](../3eed_v1_language_displacement_rays_20261120/RUN.md)

在项目根目录运行以下命令。第一次预测写入独立目录并封存。已有封存时可从核验步骤运行。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
study=studies/3eed_v1_native_seed_code_decoding_20261120
crossplatform/env/bin/python "$study/predict.py"
crossplatform/env/bin/python "$study/audit_native.py"
crossplatform/env/bin/python "$study/score.py"
crossplatform/env/bin/python "$study/verify.py"
```

- `seed_code.py` 实现候选编码、全负参考、两种聚合、同帧保持编码大小与重复数的错配及冻结平局读出。原有模型无需修改。
- `predict.py` 校验源模型、清单和 K16 的封存摘要，依次重放三份冻结模型的完整描述前向，采集 1,024 个输出和两套点索引。预测文件保存在本地。
- `audit_native.py` 在预测封存之后调用原训练标签构造和 Focal 损失。它将 192 个 GT、3,072 个候选分别当作单目标样本，验证每一点的正负标签，以两个有向框评分器复算 IoU 并核对历史值。
- `score.py` 汇总全部 14 个读出和全体匿名记录。严格超过 0.25、0.50 才算命中。排名表中的 `macro_pair_win` 在每条同时含好坏框的记录内计胜率，平局计半，再对这些记录等权。每条记录的最佳好坏分差为正表示好框领先。逐例表里留空表示缺少其中一类候选。
- `verify.py` 用全部 1,024 项正负标签损失独立重算赢家、指标和配对变化，并核对官方 git 版本的损失、成员判断及数据加载器的目标函数。

公开结果可直接复算。将 `results/source_all_rows.csv.gz` 中各方法的 `_iou` 列分别与两项严格阈值比较即可得到命中数。逐行与 `frozen_joint_iou` 比较可得救回与误改。编号为打乱后的匿名行号；原始采样、传感器坐标和权重只保存在本地。
