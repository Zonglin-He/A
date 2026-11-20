# 运行与复核

使用仓库已有的 3EED CUDA 环境、Waymo 权重、源域录制划分和普通 Validation 清单。本次环境为单张 RTX 5090、Python 3.11.16、PyTorch 2.7.0+cu128、NumPy 1.26.4 和 SciPy 1.15.3。在项目根目录运行以下命令。目标端每个输入只需一帧和一条原始自然描述。

```bash
study=studies/3eed_v1_paired_description_subject_readout_20261120
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
crossplatform/env/bin/python "$study/source_pairs.py"
crossplatform/env/bin/python "$study/fit.py"
crossplatform/env/bin/python "$study/diagnose_source.py"
crossplatform/env/bin/python "$study/target.py"
crossplatform/env/bin/python "$study/score.py"
crossplatform/env/bin/python "$study/verify.py"
```

- `PROTOCOL.json` 锁定数据来源、模型指纹、两个损失、步数、评估口径和比较对象。
- `source_pairs.py` 在固定采样上为每个源域配对读取两句原文。每帧使用登记原句产生的四个主体框。换句后，重新计算该句对同一组物理框的重叠投票先验、主体状态和整套参照读出。源域标签仅用来为两端确定监督槽位。遇到候选对多个标注框的有向 IoU 均大于 0.25，该槽保留为身份不清。
- `fit.py` 完整使用可监督对，两个头从同一初始化执行 64 个全批次 Adam 步。写入模型锁之后才评估留出描述对。`diagnose_source.py` 用锁定后的源域原始 A256 缓存追踪两端覆盖在哪个环节丢失。
- `target.py` 每 128 条写入一个带哈希的可恢复预测块。对每条核对冻结 v1 的四个物理框和选择与原普通验证档案一致。每块首条额外在完整 decoder 内直接置换第二轮主体线性投影，比对三组分数及选框。
- `score.py` 在两个方向共同封存后，复算每个物理框的两套有向 IoU。所有阈值使用严格大于。等权汇总时，每个方向占一半。逐例表包含全部输入，`target_switches.csv` 列出任一训练头改变了选框的子集。
- `verify.py` 用 NumPy 手算两个损失的导数及 Adam 更新，复核学习到的 194 个系数，并从公开逐例表独立复算任务指标和正负变化。

公开结果包含匿名记录、统计和代码。复现真实前向需本地原始数据、冻结权重、已注册的普通清单和私有依赖缓存。源域成对通过率是在两句不同的自然描述下分别指对各自的主体，用于衡量同类实例区分；目标定位单独采用 Acc@25、Acc@50 和 mIoU。
