# 复现

从包含精简 v1 权重、已登记普通清单和原始数据的项目根目录运行。源域和目标端的特征提取互不依赖。目标端特征提取逐块核对历史 v1 的全部几何和选择，可从最后一个有回执的块继续。

```bash
study=studies/3eed_v1_pretruncation_physical_duels_20261120
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
crossplatform/env/bin/python "$study/source_features.py"
crossplatform/env/bin/python "$study/fit_duels.py"
crossplatform/env/bin/python "$study/target_features.py"
crossplatform/env/bin/python "$study/transfer.py"
crossplatform/env/bin/python "$study/verify_transfer.py"
```

每个跨平台样本使用一条自然描述。监督配对仅发生在源域。源域预截断面板用源域真值匹配原有提议，测量两端都在场时的排序能力。双句提名面板只用两句各自的提名构造公共四槽。两项诊断均列出所有潜在对的分母。目标端保持单句 v1 四槽。

头部封存、基于源域普通单句的换框率校准、目标预测封存与事后评分按此顺序执行。私有缓存包含权重、特征、框和原始数据索引。公开结果的记录标记均为匿名标记。
