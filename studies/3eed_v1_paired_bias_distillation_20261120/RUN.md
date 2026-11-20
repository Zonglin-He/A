# 复现

先运行同日期的预截断物理候选配对实验，保留源域特征、无标签目标单句特征及其封存回执。从项目根目录运行下列命令。

```bash
study=studies/3eed_v1_paired_bias_distillation_20261120
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
crossplatform/env/bin/python "$study/bias.py"
crossplatform/env/bin/python "$study/fresh_source.py"
crossplatform/env/bin/python "$study/target_bias.py"
crossplatform/env/bin/python "$study/verify_bias.py"
```

单句推理入口为 `bias.predict(h,j,model)` 的 `paired_bias` 输出，输入是精简 v1 第二轮四个主体槽进入最后线性层之前的 96 维激活和四个最终匹配分。取输出的最大分数对应的原框。出现第二句话的地方是源域拟合和独立机制检验。

新源域机制面板从已有普通 Waymo VAL 清单里选择两句均在清单内、同类异实例的自然描述。在模型得分产生前按录制、物理帧和固定哈希确定名单，保留每段录制至多两帧。读出封存后才将预测与实例身份对应。目标端沿用两个原始迁移方向和全部 5,528 条单句。
