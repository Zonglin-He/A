# 复现

使用同日期两个配对研究的源域特征与已封存的录制面板。运行入口为下面的命令。

```bash
study=studies/3eed_v1_nomination_intervention_20261120
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
crossplatform/env/bin/python "$study/intervene.py"
```

每条句子在同一份冻结骨干输出上批量执行原序和六次主体槽互换。物理框与自身的先验共同移动，输出按框还原。原序逐分与前一轮的单次解码核对。源域拟合只经过 Waymo 训练录制，系数封存后才评价留出录制和已登记的 142 对描述。
