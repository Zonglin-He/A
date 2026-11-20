# 复现 TCR 源域首次检验

在已有三份单源权重、精简 v1 的工作区、R180/R181 及语言位移实验完整私有缓存的环境中，从项目根目录运行以下命令。原句、点云、原始帧标识、框坐标、query 编号和权重保存在本地。

~~~bash
study=studies/3eed_v1_terminal_conditioned_replay_20261120
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
crossplatform/env/bin/python "$study/probe.py"
crossplatform/env/bin/python "$study/assess_probe.py"
crossplatform/env/bin/python "$study/verify.py"
~~~

第一个入口核对原数据清单与 K16、记录各次前向并封存；第二个入口在封存后重用两份历史有向 IoU 记录评分；第三个入口从封存分数独立计算赢家、基于秩的成对胜率、翻转上界和既有案例的变化，另检验线性头零权重复现与损失解析梯度。全部 192 条都保留在任务指标分母。

六层原轨迹重放使用每层捕获的 query 位置及原始 memory；初始几何是提议头的首次框预测，单层版本从原第五层的输出开始。每次回读都对全部 256 个 query 计算 self-attention，随后按原 K16 的 native 索引取出表示。用于实际匹配的标量为既有 noun mask 的原生 log-odds，损失与源域门槛见协议。
