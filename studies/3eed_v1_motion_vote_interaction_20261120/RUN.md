# 重放方式

从能运行精简 v1 的研究工作区根目录执行。需要 Waymo 单源权重、完整普通数据、已登记源域同帧双句及前两轮的私有封存。同帧的点云与 256 个 query 种子坐标逐值对齐。源域只计算一组系数，目标端读取每条输入自身的原句与其解析出的主体类别。

~~~bash
study=studies/3eed_v1_motion_vote_interaction_20261120
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
crossplatform/env/bin/python "$study/source_motion.py"
crossplatform/env/bin/python "$study/verify.py" source
crossplatform/env/bin/python "$study/target_motion.py" extract
crossplatform/env/bin/python "$study/target_motion.py" score
crossplatform/env/bin/python "$study/verify.py" complete
~~~

源域系数在首次拟合后封存。目标输入重放完成后，封存候选赢家，再复用前轮两种独立的有向三维 IoU 实现评分。公开录制号、方向记录号和候选选择等价类均为匿名编码；原句、坐标、检测框、真实 query 索引和模型权重仅保存在私有缓存。
