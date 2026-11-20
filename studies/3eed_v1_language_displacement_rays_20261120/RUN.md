# 复现语言位移的源域检验

在包含三个单源 epoch100 权重、3EED 编译算子、RoBERTa、既有角色解析器、R180 检索器、源域标定数据及 K16 缓存的工作区根目录执行下列命令。公开记录提供算法、协议与匿名结果，本地私有资产提供完整模型重放的输入。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
study=studies/3eed_v1_language_displacement_rays_20261120
crossplatform/env/bin/python "$study/capture.py" source
crossplatform/env/bin/python "$study/score.py" source
crossplatform/env/bin/python "$study/verify.py"
```

采集进程先保存协议、代码摘要与抽样清单。每个源的原检查点按登记摘要校验，首段前向保存物理 seed 索引、全句轨迹、原始 logits 和 K16。类别前向复用点云骨干输出，固定相同 seed，执行原生文字编码、跨模态编码、提议头和六个 decoder 回归头。全句同提示重放覆盖最初 32 条。各平台开始和结束时对原生模型及缓冲区计算指纹。所有数组、坐标、文本和标注均留在本地。

`score.py` 逐一校验全部预测摘要后读取真值。主体 GT 的中心、尺寸与候选采用同一模型坐标系。阈值采用严格大于 0.25、0.50 的有向三维 IoU。事后 Hungarian 指认使用原生末层 matcher 与本次前向 token 序列中对齐到类名同义词的正位，结果只用于检查位移权重。它描述本次前向，训练时每个 epoch 的历史匹配未保存。

`verify.py` 用显式投影独立复算 192 条的四种几何能量与赢家，并用历史逐候选 IoU 重建全部命中、救回和误改。普通面板的预选规则见 `PROTOCOL.json`。条件入口 `capture.py ordinary` 会检查源域封存和进入下一阶段的裁决。本次实验止于源域。

公开逐例表的每行是一条方向记录，行序已匿名打乱。空缺的方向诊断表示该行回退或缺少相应的对比候选。回退的真实任务 IoU 在每个方法列中完整保留。百分比用于任务成绩，`source_mechanism.csv` 的成对排序胜率在 0 到 1 之间，中心距离以米计。重新生成预测需使用独立的输出工作区；已有完整封存可直接运行复核脚本。
