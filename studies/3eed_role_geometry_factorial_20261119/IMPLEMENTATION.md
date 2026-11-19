# 两个原有反馈路径的独立开关

将本研究目录置于项目的 `studies/`，使用上一轮训练、预测和源特征封存。公开代码依赖上一轮研究目录和项目本地资源。

- `factor_decoder.py` 继续调用原模块，单独控制角色成对打分和消息，以及预测框到下一轮点支持的空间偏置。
- `train_factorial.py` 只训练两组单开配置。沿用原训练轮次、随机种子、损失、样本顺序和最后轮权重。
- `replay_factorial.py` 生成两组的新预测。全关和 R158＋PCA 复用旧端点。对全开提取反事实需要的四个参照槽和末轮分数，逐条核对原主体槽、已选框、参照框、采样和选择索引。
- `analyse_factorial.py` 在封存后进行六方向配对评分。全量反事实只改末轮选择，已有目视核验与注释代理各自计数。
- `audit_factorial.py` 独立复算主要端点、四个主体槽和实际参照列干预的读出。
- `render_report.py` 输出配对表、诊断分母和裁决所需数据。

训练和预测沿用项目单卡互斥锁与现有 CUDA Python 环境。数据使用原普通 Validation，FINAL 未包含在本研究的输入。

```bash
python check_switches.py
python train_factorial.py
python replay_factorial.py
python analyse_factorial.py
python audit_factorial.py
python render_report.py
```
