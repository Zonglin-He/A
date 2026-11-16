# 原论文与实际迁移边界

- 3EED 官方 `src/grounding_evaluator.py` 与 `models/losses.py` 区分原生 soft-token 分类头（BBS）和 query–token 对比匹配（BBF）。本轮使用 checkpoint 已有两个分支，不新增专家。分类头的未匹配 query 监督在第255类，不能当作外部前景检测真值。当前主体跨度由既有纯文本程序产生，没有使用评估标注跨度。
  - https://github.com/worldbench/3EED/blob/main/src/grounding_evaluator.py
  - https://github.com/worldbench/3EED/blob/main/models/losses.py
- MDETR（ICCV 2021），2.2.2 和 B.4.1，分别采用 token 位置分类与对比表征对齐。其监督训练消融支持两类目标互补；不证明两个冻结头的分数平均或目标域统计适应一定改善3EED。论文中其它数据集的 no-object 读出也不自动适用于这里。
  - https://arxiv.org/abs/2104.12763
- TTN（ICLR 2023）在源域后训练阶段学习逐通道 source/target 统计插值权重，然后测试时固定该权重。其深层归一化对 source 统计的依赖，是不能无条件以测试批统计替换的反例。当前协议不允许新增源域准备，因此 R110 没有复现 TTN，也没有训练插值权重。
  - https://openreview.net/forum?id=EQfeudmWLQ
  - https://sites.google.com/view/ttn-test-time-normalization/home

R110 的机制仅是：相同当前A输入上，原source表示O与已有source-prepared表示P的逐通道矩提供可计算的接口变换，将原生分类头的两个BN运行统计作 epsilon 一致的代数输运。O不是正确身份标签，原始模型也会错；相同均值方差不保证候选语义对齐。它是待验证的 buffer-only 适应，不是伪标签拟合、TTN复现或共享三模态语义空间的证明。
