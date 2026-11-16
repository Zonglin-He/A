# 本轮机制依据与不可迁移条件

R112是现有source-prepared系统的受控组件撤回，不是新TTA损失，也不声称复现以下方法。3EED原生实现和R76/R91实际计算图决定实验位置；R111投影参数逐值相同、native同信息路径更好，是本轮竞争解释的本地证据。

- xMUDA（CVPR2020）在RGB与点云语义分割中将跨模态模仿与主任务预测分开，以减少较弱模态错误污染较强模态。这支持“跨模态作用并非无条件有益”的问题意识；原方法仍包含有标签source训练、独立模态语义预测，不能直接作为当前grounding辅助损失的有效性依据。
  - https://arxiv.org/abs/1911.12676
  - https://github.com/valeoai/xmuda
- MM-TTA（CVPR2022）结合模态内部快慢预测与跨模态可靠性选择，为自训练提供监督。当前3EED/R91没有两个独立且已经训练好的RGB/点云语义分割输出，不能把两个残差门直接当作该方法的模态可靠性估计，也不能由本轮门值或范数断定谁可靠。
  - https://arxiv.org/abs/2204.12667
- 3EED原生cross-encoder、decoder和匹配投影提供固定几何与语言路径。本轮保留原A256及其六层位置，只有新增point_adapter残差与早期image_to_point交换的受控静态改写；反向交换、图像编码、lateRGB和实例竞争模块仍保留。
  - https://github.com/worldbench/3EED/blob/main/models/bdetr.py

当前实际核对范围：xMUDA/MM-TTA的作者arXiv摘要与xMUDA官方仓库说明；本轮没有将其称为全文复现，也没有新训练其网络。CVF页面本次403，改用作者arXiv主来源。撤回效果只能由全体任务指标裁决，不能由这些论文的成功或当前隐藏表示范数推出。
