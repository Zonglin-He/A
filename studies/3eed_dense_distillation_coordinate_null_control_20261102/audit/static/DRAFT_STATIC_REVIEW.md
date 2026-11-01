# R43 训练草案独立静态审查

裁决：**修订后草案符合所登记的单因素与 null 对照口径；这是可供 root 考虑授权的提案，不是已执行算法、训练许可或实验质量 PASS。** 本审查不执行损失函数、模型、优化器或任务评分，也不读取 FINAL。

审查文件为 `review/dense43_draft.py`、`EXPERIMENT_DRAFT.md`、`DRAFT_STATUS.json`、`PROPOSED_PLAN.json`，并对照原 `crossplatform/round24/fit24.py`、`memory24.py` 与 R30 更新范围。代码只有损失代数函数定义；没有文件读取、模型或 optimizer 入口。main 直接拒绝执行训练。

实际函数将 teacher KL 的 student 输入在 native probability 和 pretoken token-log-odds 之间切换；teacher F 和 source probability 都 detach，候选温度始终 0.25。源 KL 两臂始终为 `KL(softmax(s0/.25)||softmax(sθ/.25))`、权重 0.2；hinge 两臂都使用 native sθ，原 0.05 margin 和权重 0.5；参数锚定作为原逐张量均方变化之和传入。没有把源保持、间隔或最终读出一起换成 z，也没有加入 R30 的 keep 或 22 条偏好。最终 native 读出写在提案中；未来实际 runner 仍须逐项审计，当前函数本身不实现训练或读出。

full/null 两种 teacher 在 probability/z 两臂分别固定相同。null teacher 是 source native probability，不能换成 source z 来消去前提诊断已揭示的初始化失配。full/null 改变 teacher target 的 argmax 是这项信息干预的预期组成，不是额外使用 GT。null 仍保留原 source-top hinge，因此 probability-null 不可拿 frozen 代替。

更新位置固定为原查询投影 MLP，184960 参数；拟用原 source 独立初始化、每平台四状态、每状态 16 步，合计 128 optimizer steps。此预算在提案中明确未获授权。原 full-probability control 必须先与 R24 `query_mlp_lr1_s16` 的状态和原 A/V64 query 复现，才讨论与 z 的效应；未来若状态不完全相同，应先按可核验数值合同裁定复现/测量问题，不能直接把差异认作方法增益。

原 `memory24.py` 保存 `teacherF-source_scores`，null 置零时其值为零；在原 finite-source/finite-key 合同下零修正严格返回 source。修订稿已区分：旧 G direct/memory 是 full-G 的同信息对照，null 以 source/零增量 memory 为同信息控制；把 full-G memory 与 null 列在一起只能作为额外信息的系统参照。两种参数臂仍在 native 概率单位读出，没有把旧概率单位缓存增量加到 z。

独立审查提出并已落实两处修订：

1. 删除从“teacher 越过表达界”推断存在正确可迁移任务信息的表述。错误 G 同样可以越界；当前只支持表达前提及校准竞争解释。
2. 明确 full/null 信息预算各自的直接与缓存对照，避免把 full-G 对手误标成 null 的同信息基线。

所列文献用于区分原生 logits、温度、梯度相对尺度和奖励监督，不把该自有 dense KL 提案命名为 Hinton 蒸馏或 RLCF 的完整复现。保持 0.25 和原损失权重是为了隔离当前因素；不擅自加 T²、扫温度或扩层。

后续若获授权，需另锁实际 runner/hash/输入/预算，预测先 seal 再独立评分，保留全部 208 行、256 候选、109 零增量观测、所有负值及全部任务得失。V64 已暴露，仅开发诊断；不读取 FINAL，不因前提结果改反馈接收或按 GT 删除样本。当前没有新增训练结果，也没有采用该方法。
