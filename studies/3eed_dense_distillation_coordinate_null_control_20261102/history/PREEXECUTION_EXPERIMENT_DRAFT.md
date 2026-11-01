# R43 后续候选：同旧 G 稠密分布，仅改变 teacher KL 的学生坐标

状态：**只完成无任务标签前提诊断和损失草案；未训练、未任务评分、未读取 FINAL。** GPU 运行与预算须由 root 另批。

## 前提结果改变了什么判断

实际旧 G 在 94/208 条上的 F32 跨度大于 1，20/208 条 teacher posterior 峰值超过原 native 质量候选 softmax 的理想上限 0.1763517。原概率接口确有表达限制，但尚未证明旧 G 在这些行提供了正确且可迁移的信息；错误 teacher 同样可以越过该范围界。

同时，直接换 z 不是中性的无证据修复。208 条中有 109 条 ΔG 全为零；这些行 pG 与原 source-prob posterior 完全相同，原 teacher KL 在 source 为 0。但换 z 后，109 行的初始 teacher KL 平均为 **22.7983 nats**。全体 null teacher 的 KL(p0-prob||p0-z) 平均为 **25.7866 nats**，而 full G 的原 teacher KL 仅平均 **0.4660 nats**。源概率候选分布熵均 5.2814，源 z 候选分布熵均 1.0902。

因此，新的 z teacher 项会强烈重整形很多没有 G 增量的分布。即使后续任务指标提高，也不能只根据“teacher 没变”就归因于学习了更好的外部监督。

## 最小主因子和原样保留项

固定原 R24 source、D208/A208 顺序与缓存、旧 G 全 256 分数和全部反馈；每平台 source 独立初始化。只更新原查询投影 MLP（288→288→288→64，184960 参数）。冻结文字投影、decoder、backbone 和所有框。AdamW lr=5e−4、weight_decay=0、16 步，只有这一组预算。

teacher 分布始终由原实际 F32 构造：

\[
F=s_0+2v+2p+2r,\quad P_G=\mathrm{softmax}(F/0.25).
\]

主配对只把第一项中的学生候选坐标从 sθ 换成 zθ：

\[
L_c=KL(P_G\|\mathrm{softmax}(c_\theta/0.25))
+0.2 KL(P_0^{prob}\|\mathrm{softmax}(s_\theta/0.25))
+0.5[0.05+\max_{i\ne q_G}s_{\theta i}-s_{\theta q_G}]_+
+R(\theta,\theta_0),
\]

其中 cθ∈{sθ,zθ}，P0prob=softmax(s0/0.25)，qG=argmax F，R 仍为原参数逐张量 mean-square delta 的和。源 KL、hinge 和参数锚定**全部保持原 R24 坐标、权重和实现**，不加入 R30 keep，不用其 22 条局部偏好替换 dense teacher。最终推理始终按 native sθ，避免混入 R41 末端读出因素。

zθ 必须来自同一投影和有效 token 的 LSE(pos)−LSE(complement)，保持温度 0.07、L2 归一化、缓存 mask、特殊 token 和 not-mentioned sentinel。不能对舍入后的 sθ 取 logit。z 的候选分布范围和梯度传递都会改变；这不是仅放大一个数值比例，也不是严格无界模型。

## 必要 nullΔG 控制：只提案，不自行加训

建议在 root 批准时登记完整 2×2：

| teacher 信息 | 概率学生 KL | z 学生 KL |
|---|---|---|
| 完整原旧 G：F=s0+ΔG | 原 R24 配对 control | 主新接口 |
| nullΔG：Fnull=s0 | probability-null | z-null |

两条 null 的 teacher 都固定 P0prob，不能给 z-null 偷换成 P0z。null 中 teacher argmax 跟随 Fnull，源 KL/hinge 公式不变。原 hinge 即使没有 G 也可能在 source 为正（平均 0.02027）；因此 probability-null 也不保证零更新，不能把它省成 frozen。

这会是每平台 4 个独立状态、合计 8 状态×16=128 步；比只跑两主臂增加两个 null 控制，**该增加尚未授权**。不新增外部模型或 G 调用，不筛掉 109 零证据行，不按范围、teacher 熵或最终 IoU 删除反馈。

应报告每个 teacher 内的 z−prob 变化，以及两种坐标内 full−null 的 G 信息增量；差上差只能解释这个固定干预，不能当一般因果保证。若 z-full 与 z-null 同幅改善，收益主要不能归为 G 信息。若只改善拟合而迁移总体不改善，停止该配置；不得靠增步或换温度把失败改写为成功。

## 原 R24 配对 control 必须先复现

使用原 `capture336_attempt001` 的 source 特征和 `projection_source_PRIVATE` 权重、原 D208 列表、原 bank272 的 v/p/r，按原 teacher 构造顺序 cast F32。不要用后来重新捕捉的 h 或被 import hook 覆盖的 feature loader 替代。

full-probability 臂应与原 `query_mlp_lr1_s16` 两平台参数和原 D208/V64 query 重放核对，保留原 FP32/无 TF32、批次 token padding 与固定读取合同。失败先记为复现或测量问题，不按标签挑容差；不能在 control 未复现时把差值归因于 z。

R30 仅作为外部系统参照，不是这个单因子的因果 baseline。原 rank_adapter 无界残差对照及其负结果继续保留；不能说整个 R24 从未尝试无界候选分数。

## 同信息无参数与读出控制

- A 当前有 G：保留 frozen 原 F 直接重排；参数模型若再叠同一 ΔG，必须与它比较，不把额外当前 G 收益算参数适应。
- 未用于拟合的原 V64：保留相同 A 历史信息的原无参数 memory（source query×noun key、每 A 最近候选、四邻居、原温度和已锁 α），保留所有负例与已知数值边界。
- 两条学生最终读出都仍是 native s，故 memory 的原 ΔG 分数仍在相同最终坐标使用。**不能把原概率单位修正加到 z 作为一个暗换的控制**；本项不更换 memory 坐标或检索规则。
- full-G 的 direct G 和 memory 是完整旧 G 的同信息控制；null 臂的同信息无更新控制为 source/零增量 memory（值为零，输出 source）。若表中额外列 full-G memory 对比 null，只能标注它拥有更多信息的系统参照，不能称公平同信息基线。
- source、原 R24 probability、当前 z 候选以及 null 两臂的预测先封存，再独立 CPU 评分总体 mIoU、Acc@25/50、全部得失和场景聚类。原 V64 已暴露，不充当新独立确认；FINAL 不读取。

## 文献核对后的定位

Hinton 等蒸馏使用 softmax 前 logits 和共享温度产生软分布；温度和梯度相对尺度需要同时考虑。当前 R24 是对已经 softmax 汇总的质量再做候选 softmax，和经典 logits 蒸馏不同。其温度 0.25 也不是原论文常用的高温软化。为了原配对可比，这次不偷偷加 T² 系数或改变温度：那会同时改变 teacher 项相对 source KL/hinge 的权重，是另一个因素，而不是本项“缺 trick 已自动修复”。

官方 RLCF 中学生输出是归一化图文特征经模型 logit_scale 放大的相似度 logits，RL 通过采样候选的带奖励 cross-entropy 更新；奖励来自冻结 CLIP 分数并作均值基线等处理。它没有把一个 [0,1] token 质量再当通用类别 logits，也不是这里的固定 dense KL 复现。它支持审计匹配接口和尺度，不保证当前 G 正确或新接口有效。

详见 `SOURCES.json` 与 `dense43_draft.py`。所有外部代码只当资料静态阅读，未下载导入或执行其 main、eval 或脚本命令。
