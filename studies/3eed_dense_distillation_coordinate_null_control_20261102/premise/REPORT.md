# R43 无任务标签前提：旧 G 稠密分布与学生坐标

## 判断

两项前提同时成立：**原 native 质量接口在相当一部分观测上不能精确表达旧 G 的候选分布；但直接把稠密 KL 的学生坐标换成 token-log-odds，也会在大量没有 G 增量的观测上施加强烈分布重整形。** 因而后续若比较两种接口，必须保留 nullΔG 配对控制，不能把任何变化自动归为外部监督被更好地学习。

本轮只完成 CPU、无 GT 的前提诊断。未实例化任何新模型，未训练、未任务评分、未读 FINAL。原 208 条和全部 256 候选都保留，正、负、零 evidence 数值逐值复制和绑定哈希，不按范围、峰值、熵或标签过滤。数字零不被重新命名为语义 UNKNOWN。

## 实际旧 teacher 的范围

原 R24 公式及实际输入 cast 完全复构：

\[
F_{64}=s_0+2v+2p+2r,\quad F=\operatorname{float32}(F_{64}),
\quad P_G=\operatorname{softmax}(F/0.25).
\]

原 query/text 投影路径先输出 native token 质量 s∈[0,1]，再对 s/0.25 作 256 候选 softmax。在理想实数算术中，任意单候选最大概率不超过 **0.1763516676**；完整 teacher 分布可被自由的 [0,1] 候选分数精确表达还要求 F 的跨度不超过 1。这个结论作用于原固定 256-query 目标，不把重复 query 当作独立物体。

| 无标签诊断 | Drone（104） | Quad（104） | 合计（208） |
|---|---:|---:|---:|
| F32 跨度 >1 | 38 | 56 | **94** |
| Teacher 峰值超过 native 理想上限 | 5 | 15 | **20** |
| ΔG 全零 | 62 | 47 | **109** |
| ΔG 为常量 | 62 | 47 | **109** |
| 最大 F32 跨度 | 6.9605 | 9.1697 | 9.1697 |
| 最大 teacher 峰值 | 0.4382 | 0.9445 | 0.9445 |

峰值未超过上限不意味着整个分布可表示，跨度约束还能检出候选间相对概率不匹配。另一方面，共享 MLP 和冻结特征进一步约束可达集合；本诊断没有证明该表达限制是任务指标受限的主要原因，也没有评价 teacher 主体身份或关系证据是否正确。

原 **rank_adapter 不受这个 [0,1] 范围界约束**，因为它返回 source 质量加无界线性残差。其历史负结果仍保留，不能说整个 R24 从未试过更宽的候选分数。

## 初始化与 nullΔG 的竞争解释

以下为 nats，使用同一实际 F32 的 CPU64 稳定分布；CPU32 softmax 同时保存作数值对照，未声称与原 GPU target 逐位相同。

| 全体均值 | Drone | Quad | 两平台合计 |
|---|---:|---:|---:|
| Teacher 熵 | 5.0416 | 4.5955 | 4.8185 |
| Source native-prob 候选 posterior 熵 | 5.3737 | 5.1891 | **5.2814** |
| Source token-z 候选 posterior 熵 | 1.5736 | 0.6068 | **1.0902** |
| KL(teacher‖source-prob posterior) | 0.3547 | 0.5772 | **0.4660** |
| KL(teacher‖source-z posterior) | 18.4794 | 32.4737 | **25.4766** |
| Null KL(source-prob posterior‖source-z posterior) | 18.3966 | 33.1766 | **25.7866** |

109 条零增量行中，teacher 与 source-prob posterior 完全相同，原 teacher KL 为 0；换 z 后这些行的初始 teacher KL 平均 **22.7983**。这不是 teacher 反馈质量的问题，而是源模型原概率坐标与 z 坐标产生的候选分布本来就差很大。

所以“teacher 固定，只改学生 KL”是一个可控因素，却不是对无证据样本中性的修复。若它后续提高任务分数，可能来自 G 信息，也可能来自校准/展平原候选分布，或两者交互。不能在没有 null 对照时只选有利解释。

原 native hinge 也不是无 G 必为零：nullG 选 source 第一名后，固定 0.05 间隔在 source 的平均初始 hinge 为 **0.02027**，因此 probability-null 仍可能更新，不能用 frozen 冒充它的零成本配对。

## 保存与测量范围

完整保存每条的 F64/F32、原 source 概率、v/p/r 正负零数值、ΔG、CPU32/64 teacher 与 source-prob posterior、从缓存 query/tokens 得到的 CPU64 source-z posterior、完整 KL/熵和原始文件 hash。原 source-z 使用实际 token mask、温度 0.07 和全部实际 token 分母，不重新运行投影 MLP或backbone；与原 GPU float32 的 source-z 存在数值实现区别，已明确记录。

零 evidence 不等于显式 UNKNOWN；原 bank 数值和输入 provenance 全保留，没有发明未知标签、把候选补负或修改原 G。所有带阈值的计数仅使用理论表达界，不进入样本接收规则。teacher peak 的 Bernoulli KL 下界是保守的必要限制，不是完整约束族的最优 KL。

语义计划先于代码，代码 hash 先于数组计算。208 条完整数值先封存，再做不含任务标签的总体 reduction。Seal SHA：`cc7fc93aa1c5a84f0c82f43caf80550ef949314aeaa2a668f0078e051787cf6f`。

## 后续草案与未授权内容

`../review/EXPERIMENT_DRAFT.md` 和 `dense43_draft.py` 只提出：同旧 teacher/source/MLP、同 AdamW5e−4×16，只替换 teacher KL 的学生 s→z；source KL、hinge、参数锚定和最终 native 读出保持原 R24。必须先复现原 probability control。

鉴于上述初始化偏差，另提出 coordinate(p/z)×teacher(full/nullΔG) 的最小 2×2，null teacher 在两臂都为 source-prob posterior，不能给 z 臂改成 source-z 从而偷偷换监督。该控制增加训练状态，**尚未执行或获训练授权**。同信息直接 G 和原无参数 memory、全部负例及已暴露开发边界必须保留。

Hinton 蒸馏的原生 logits/shared-temperature 及 RLCF 的归一化、logit_scale、奖励加权 CE 已按原论文/官方源码核对；它们提供接口审计依据，不构成本项目复现或有效性证明。当前不更改温度、添加 T²、扩大接口或新增教师。源代码引用与实现边界见 `../review/SOURCES.json`。
