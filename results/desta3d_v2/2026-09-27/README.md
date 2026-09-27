# DESTA-3D v2：实现修复与已完成诊断

本页记录截至 2026-09-27 已完成、独立回读的源开发实验。完整源训练仍在进行，目标 v2 TTA 未启动。四份 JSON 为原始审计报告的汇总导出，不含视频、标注、身份清单、逐样本预测或权重。

## 这轮实现了什么

- **共享 reference/time、独立 KV：** event pass 生成实际 token；spatial 用自己的 visual residual 重新 prefill，沿官方 semantic/time/box 缓存顺序强制继承 event tokens。没有复用 event KV。
- **真实零残差等价：** 首次 whole-prefix prefill 的 4/4 真实样本框不等于 Frozen，失败保留；中间一次导入时序失败；最终 cached revision 的 4/4 reference、时间和框与原 PTD 精确一致。
- **具体 TTA 目标：** normalized reader view-consistency、referent/event Bernoulli KL、mean parameter anchor。FiLM/LN 共 66,816 参数，两个 residual gate 冻结；alignment=0、joint=0。
- **精确续跑与每轮权重：** 原训练优化器/RNG/游标保持；从 E1 起在完整 epoch 训练边界保存不可变四臂 adapter 权重。E0 权重此前已覆盖，不能伪称恢复；E0 预测保留。
- **选态澄清：** 四臂共同 eligible epochs 1–5。current-recipe 的 E0 虽为 B 阶段也排除，不重排历史状态。

代码入口见仓库 [README](../../../README.md)，新增协议见 [reference contract](../../../protocols/desta3d_v2_reference_contract_v1.md)。共享缓存实现与原独立-reference 源训练分别保留，源训练没有热改科学配置。

## 完整共享 reference 对照

冻结官方 PTD-4B；clean 输入；198 条 VidSTG 源验证查询、31 个父视频。固定 dual_repaired E1 cursor24 checkpoint：155 个 evidence 优化步、6 个 integration 优化步，非分数选态。两臂共 396 预测及 198 诊断全部封存后才读源验证标签；无优化、无目标标签。

| 读出 | 父源宏 vIoU | 父源宏 sIoU | 父源宏 tIoU |
|---|---:|---:|---:|
| 匹配 Frozen PTD | 40.1736% | 56.6043% | 51.2066% |
| 独立 reference | 40.0385% | 56.5304% | 51.0397% |
| 共享 reference＋time | 40.0385% | 56.5304% | 51.0397% |

所有新旧框和轨迹相同，差值为 0。197 个有效 reference 对均一致；另一个 event/time 格式失败，仍保留在全部 198 查询的任务指标中。最初保存列表比较的 0/198 包含一个空列表对，科学解释已修正为 0/197 有效 reference 对。

两臂相对 Frozen 的 ΔvIoU 为 **−0.13507 pp，95% 配对父源 bootstrap CI [−0.34488,+0.02463] pp**（10,000 次，seed 20260927）。31 父源中 14 正、17 负，0 个损害超过 5 pp，最差 −2.5027 pp。原 vIoU>0.5 的 68/68、tIoU>0.5 的 83/83 查询保留该阈值。5 条无空间 GT 支持的查询保留，空间指标按零计。

独立 event head 的父源宏 AUROC 为 0.5394，仅 28/31 父源可定义；173/198 查询可定义。它不能代替完整轨迹评价。固定相同 reference/time，空间 residual 有/无的 197 个可测查询均改变坐标 logits，平均 KL(with||zero)=0.00218204。这说明注入存在，不说明注入有益。

结论范围：共享 reference 合同已经修好，但所测早期 checkpoint 原两路没有 reference 分裂，因此没有新增轨迹收益。该状态没有完成一整轮 B，不能据此裁决最终源训练或目标 TTA。

完整机器可读汇总：[shared_reference_summary.json](shared_reference_summary.json)。

## 源训练第一轮 E0

同 198 查询 / 31 父源，独立几何复算。三种 repaired 臂 E0 仅 evidence 训练，current 臂 E0 是 integration；不能将其当作同阶段、单因素 recipe 比较。

| 方法 | E0 阶段 | 相对 Frozen 的父源宏 ΔvIoU (pp) | 95% 配对 CI (pp) |
|---|---|---:|---|
| early_repaired | A | −0.1526 | [−0.3835,+0.0876] |
| shared_repaired | A | −0.1135 | [−0.3296,+0.0976] |
| dual_repaired | A | +0.1094 | [−0.2489,+0.5573] |
| dual_current | B | −0.0481 | [−0.2812,+0.1775] |

均未出现父源 >5 pp 损害。E0 不属于共同选态窗口；A 阶段的小正点估计不是源门通过或方法成功。见 [source_E0_summary.json](source_E0_summary.json)。

## 校准梯度与监督尺度

真实单个源训练 query、clean teacher / gamma 0.9 student、相同物理帧，无 GT；临时执行一步 AdamW（LR 1e-5，wd0，clip1）后精确复位。FiLM/LN 实际非零梯度，冻结 backbone 无梯度，两个 gate 不动。40,040 坐标 logits 中 31,255 改变；同 prefix 的 KL(before||after)=0.00252999，argmax 框未改变。因此是连通性检查，不是任务收益。见 [calibration_connectivity.json](calibration_connectivity.json)。

另一个固定 8 个源训练父视频的小面板，使用合法源监督，只算梯度、不优化。同一组 19,968 维 shared stem 参数上：

\[
\frac{\|\nabla[0.1(L_{ref}+L_{event})]\|}{\|\nabla[L_{CE,event}+L_{CE,spatial}]\|}
=64.24\sim1123.21,\quad \mathrm{median}=527.60.
\]

task/aux 余弦约 −0.048 到 +0.144，多数接近正交。第一个已观测样本促成后续七例，因此是开发诊断，不是独立确认；它只描述该早期 checkpoint 的 shared stem，不能推广为所有参数或性能损失的因果证明。此结果尚未触发 PCGrad、损失重权或活跃训练配置修改。见 [gradient_scale_summary.json](gradient_scale_summary.json)。

## 下一阶段与复现边界

继续已锁定四臂源训练并逐轮审计完整 tube、event 读出、原好保持及负尾。最终选中的完整 B checkpoint 还需做共享 reference 对照。达到用户设定的源验证资源门后，再登记目标 8 源接口与 64 源扩展；应分别报告 Frozen、同 source-fit 无更新和 TTA，并对照 view-only 与 +alignment。v1 source feature statistics 不可移用。

本公开快照不包含运行上述数据实验所需的权重、源/目标数据、local manifests 或封存张量。JSON 保留原审计报告哈希用于身份追踪；没有这些私有输入，不能仅凭汇总从头重算全部指标。原始正负结果和失败记录保留在本地档案。
