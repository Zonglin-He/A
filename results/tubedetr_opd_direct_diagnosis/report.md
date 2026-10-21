# TubeDETR OPD / Direct 完整128封存后诊断

同奖励比较是主对照；固定预算Direct尚未独立公平调参，128及原目标全量已曝光。
所有差值为OPD减对应Direct，单位pp；置信区间按父视频配对bootstrap，仅开发诊断。

| 比较 | m_vIoU差值 [95%CI] | dense sIoU差值 | 未观测sIoU差值 |
|---|---:|---:|---:|
| Layout3OPD-Layout3Direct | -0.03132 [-0.1711979225552683, 0.1298160662282634] | -0.69677 | -0.627016711250599 |
| GeometryOPD-GeometryDirect | 0.06571 [-0.3753243852514624, 0.4633724162997479] | -0.38098 | -0.42568898643097675 |
| Layout3OPD-GeometryOPD | -0.03843 [-0.448034012434044, 0.41689244741772835] | 0.07293 | 0.10295199451195594 |
| Layout3OPD-Layout1OPD | 0.14180 [0.0104706306143266, 0.2927243227315963] | -0.40059 | -0.3773152096450892 |

后续机制研究触发：{"required": true, "reasons": ["m_vIoU improvement below pre-outcome practical triage threshold", "paired parent bootstrap interval does not exclude zero", "no positive m_sIoU difference", "no positive unobserved_actual_sIoU difference"], "no_equivalence_or_independent_confirmation_claim": true}
0.5pp为结果前登记的工程分流阈值，不是等效界限、临床式显著性标准或128统计功效保证。
目标尺寸/运动/事件占比的中位数分组仅描述；无GT观测不等于遮挡。结果不回流当前试验。
教师质量、候选覆盖、排序、初始熵、梯度及局部到未观测传播须结合分析，不能仅凭伪损失下降归因。
后续主线程须先以本证据提出可证伪假设，再检索原论文/官方代码、冻结有限验证协议；禁止只为胜Direct而削弱其对照。
