# 截至2026-09-28的完成结果

所有下列实验已完成、先封存后评分并独立复算。这里是公开汇总；逐样本原件、checkpoint、输入和raw梯度留在本地。历史报告保留，恢复事故见[更正](../../../docs/DESTA3D_V2_CORRECTIONS.md)。当前没有活跃GPU阶段；下一候选仅草案。

## 修复后的固定B源对照

618 train query/95 Vid父源；198 val query/31父源。每个B固定155实际Adam步、末窗2，A不重训。表内vIoU是父源宏均值，差值与95%配对父源bootstrap CI单位pp。

| 状态 | vIoU % | 对Frozen Δpp [95% CI] |
|---|---:|---|
| common | 40.283007 | +0.109409 [-0.248892, +0.557256] |
| B0 | 40.010384 | -0.163214 [-0.339765, -0.021365] |
| B1 | 40.182395 | +0.008797 [-0.368621, +0.479477] |
| B2 | 40.048269 | -0.125329 [-0.348332, +0.093260] |
| Frozen | 40.173598 | — |

B1仅是固定B候选中均值最高且达到literal源均值门；CI跨0，common A均值更高。不能称稳定收益或生产晋升。三臂降低aux dominance的假设优先级下降。完整比较、native-good和负尾见[source_fixed_B.json](source_fixed_B.json)。

## 同8父源/24条件例目标开发对照

HC4/Vid4，每父源1query。三个条件clean/noise_medium/defocus_extreme。corruption先在同父源内平均noise+blur，再父源等权。所有更新固定3步；下表均为对B1 noTTA的Δv(pp)。

| TTA臂 | clean | noise | blur | corruption | corruption 95% CI |
|---|---:|---:|---:|---:|---|
| convolution_view | -0.201010 | -0.065518 | +0.165674 | +0.050078 | [-0.036085, +0.125088] |
| calibration_view | -0.040383 | -0.122099 | +0.150793 | +0.014347 | [-0.076736, +0.100624] |
| convolution_alignment | -0.185743 | -0.093857 | +0.044341 | -0.024758 | [-0.105106, +0.056418] |
| calibration_alignment | -0.366592 | +0.284751 | +0.017851 | +0.151301 | [+0.034018, +0.283803] |
| calibration_alignment_output_anchor | +0.015089 | +0.055788 | +0.098874 | +0.077331 | [-0.007706, +0.189161] |
| calibration_alignment_temporal_anchor | +0.216824 | -0.150346 | +0.026012 | -0.062167 | [-0.334171, +0.120371] |
| calibration_alignment_identity_view | -0.020957 | -0.026756 | +0.068008 | +0.020626 | [-0.040562, +0.082362] |

原calib-align在noise有效应、clean受损；双输出anchor改善clean但削弱noise收益；time-only并未恢复noise；identity减clean损害也丢noise。描述性小panel不能支持稳定全正，多个对Frozen主CI仍跨0。既有GT多次开发曝光，CI未多重比较校正，不是held-out成功。

全部三指标、HC/Vid分域、每条件比较、native-good空间帧保持和>5pp负尾均保留在[target8_all_completed_arms.json](target8_all_completed_arms.json)。存储IoU/contrast单位是fraction，表内乘100；事件head AUROC不是native temporal endpoint准确率。

## 源16监督正控及FP32最终head对照

固定16训练父源各1query，同B1、66816FiLM/LN、freshAdamW1e-5/wd0clip1、固定3步。监督明确用源GT，不是目标TTA。只将新臂task CE最后投影FP32，body/native仍BF16。

| 状态 | vIoU % | sIoU % | tIoU % |
|---|---:|---:|---:|
| no_update | 41.113359 | 61.929737 | 54.003027 |
| unlabeled | 40.929489 | 60.712075 | 53.918525 |
| supervised | 41.144575 | 62.190582 | 53.972977 |
| supervised_fp32 | 37.013729 | 57.770907 | 48.029211 |

新FP32监督对B1 Δv=-4.099629pp，CI[-10.937813, -0.009124]；2正/8负/6零，2个父源损害超过5pp。v>.5保持6/7，t>.5保持7/8。旧BF16监督仅小趋势且CI跨0。

FP32任务梯度与实际Adamdelta的48个一阶量均为下降方向，最终同定义FP32 CE仍只8/16下降；即使CE下降也出现native损害。两处大负尾和两处正向案例全部计入。单seed/分层训练panel/未校正CI，不推普遍FP32无效或双支路线失败。详见[source16_task_controls.json](source16_task_controls.json)。

## 诊断链的已知事实和限制

- 原optimizer整数key转str的连续恢复声明已撤回，修复后完整对照独立完成。旧raw及no-update仍保留。
- 完整B1共享reference两路径198query几何全等；真实logit连通性未产生新tube增益。
- source4无更新梯度诊断显示mild view与alignment方向有竞争；替换source std估计量后total cosine约.998，未据此开新target。
- 单源cast、输入VJP、完整词表head投影逐级核验揭示真实离散数值变化。固定hidden的最后输出舍入可翻转空间CE差的符号；这不是官方BF16 bug或全部任务根因。FP32训练实际负结果限制该补救解释。
- GT-prefix/native支持CPU审计覆盖固定16源既有输出：48语义reference均与GT相同，但时间/框支持与GT完全相同分别仅4/16 B1、4/16旧监督、3/16新监督。NTP/MTP不等于time/coordinate类别；token比例不是梯度贡献。详见[source16_prefix_support.json](source16_prefix_support.json)。
- 原no-output无标签loss在adapter内部、不调用lm_head，FP32 task-head helper不能直接改名为无标签修复。

下一native-endpoint源正控只proposed，未注册/执行。当前将代码交给外部审阅，没有自动新target、64源扩展、生产晋升或参数网格。

## 来源与可验证范围

四份JSON为原已审计报告的聚合提取，包含source evidence相对路径及SHA-256；省略逐样本标识和明细而保留均值、CI、正负数、极端差与保持率。公开代码允许审查算法/作用域及CPU合同；没有本地raw就不能重新验证全部历史实测。
