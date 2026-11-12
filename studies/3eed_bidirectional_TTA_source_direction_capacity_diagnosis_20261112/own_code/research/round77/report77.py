"""Render the measured R77 diagnosis from saved audited summaries."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round77';read=lambda p:json.loads(p.read_text());S=read(H/'SOURCE128_SUMMARY.json');M=read(H/'MECHANISM_SUMMARY.json');A=read(H/'source_audit_score_attempt002/AUDIT_RESULT.json');C=read(R/'crossplatform/round77/source128_attempt001/COUNTS.json');T=read(R/'research/round76/score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']
labels={'source':'原始 3EED','bidirectional_frozen':'冻结双向模型','aux1':'原辅助任务，1 步','aux3':'原辅助任务，3 步','aux_raw1':'辅助首步原方向 ×1，无接受筛选','aux_raw10':'辅助首步原方向 ×10，无接受筛选','ground_same_radius1':'GT grounding 方向，同幅度 ×1（诊断）','ground_same_radius10':'GT grounding 方向，同幅度 ×10（诊断）','ground_fit10':'GT grounding 拟合相同 TTA 接口，10 步（诊断）','ground_match10':'GT grounding 拟合匹配头，10 步（诊断）','wrong_binding_aux3':'错配辅助监督，3 步'}
table='\n'.join(f"| {labels[a]} | {s['mIoU_percent']:.6f} | {s['gain_vs_bidirectional_frozen_pp']:+.6f} | {s['Acc25_percent']:.4f} | {s['Acc50_percent']:.4f} | {s['query_changes_vs_frozen']} | {s['improved_vs_bidirectional_frozen']}/{s['worsened_vs_bidirectional_frozen']} |" for a,s in S.items())
tnames={'source':'原始 3EED','frozen_ABC':'ABC','R59':'R59 直接表示融合','bidirectional_frozen':'冻结双向模型','bidirectional_frozen_budget3':'同双向模型三观测直接平均','bidirectional_TTA1':'双向 TTA1','bidirectional_TTA3':'双向 TTA3','bidirectional_TTA3_wrong_correspondence':'错配辅助 TTA3','bidirectional_wrong_RGB':'冻结双向，错配 RGB'}
tt='\n'.join(f"| {name} | {T['drone'][a]['mIoU_percent']:.6f} | {T['quad'][a]['mIoU_percent']:.6f} | {T['balanced_mIoU_percent'][a]:.6f} |" for a,name in tnames.items())
text=f'''# R77：双向 RGB–LiDAR TTA 的步幅、方向与接口能力诊断

本轮结论：**小步幅确实限制了选框变化，但当前辅助任务的更新方向与 grounding 缺乏稳定一致性；单纯增大更新不是有效修复。保留双向结构，不采用当前 TTA 配方。** 同一组102498个参数在 grounding 标签指导下具有明显纠错能力，因此不能把失败归结为“这个接口影响不了选框”。

## 固定范围与标签边界

保留 Waymo-only epoch100、R76 双向模块 epoch6、原 A256 框、全部锚点及六层 decoder 的固定位置。没有新训练源模型，没有增加外挂专家，没有访问 FINAL。原始3EED冻结；本轮修改的是已经源域训练过的轻量模块或单独的匹配头副本，逐输入恢复。源域真实标签只用于用户授权的受控能力／方向诊断；**没有把 GT 方向、GT 参数或 GT 接受规则送入目标域**。

使用相同128条 Waymo VAL、82个 recording 分组，全部计入分母。这些数据在R76已经用于源域epoch选择，不能当作独立确认集。GT更新后在同输入评分是能力诊断，不是TTA结果，也不证明泛化。源域候选 oracle 平均IoU为 **{M['candidate_oracle_mIoU_percent']:.6f}%**；监督使用候选IoU经temperature0.1 softmax的软标签，仍是代理任务，而不是每例都选择最佳IoU的完美监督。

对照固定为：原R76辅助目标（两种互补mask、恢复／图像稀疏深度、source保持）、Adam 1e-4、clip1、9次回溯、1／3步；原始第一步Adam方向一次赋值×1／×10；GT grounding 第一Adam方向按每例归一到**相同参数L2半径**×1／×10；相同TTA接口和单独匹配头各做10步GT grounding能力拟合（lr1e-3、clip1、grounding loss回溯）。×10是预先固定的受控幅度干预，没有逐例搜索，没有选择最好步长。

## 全128条源域结果

mIoU、Acc为百分数；变化为百分点。改好／改坏指相对冻结双向模型的IoU变化，非人工实例身份认证。

| 对照 | mIoU % | Δ冻结双向 pp | Acc@25 % | Acc@50 % | 换query数 | 改好／改坏 |
|---|---:|---:|---:|---:|---:|---:|
{table}

**第一项证据是步幅。** 原1／3步辅助更新均未改变任何选框；GT同幅度×1也只有两条改变且净收益为负。小的有效表示改动确实不容易跨越排序边界。不是所有“监督更正确”的微小更新都立即提高离散Top1指标。

**第二项证据是方向。** 相同102498参数、相同每例修改半径的×10干预，辅助方向得到 **{S['aux_raw10']['mIoU_percent']:.6f}%**，GT方向得到 **{S['ground_same_radius10']['mIoU_percent']:.6f}%**，相差 **{S['ground_same_radius10']['mIoU_percent']-S['aux_raw10']['mIoU_percent']:.6f} pp**。辅助方向改好2条、改坏4条；GT方向改好12条、改坏9条。不能要求每例都涨，但总体差异否定了“只需把同一个辅助方向推大”的修复解释。

**第三项证据是接口容量。** 相同TTA接口的GT10步达到 **{S['ground_fit10']['mIoU_percent']:.6f}%**，相对冻结提高 **{S['ground_fit10']['gain_vs_bidirectional_frozen_pp']:.6f} pp**，55条改善、24条恶化。单独匹配头GT10步为 **{S['ground_match10']['mIoU_percent']:.6f}%**，提高 **{S['ground_match10']['gain_vs_bidirectional_frozen_pp']:.6f} pp**，70条改善、12条恶化。后者参数量184960，与TTA接口102498不同，仅作额外能力参照，不能当同预算优胜算法。

## 梯度与实际Adam方向

初始点同时计算原辅助总目标与源域grounding目标梯度。原始梯度cosine均值 **{M['gradient_cosine_mean']:.6f}**、中位数 **{M['gradient_cosine_median']:.6f}**，只有53/128为正。进一步计算实际第一步Adam参数变化与grounding梯度的内积：61/128预计降低grounding损失，**67/128预计升高**。Adam会改变梯度度量，因此同时保留原梯度和真实优化步两项，而不单凭cosine下判断。独立审计重新计算了全部128条的两种原生梯度，并核验1152份保存的真实参数状态。

原辅助3步将辅助损失均值从 **{M['frozen_auxiliary_mean']:.9f}** 降至 **{M['arms']['aux3']['auxiliary_mean']:.9f}**，grounding损失却从 **{M['frozen_ground_loss_mean']:.9f}** 变为 **{M['arms']['aux3']['ground_loss_mean']:.9f}**；57条grounding损失下降、71条升高。×10后辅助损失继续下降至 **{M['arms']['aux_raw10']['auxiliary_mean']:.9f}**，grounding损失为 **{M['arms']['aux_raw10']['ground_loss_mean']:.9f}**，mIoU下降。

GT同幅度×1使全部128条grounding代理损失下降，最终mIoU仍略降。这是另一项限制：即使grounding软标签CE改善，也不等于每例IoU或Top1改善。后续不能只用梯度对齐或CE下降宣布成功，仍需完整任务指标。

## 保留的目标域参照（R76历史结果，本轮没有重跑）

全部DEV336，两平台等权平均。保留此前4032个任务值、3534个有序负比较和全部五个场景结果；本轮没有新增目标域推理或评分。

| 方法 | Drone mIoU % | Quad mIoU % | 两平台平均 % |
|---|---:|---:|---:|
{tt}

预先注册的目标域放大更新准入条件未通过：源域×10辅助方向没有优于冻结或原1步，且只有61/128个方向预测有利。依规则不把无依据的增步／增幅搬去目标域。该决定只限制这份配方，不把source梯度结论包装成已测得的target梯度结论。冻结双向模型仍是development候选，与R59总体几乎持平并存在平台取舍；当前辅助TTA没有超过同信息冻结基线。

## 下一步机制决定

保留几何约束双向交互和冻结基线；停止当前辅助目标的纯步长／步数扩张，也不重新找decoder层。下一项有依据的改变是：**在Waymo上只对新增轻量模块做“更新后grounding”训练准备**，内循环使用现有无标签辅助更新，外循环用源域grounding标签评价更新后的任务表现；与同数据、同优化预算的普通grounding+aux联合训练比较。这样直接检查“辅助更新是否服务于定位”，而不是只在同一状态联合降低两项损失。

这借鉴的是 [Point-TTA（ICCV2023）的meta-auxiliary准备](https://openaccess.thecvf.com/content/ICCV2023/html/Hatem_Point-TTA_Test-Time_Adaptation_for_Point_Cloud_Registration_Using_Multitask_Meta-Auxiliary_ICCV_2023_paper.html)：它在配准任务中使用辅助更新后的主任务损失训练初始化。原论文也区分普通联合训练与更新后的任务对齐。迁移到grounding是后续待验证设计，不是论文已经证明的结论。本轮**未启动**这项新训练，也没有采用GT选择后的目标域更新。

## 完整性与成本

128×11=1408个源域任务值、2143个有序负比较、902个recording×arm行全部保留，没有GT筛样本，没有删除误改；DEV336历史控制及全部负例单独保存。源码、配置、匿名逐例、每层参数梯度标量、所有优化步骤与recording表公开；原始媒体、文字、标定坐标、标识符、特征、梯度向量和权重不公开。

源域干预实测 **{C['seconds']:.3f}秒**，13568次完整grounding后缀计算、7746次辅助模块计算；独立重放与梯度核验 **{A['seconds']:.3f}秒**、2688次后缀。使用已缓存的点云／文字前缀，不能解释为无缓存端到端延迟；峰值新进程torch分配 **{C['peak_GPU_bytes']/1e9:.3f}GB**，不是整系统显存。

重放最大分数误差{A['max_probability_error']}、最大损失误差{A['max_loss_error']}，方向重建误差{A['max_direction_reconstruction_error']:.3g}。原epoch100文件SHA不变，R76已训练模块逐例恢复，其他paused方向登记不变。审计001的dict迭代错误和分析001的冻结端点表缺项已修复并保留失败源码／日志／部分输出；没有重跑或修改模型预测，也没有放宽数值阈值。

**最终决定：保留冻结双向结构；不采用现有辅助TTA；证据支持同时存在小步幅限制与更新方向不一致，其中后者阻止“只加大更新”成为修复。下一步应对齐源域的更新后任务目标，先与普通联合训练作受控比较。FINAL未动，没有后续任务排队。**
'''
(H/'REPORT.md').write_text(text)
print({'report_bytes':len(text.encode()),'source_n':128,'target_rows_new':0})
