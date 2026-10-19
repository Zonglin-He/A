"""Render the frozen64 readout comparison without new scoring or selection."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_spatial_mbr_dev64_v1 import D,P,read,put

def emit(path,value):
 if path.exists():assert read(path)==value
 else:put(path,value)

def main():
 x=read(P/'RESULTS_PUBLIC.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
 def v(pan,ds,c,e):
  r=x['groups'][pan][ds]['paired'][c][e];return f"{r['mean']:+.4f} [{r['CI95'][0]:+.4f}, {r['CI95'][1]:+.4f}]"
 def table(pan,comparisons):
  z=['| 对比，pp | HC完整vIoU | Vid完整vIoU | HC未观察actual sIoU | Vid未观察actual sIoU |','|---|---:|---:|---:|---:|']
  for label,c in comparisons:z.append('| '+label+' | '+' | '.join(v(pan,ds,c,e)for e in ['v','unobserved_actual_s']for ds in ['hc2','vidstg'])+' |')
  return '\n'.join(z)
 s='''# 固定最后轮候选的空间MBR：加入当前中心的匹配读出对照

**本轮读出实验完成、测量通过；没有支持q-MBR稳定替换已有Gaussian均值输出。HC有教师权重相对中心/先验的开发信号，但对均值的优势未建立；Vid相对均值下降。关闭这一个末端读出配置，不改教师、步长或生产登记。**

这是已有OPD路径的末端直接决策实验，不是新的学生更新、增强蒸馏或新的on-policy训练。结果也不能唯一证明教师无效。

## 配置与证据边界

原NH16/NV16开发面板和参与选参的CH16/CV16，共64父，每父1query；冻结TA-STVG跨域EMA（HC用Vid源权重、Vid用HC源权重）、E+finalMBR、Fine layer3 stride16/1024、有序3x3 ROI、四帧原缓存DINO及纯预测插值支持范围。HC step1、Vid step.5，tau.1，原两轮17候选。只使用已保存Live第二/最后轮的候选、valid mask、p和q。正式FH/FV64从本轮计算排除，只保护原文件hash。全体都已曝光开发，不是新增独立确认。

五组为E_MBR、保存的Gaussian最终输出、Last_center、P_MBR、Q_MBR。Last_center是最后轮采样前、通常已经做过首轮更新的学生框，不是E。P_MBR同样继承首轮教师路径，不能称为teacher-free。

P/Q共享同一个原有效ROI积分/动作支持，用normalized完整框的未裁剪IoU作效用；原允许部分框出界规则保留、不新裁剪或重定位。cached所有候选finite且正尺寸，active节点mask外权重为零后仅数值归一化。只在原allowed&informative gate上决策，gate外保留Last_center；scope外与planned节点保持E、宽高及时间精确不变。1e−12数值平分优先current，否则原最低index。没有GT参与决策、选步、gate或回退；所有64×5输出先seal后读GT评分。

q只是工作权重，不是校准后的目标后验。MBR仅表示在这些权重与候选集合下最大化期望候选间IoU，不等于GT风险最优。

## 合并开发面板：主结果

数字为父macro配对差值pp与描述性95%bootstrap区间，主分母每域32父，不把重复帧/候选当独立样本。保留无参考、unknown、零变化与原all-known-unobserved分母；没有以active-only均值替代。区间未作多重比较校正。

'''+table('combined',[(a+' − E',a+'_minus_E_MBR')for a in ['Gaussian','Last_center','P_MBR','Q_MBR']]+[('Q-MBR − Gaussian','Q_MBR_minus_Gaussian'),('Q-MBR − Last_center','Q_MBR_minus_Last_center'),('Q-MBR − P-MBR','Q_MBR_minus_P_MBR')])+'''

HC的Q−Gaussian完整v+0.0057pp、actuals+0.0800pp，两区间均跨零；不足以替换当前输出。Q−Last_center/P的两项区间为正，说明本配置教师权重相对先验/保持当前并非完全无信息，但这部分信息原均值输出也已利用，不能宣称新增蒸馏价值。

Vid的Q−Gaussian完整v−0.0281pp、actuals−0.0838pp，两区间均负；Q相对E四项中的Vid两项仍不确定。Q不能靠“均值为正”晋升为更有效方法。各域分别裁决，不强加同参数或双域共同通过要求；本轮不事后按域拼新的输出赢家。

## 两个面板分别报告

原NH/NV，16父/域：

'''+table('original',[('Q-MBR − E','Q_MBR_minus_E_MBR'),('Q-MBR − Gaussian','Q_MBR_minus_Gaussian'),('Q-MBR − Last_center','Q_MBR_minus_Last_center')])+'''

CH/CV选参面板，16父/域：

'''+table('selection',[('Q-MBR − E','Q_MBR_minus_E_MBR'),('Q-MBR − Gaussian','Q_MBR_minus_Gaussian'),('Q-MBR − Last_center','Q_MBR_minus_Last_center')])+'''

HC对原均值两批v差分别−0.0054/+0.0169pp，均不确定；Vid原面板的两项下降更明显，后续面板接近零且不确定，不能合并后隐去这种异质性。

## 读出实际做了什么

P-MBR在所有节点的输出与Last_center逐元素完全一致。这个配置下，匹配p控制恰好等价于省去最后一轮更新；不是所有候选集上MBR都必然如此的定理。

Q-MBR在HC798个active节点中28个选了非中心候选，Vid722中57个。对其余active节点保留采样中心。主分母仍然是全部query和原known未观察actual节点，不用这85个被修改位置重算好看的主均值。原均值输出是连续小步，离散17点MBR可能保留中心或跳到完整候选，二者输出表达不同，不能将差异直接归为拟合量或普遍抵消。

所有五组时间区间完全相同，observed actual框和sIoU不变。无参考HC1/Vid12父全部保留；有参考但无有效更新机会也不替换样本。完整匿名逐query signed/unknown、几何位移、插值贡献、index/utility及所有gate关闭节点见RESULTS_PUBLIC与NODE_DECISIONS_PUBLIC。插值与actual指标分别记录，不把插值改善当成新增独立视觉监督。

保留成功与负例（事后解释，不是在线选择）：

- NH013：Q相对Gaussian完整v+0.7082pp，Q相对E+0.1256pp；原均值的损伤有部分恢复。
- NH011：Q相对Gaussian完整v−0.5056pp、actuals−1.1843pp，但Q仍比E完整v高+1.2669pp；不能只记为Q成功。
- NV005：Q相对Gaussian完整v−0.4693pp、actuals−0.5587pp，Q仍比E完整v高+0.8581pp。
- CV012：Q相对Gaussian完整v+0.0477pp，但Q相对E仍为−0.0566pp；缓解损伤不是超过E。

## 判断与停止范围

本轮没有达到Q同时超过E、原均值、P-MBR和最后轮中心且fullv/actuals一致改善的证据要求。HC只确认对保持中心/先验的条件收益，对当前均值不确定；Vid不支持采用。**关闭当前固定支持的末端q-MBR替换，不调tie、候选支持、温度或步长救结果，不自动扩大数据。**保留当前Fine-Center及原正式64评估记录，不因开发诊断改变它们。

这只限制该readout配置。原候选覆盖、教师错权重、工作分布多峰与单框表达仍可能共同影响结果；本轮没有唯一归因。旧correction_distribution64共享全轨迹MBR负结果保留，本轮逐帧Fine ROI局部支持不同，不能写成同一实验重复或所有空间MBR不可能。

## 工程恢复、独立核验与成本

第一次CPU运行完成NH/NV32个输出后，在CH缓存缺少active元数据字段时报错，评分尚未开始。修复仅将参考存在性读取为缓存per_anchor_rewards首维是否非零；原源有active字段时额外核对一致。没有修改候选、概率、mask、gate或读出规则。原代码、原锁、故障日志及原32个输出全部保存；第二次逐字段验证原32输出bitexact后复用，没有覆盖。原PLAN/RUNTIME不重写，单独recovery锁只替换该代码pin，原其余pins保留。全64barrier之后才启动GT评分，读出阶段不读新GT。该工程故障不是方法负结果。

root独立scalar candidate-pair IoU439,280次、P/Q选择3040次、128个重构读出、1920个指标、360个父bootstrap对比通过；utility最大误差4.44e−16，指标误差0。原Source/缓存/参数/生产注册/formal64保护hash未变。两个owned CPU单元已终止，第一次失败记录保留，第二次success inactive；无其它服务信号。

新增GPU/模型/Source/专家/DINO/训练/特征/ROI评分/采样/学生更新均0。读出、评分、独立审计与工程阶段成本分列COST_BREAKDOWN；旧缓存/Source/E/教师构造与公开成本不包含，不冒称首次部署延迟。实现、协议、冻结配置、所有匿名signed/unknown、负例、audit及工程恢复说明按allowlist公开，原媒体/GT或候选坐标/权重/rawfeatures/父ID和文本不公开。CURRENT/MAIN/PAPER不晋升，旧探索和heartbeat仍暂停。
'''
 (P/'REPORT.md').write_text(s)
 emit(P/'COST_BREAKDOWN.json',dict(new_GPU_seconds=0,total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_models_experts_training_OPD=0,readout_recovery_pass_CPU_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['CPU_seconds'],offline_score_CPU_seconds=read(D/'SCORE_COMPLETION.json')['CPU_seconds'],root_audit_CPU_seconds=audit['CPU_seconds'],first_attempt_failed_after32_no_GT=True,original32_partials_preserved_bitexact=True,old_cache_and_publication_cost_excluded=True))
 emit(P/'PROTOCOL_PUBLIC.json',dict(status='complete',count=64,panels=['NH16/NV16','CH16/CV16'],frozen_HCstep1_Vidstep05_tau01=True,arms=['E_MBR','Gaussian','Last_center','P_MBR','Q_MBR'],original_valid_support=True,active_gate='original allowed & informative',fixed_final_round=2,finite_candidates=17,numerical_tie_tolerance=1e-12,tie_priority='current if valid, otherwise original lowest index',partial_outside_boxes_unchanged=True,IoU_utility='unclipped normalized boxes',all_outputs_before_GT=True,new_OPD=False,formal64_excluded=True,production_registries_unchanged=True))
 emit(P/'ENGINEERING_RECOVERY.json',dict(status='pass',first_attempt='KeyError active after32 outputs; no scoring',only_change='derive reference presence from cached per_anchor_rewards shape[0]',original_plan_runtime_and32_outputs_preserved=True,old32_predictions_bitexact_verified=True,predictions_or_metrics_not_selected_by_GT=True,readout_rules_and_support_unchanged=True))
 emit(P/'DECISION.json',dict(status='evaluated',HC='Q better than last center/prior; superiority over Gaussian unresolved',Vid='Q below Gaussian for fullv and actuals on combined development panel',adopt_Q_MBR=False,close_this_readout_configuration=True,not_teacher_or_OPD_route_refutation=True,no_parameter_change_or_auto_followup=True))
 print('report written',flush=True)
if __name__=='__main__':main()
