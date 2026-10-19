"""Render measured cached diagnostics; no additional scoring or predictions."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_pair_direction_dev64_v1 import D,P,read,put

def emit(path,value):
 if path.exists():assert read(path)==value
 else:put(path,value)

def main():
 a=read(P/'RESULTS_PUBLIC.json');b=read(P/'CONDITIONAL_ANALYSIS.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
 def val(pan,ds,k,extra=False,scale=1):
  x=(b if extra else a)['groups'][pan][ds]['metrics'][k]
  if x['mean']is None:return '未定义(n=0)'
  return f"{x['mean']*scale:.3f} [{x['CI95'][0]*scale:.3f}, {x['CI95'][1]*scale:.3f}] (n={x['parents']})"
 def table(keys,pan='combined'):
  out=['| 指标 | HC | VidSTG |','|---|---:|---:|']
  for title,key,sup,scale in keys:out.append('| '+title+' | '+val(pan,'hc2',key,sup,scale)+' | '+val(pan,'vidstg',key,sup,scale)+' |')
  return '\n'.join(out)
 text='''# 开发缓存对称候选方向检查：教师偏好、候选可用性与真实合成更新

本轮完成离线诊断，没有产生新的 OPD 或部署成绩。已有教师反馈包含有限方向信息，也存在明显错误偏好；合成位移没有普遍反转正确分量。现有数据不能将净收益小唯一归因于教师、采样尺度或单框读出。

## 范围和数据边界

两批已曝光开发数据：原 NH16/NV16、后续参与选参的 CH16/CV16，共64个独立父视频，每父1query。当前冻结 Live 路径：HC步长1.0、Vid步长0.5、温度0.1、两轮，每节点17候选（当前框加8对对称偏移），Fine layer3有序ROI反馈、四帧缓存DINO、E+finalMBR和纯预测插值支持范围均不变。HC原面板取已存 Step_high，Vid取 Fine_live；选参面板分别取 Live_step10/05。

FH/FV正式评估64排除，仅验证原文件hash未变，不读取其标签或重新评分以选择本轮改动。预训练模型、原预测、配置及CURRENT/MAIN/PAPER均不改；无GPU、模型前向、专家、DINO、特征提取、训练、新候选或新适应更新。

全64缓存几何与贡献分解先封存，再做GT离线诊断。GT未知与ROI无效分开记录，不能当负例。64父全部保留在逐query结果；条件方向统计仅对有相应支持的父定义，表中n明确给出。所有比例先query内汇总再父macro；不是把大量pair当独立样本。区间是按父bootstrap的描述性95%区间，未作多重性校正，不能当独立确认。

## 1. 有相对方向信息，但不等于知道如何超过当前框

'''+table([
 ('有限±pair的方向准确率，%','raw_pair_accuracy',False,100),
 ('pair至少一侧优于当前框，%','pair_has_improver_rate',False,100),
 ('pair两侧均差于当前框，%','pair_both_harm_current_rate',False,100),
 ('被偏好侧确实优于当前框，%','preferred_side_improves_current_rate',False,100),
 ('相对方向正确但pair无改善候选，%','correct_but_no_improver_fraction',False,100),
 ('仅在有改善候选pair上的方向准确率，%','has_improver_direction_accuracy',True,100),
 ])+'''

“正确”只表示有限幅度的两个候选中偏好IoU较高者，不是GT局部梯度，也不保证优于当前框。对称两侧先验相同，因此raw分数与posterior的方向准确率相同，是应有的数值性质，不能说softmax改善了方向。

进一步按has_improver、both_harm、both_equal、one_equal_harm四个互斥GT层记录方向及范数质量，完整交叉表见CONDITIONAL_ANALYSIS。已知GT、双侧valid pair的贡献范数中，有改善候选pair占HC25.739%/Vid16.544%，两侧均差pair占60.310%/56.436%。这些是向量范数质量，不是IoU损失的因果份额。两侧均差并不证明小步差分方向无用，不能据此在线用GT删pair。

## 2. 先确认候选是否能超过当前框，保留面板差异

'''+table([
 ('17点有效视觉支持内有改善候选的节点比例，%','valid_support_has_improver_rate',True,100),
 ('有效支持内存在至少+5 IoU pp候选，%','valid_support_headroom5pp_rate',True,100),
 ])+'''

以下按原面板和选参面板分别给出有效支持内有改善候选的节点比例：

| 面板 | HC | VidSTG |
|---|---:|---:|
'''+f"| 原NH/NV | {val('original','hc2','valid_support_has_improver_rate',True,100)} | {val('original','vidstg','valid_support_has_improver_rate',True,100)} |\n| CH/CV选参 | {val('selection','hc2','valid_support_has_improver_rate',True,100)} | {val('selection','vidstg','valid_support_has_improver_rate',True,100)} |\n"+'''

特别是Vid从54.832%变为7.087%，不支持把候选覆盖当成跨面板固定事实；也不支持因后者低就立刻扩大尺度。core的candidate_has_improver/headroom5pp最初未mask valid，是全部17点的几何诊断；独立补充另给valid-support版本。本轮这两种headroom数值相同，定义仍分别保留，不能将不看图像的几何候选默认为有教师支持。

## 3. 真实更新可逐pair精确重构，错误分量有实际权重

标准化中心坐标中，每对贡献为：

`c_j = step * [(q_plus-p_plus)-(q_minus-p_minus)] * epsilon_j`。

所有correct、wrong、GT_tie、teacher_tie、unknown_GT、support_limited分量都保留；sum(c_j)重构真实eta_after−eta_before，独立17点一阶矩核验最大误差1.665e−16。没有裁掉tiny项来制造等式。

'''+table([
 ('wrong范数占correct+wrong，%','wrong_norm_fraction_among_classified',False,100),
 ('有改善pair中wrong范数占已分类范数，%','has_improver_wrong_norm_fraction_classified',True,100),
 ('wrong范数占全部实际pair范数，%','wrong_norm_fraction_all',True,100),
 ('未知GT范数占全部实际pair范数，%','unknown_GT_norm_fraction_all',True,100),
 ('正确分量之和与实际位移反向的节点，%','correct_vector_opposed_rate',False,100),
 ('合成向量范数/各pair范数之和','cancellation_ratio',False,1),
 ])+'''

wrong仅在双valid、已知GT且两侧质量与教师偏好均非tie时定义。不能把36.7%/42.0%解释成全部更新或全部真实损失都错了：已知分类之外仍有大量未知GT分量。反向比例也只对两个向量非零的节点定义，不是GT最优方向。

合成保留了约0.69/0.71的范数比例，正确分量与最终方向反向约18.7%/23.1%；存在冲突，但没有证据说普遍严重抵消是主要原因。投影到实际delta的signed分解详见RESULTS_PUBLIC；它可以为负或超过总量份额，不是IoU加性归因。

## 4. 候选分布变好，与单框输出变好是两项检查

'''+table([
 ('教师期望候选IoU−先验期望，pp','expected_quality_tilt_pp',False,1),
 ('教师期望候选IoU−当前框IoU，pp','teacher_expected_IoU_minus_current_pp',True,1),
 ('先验期望候选IoU−当前框IoU，pp','prior_expected_IoU_minus_current_pp',True,1),
 ('期望tilt为正但实际单步框IoU下降，%','positive_tilt_actual_harm_rate',False,100),
 ])+'''

教师比被扰动先验改善约8.25/7.66pp，仍比当前框低约6.84/5.51pp。这说明正tilt不能直接当成相对当前框的新增质量。但非线性IoU的候选期望也不能预测均值框质量：它低于当前框，不等于小步更新必然失败。约31.3%/26.1%的正tilt节点实际下降，足以要求区分两者，尚不足单独证明均值压缩是主因。

## 5. 保留真实改好、改坏与未知结果

本轮不重算完整tube部署结果，两个step的原分母空间变化精确望远镜回已有最终未观察actual sIoU。全64父和scope外/无参考/不动节点保留在原分母，未用active-only增益替代：

'''+table([('原缓存最终未观察actual sIoU−E，pp','all_known_unobserved_final_delta_pp',False,1)])+'''

这只是已曝光开发数据上原成绩的重构，不是新实验收益，也不改变正式64评估结论。保留反例：NH009最终空间−0.9735pp，pair准确率27.1%、wrong已分类范数87.3%；NV004−0.3140pp、准确率25.0%、wrong82.2%。成功例NH011+3.4133pp、准确率80.6%、wrong3.9%；NV005+1.8621pp、准确率69.3%、wrong22.0%。它们是事后个例，不能在线按这些GT统计筛选。

## 6. 裁决与下一步边界

不能再把小收益统一解释为“没更新”、也不能凭这次诊断宣称单一原因：有限pair排序有信息但并不可靠；部分节点有改善空间、部分面板缺乏；合成存在冲突但没有普遍反向。保留Fine-Center HC1/Vid0.5及正式64原结果，不继续扩大步长/拟合程度，不据正式测试换参。

下一项方法建议只保留一个待检验控制：在同一冻结最后轮的17候选支持上，比较当前Gaussian均值拟合与**逐帧空间期望IoU的MBR单框读出**，同时保留匹配的q-MBR与p-MBR（几何学生先验）控制。q-MBR在现有合法候选（含采样中心）中最大化`sum_k q(k)*IoU(b,b_k)`，p-MBR仅将权重换为同轮p；仅用缓存概率与框几何，GT只在全输出封存后评价。最后轮的current是该轮采样中心，未必仍为E；q是构造的working weights，不是校准目标后验。该控制检验task-metric读出是否比一阶矩更保留现有分布信息，不改教师、不新增专家或训练、不扫参数、不按GT选节点；这不是再把KL拟合到最优。

旧correction_distribution64全轨迹共享校正分布的负结果仍有效；此建议使用逐帧、DINO条件化Fine有序ROI的17个中心候选，单位、反馈来源和读出目标不同。不同机制只构成一次最小对照的理由，不证明会有效。当前q仍存在错误偏好，MBR不会把错误反馈变正确；q-MBR若只优于原均值却不优于p-MBR，不能把改善归给教师信息；无增量也不能唯一归因教师，可用范围仍受候选支持和多峰性限制。**本轮没有执行该新读出或新的OPD；也不称它已是新方法。**

## 验证、成本与公开

root独立标量官方IoU核26,690候选，逐pair分类24,320次、128真实step、64父原分母telescope、102组指标bootstrap全部通过；原缓存、配置、注册和formal保护hash未变。全部3040个active节点（其中1570有GT）的signed/unknown摘要公开；无GT帧保持未知。

CPU封存分解、离线诊断、独立审计耗时分别见COST_BREAKDOWN；模型/GPU/专家/DINO/训练全部0。原Source/E/特征/反馈缓存成本未包含，不能称首次部署耗时。公开代码、协议、全匿名父结果、条件交叉表、所有active节点摘要、独立审计及负结论；原媒体、标注坐标、候选坐标、权重、raw特征、父身份和文本排除。GPT6.1 Sol/max只读review，root独立裁决。研究候选及生产登记不自动晋升，旧heartbeat和探索保持暂停。
'''
 (P/'REPORT.md').write_text(text)
 core=read(D/'DECOMPOSITION_BARRIER.json');diag=read(D/'DIAGNOSIS_COMPLETION.json');emit(P/'COST_BREAKDOWN.json',dict(new_GPU_seconds=0,new_models_experts_DINO_training=0,cached_geometry_seal_CPU_seconds=core['CPU_seconds'],offline_diagnosis_CPU_seconds=diag['CPU_diagnosis_seconds'],independent_audit_CPU_seconds=audit['CPU_seconds'],total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,old_cache_creation_and_publication_cost_excluded=True))
 emit(P/'PROTOCOL_PUBLIC.json',dict(count=64,panels={'original':['NH16','NV16'],'selection':['CH16','CV16']},frozen_steps={'hc2':1,'vidstg':.5},temperature=.1,rounds=2,candidates=17,pairs=8,new_predictions=0,GT_offline_only=True,formal64_excluded=True,prelocked_core=True,post_core_conditional_supplement_disclosed=True,registries_unchanged=True,all_unknown_preserved=True))
 print('report rendered')
if __name__=='__main__':main()
