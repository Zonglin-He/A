"""Combine independently scored R44 phases, no new predictions or scoring."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round44'
def read(p):return json.loads(Path(p).read_text())
def main():
 p1=read(H/'phase1/DECISION.json');p2=read(H/'phase2/DECISION.json');assert p1['gate_passed'] and p2['state']=='STOP_FIXED_R44_PHASE2_CONFIGURATION_NOT_ADOPTED';s=read(H/'phase2/TASK_SUMMARY.json');table=''
 for split,label in [('D208','A208：拟合观测'),('V64','原 V64：未用于拟合、已暴露开发观测')]:
  table+=f'### {label}\n\n| 方法 | Drone mIoU% | Quad mIoU% | 等权平均% |\n|---|---:|---:|---:|\n'
  for a,m in s[split]['balanced_mIoU_percent'].items():table+=f"| {a} | {s[split]['drone'][a]['mIoU_percent']:.6f} | {s[split]['quad'][a]['mIoU_percent']:.6f} | {m:.6f} |\n"
  table+='\n'
 text='''# R44：旧 G 的实例组偏好直接有效，但本次参数迁移无增益

## 当前裁决

**保留阶段一直接使用的正信号，停止阶段二固定 TTA 配置，不扩大 full-grounder closure。** A208 中，旧 G 的两个不同完整实例组资格规则得到32条偏好；固定源代表直接选框将平均 mIoU 从19.663563%提高到24.332121%，净增4.668558pp。随后同32条偏好训练共享查询投影 MLP，在未参与拟合的原 V64 上从28.063498%下降到25.400646%，也低于同历史信息的 all-A 零U memory 25.866269%。

这不说明 G 没有可用信息，也不说明所有 ordinal TTA 无效。它说明本次16步、固定接口/损失/读出的共享参数更新没有实现可靠总体迁移。32条偏好只有19条形成正确方向的组间分数差，不能冒称已经充分拟合；也不因此未经授权加步或换温度。

## 两阶段与历史边界

原 R24/R37 的208条256×6 source boxes与256 scores先通过形状、dtype、数值、字节 exact 门。该门仅证明源兼容，不证明mask身份或G质量。旧F64规则 `s0+2vision+2position+2relation`、qG argmax与历史legacyG的208条query及原框全部精确复现。另保留3条F64/F32 argmax差异，主规则不按任务分数改精度。

阶段一只接受 qG与q0不同、F64严格偏好、各有唯一非空完整组、组不同且成员不交叠的偏好。直接输出正组原source最高分代表，同分取小query；不挑GT好框。169条没有新纠错提名，7条缺唯一映射返回UNKNOWN/source，全部留存，不当负标签。32条eligible中25条改善、1条恶化、6条IoU中性，坏例与中性例都进入阶段二，没有GT筛选。

预锁进阶门是 eligible N>0 且全A208 balanced净增量>0，阶段一PASS仅允许root另授权有限阶段二。受资格约束直接结果比unrestricted旧G低0.395252pp，这个覆盖/固定代表代价没有隐藏。不同detector mask是实例假设而非gold，固定source代表也不能让IoU自动成为语义身份认证。

R24本来已有 qG 对max-other的hinge，R16/17/19已有pair接口，R29/30已有另一条22偏好的RankNet与输出保持。本项不是发明这些机制；新反馈路径、完整组覆盖、偏好分母和keep集合一起改变，不能称只改teacher的纯因果比较。阶段一和阶段二原报告作为历史快照保留，其中未授权文字属于对应时点，不覆盖本合并报告的完成状态。

## 完整任务结果

'''+table+'''
阶段二 A 的group读出相对source −0.013596pp；V64相对source −2.662852pp，相对更强的all-A零U memory −0.465623pp。虽然高于E-only memory 2.127371pp，不能据此采用。叠加相同当前V G以后，TTA也比同信息直接G低2.417191pp；这两臂拥有当前G，不能与仅A历史的迁移臂混算参数收益。Acc25/50、每平台得失、所有query变化和匿名场景表完整保留。

## 参数、监督与对照实际怎样运行

两平台各从原Waymo-only source初始化原query projection MLP：288→288→288→64，共184960参数。decoder、文字投影、backbone与所有原框冻结。使用原R24 A208/原V64缓存，无fit29 import hook。AdamW5e-4、weight_decay0、seed0、16步/平台，共2头32步，无网格。

全部32条反馈逐值绑定阶段一：Drone14、Quad18。group-max exact pretoken log-odds RankNet温度0.1，分母固定14/18；β1相对frozen q0的z输出保持仅用精确A\\E补集，Drone90/Quad86，共U176。没有dense posterior KL、新hinge、参数锚定或第三候选自动负标签。z0在source/no_grad计算，实际token mask、0.07、L2、specials/sentinel及有效padding保持。

A以全部完整组的native概率最大值选组，再返回固定source代表，未分组query保留；另报告raw query诊断。V保持native float32 argmax。source端点精确返回q0，group偏好满足与最终全局赢家分别记录。

两个memory均沿原R30概率单位公式，P/N写入正负半间隔修正，再以frozen query×归一化noun keys作CPU32检索：每A行最近query、top4行、exp((cos−1)/0.1)、alpha1。E-only与all-A零U库分别留存；修正只加native source概率，绝不加z。没有引入QP、exact-z近邻或新超参数。all-A库匹配TTA使用无反馈A观测的历史信息，零U值保持零；当前V G补充臂单独标信息预算。

## 拟合、审计与成本

Drone loss从32.614586降到2.795977（偏好0.478369、保持2.317608），9/14组差为正；Quad从58.989407降到14.014096（偏好8.248947、保持5.765149），10/18组差为正。参数L2变化1.594423/1.624339。拟合不足与跨观测干扰尚未被隔离，损失下降不等于任务正确，不能将某个解释升级为唯一根因或按GT删个别反馈。

受guard的训练、预测和memory过程5.235秒，GPU peak allocated352381952bytes；并非新grounder前向或模型闭环。独立CPU审计272个头向量，p最大误差1.24e-6、z6.82e-6，CPU64 argmax差0；输入、source-z0、keep与全部32偏好和loss独立复算。独立审计完整重算503316480个CPU32相似度单元，128次memory检索的行最大/top4合法集合、值、权重与最终分数均通过。2个top4边界并列保留，实际与同合同重算索引差0，没有float64换赢家。

阶段一624个IoU/62条负参考比较，阶段二1424个IoU/622条负比较全保留；阶段二中624个阶段一控制值逐值复现。因此公开2048个标量、684条比较涉及重复源值和同一A观测，不是2048个独立样本或684个独立坏例。独立样本ID联合为A208加V64共272条，且都是已暴露开发观测、少量已有场景；不作一般显著性、跨新场景稳健性或独立确认声明。

## 封存与使用范围

阶段一所有三臂先seal，再独立审计/评分；阶段二两头及全部A/V/memory预测先seal，无GT数组/损失/全NN审计PASS后才评分。阶段一seal为7c52a420dcced4c426caf34056df7f6c0dfc477dd0ad7b7373347926c68ab733；阶段二为5f45f1af9ed8a7885175579293e1732e6b33c42640213a490ddeaeb218bce23f。

没有新G/VLM调用、源训练、框修正、U/T/FINAL读取或新full-grounder closure。旧G缓存仍含冻结外部视觉语言模型提供的知识和监督；没有新增EMA教师架构不等于监督是teacher-free。停止此固定配置，不追加步数/温度/层搜索，原R24/R30阳性结果及本轮直接G收益继续保留。公开仅自有代码、协议、匿名task/scalar/status/flags与审计元信息；原F分数向量、p/z、keys、features、weights、组成员/mask编号、坐标、caption、rawID、真实scene/path和媒体只在本地受许可资产中。
'''
 (H/'COMBINED_REPORT.md').write_text(text);decision={'run':'completed_both_phases','phase1':'prelocked_gate_PASS_direct_use_only','phase1_eligible':32,'phase1_direct_delta_pp':p1['balanced_gain_vs_source_pp'],'phase1_outcomes':{'improved':25,'harmed':1,'neutral':6,'no_corrective_nomination':169,'unknown_mapping':7},'phase2':'STOP_FIXED_CONFIGURATION_NO_ADOPTION_OR_FULLCLOSURE','phase2_V_TTA':p2['phase2_V_TTA'],'phase2_V_source':p2['phase2_V_source'],'phase2_V_allA_zeroU_memory':p2['phase2_V_memory_allA_zeroU'],'pair_fitting':{'satisfied':19,'total':32,'not_fully_fitted':True},'all2048taskvalues_684negativecomparisons_retained':True,'repeated_A_and_control_values_not_independent':True,'no_general_ordinalTTA_impossibility_or_newnovelty_claim':True,'no_new_G_VLM_source_train_FINAL_U_T_or_fullgrounderclosure':True};(H/'COMBINED_DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n');print('R44_COMBINED_REPORT_COMPLETE')
if __name__=='__main__':main()
