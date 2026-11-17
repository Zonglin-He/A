"""All-input repair/acceptance accounting, never labels for deployment."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round126';B=R/'crossplatform/round126';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;rr=[r for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];agg=rd(O/'AGGREGATE_ROWS.json');post=rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json');dec=rd(H/'DECISION.json');process=[];conc=[];cases=[];components=[]
 for r in rr:
  if r['protected_alias_fallback']:continue
  rec=rd(B/r['cohort']/(r['source']+'_to_'+r['target'])/'records_PRIVATE'/(r['id_PRIVATE']+'.json'));row={k:r[k] for k in ['cohort','source','target','id_PRIVATE','B_active','C_available']};row.update(full_teacher_entropy=rec['full_teacher_entropy'],extra_CPU_shared_seconds=r['seconds'])
  for k,v in rec['choices'].items():row['selected_'+k]=v
  for k,v in rec['full_AB_objectives'].items():row['full_objective_'+k]=v
  process.append(row)
 for co in ['DEV','EXP']:
  rows=[r for r in rr if r['cohort']==co];states={(r['source'],r['id_PRIVATE']):r for r in process if r['cohort']==co};account={'cohort':co,'all_n':len(rows),'nonalias':len(states),'categories':[]}
  for label,sign in [('raw_update_helped',1),('raw_update_harmed',-1),('raw_update_IoU_unchanged',0)]:
   part=[r for r in rows if not r['protected_alias_fallback'] and (sign*(r['arms']['corr3_F']['IoU']-r['arms']['mean_context']['IoU'])>1e-12 if sign else abs(r['arms']['corr3_F']['IoU']-r['arms']['mean_context']['IoU'])<=1e-12)];accepted=[r for r in part if states[(r['source'],r['id_PRIVATE'])]['selected_corr']!='zero'];rejected=[r for r in part if states[(r['source'],r['id_PRIVATE'])]['selected_corr']=='zero'];account['categories'].append({'postscore_GT_outcome_NOT_selector':label,'n':len(part),'accepted':len(accepted),'rejected':len(rejected),'retained_gain_or_harm_pp_full_denominator':float(sum(r['arms']['corr3_F']['IoU']-r['arms']['mean_context']['IoU'] for r in accepted)/len(rows)*100),'removed_gain_or_harm_pp_full_denominator':float(sum(r['arms']['corr3_F']['IoU']-r['arms']['mean_context']['IoU'] for r in rejected)/len(rows)*100)})
  for family in ['corr','wrong','random','free','corr013']:account['selection_'+family]={k:sum(r['selected_'+family]==k for r in states.values()) for k in sorted({r['selected_'+family] for r in states.values()})}
  components.append(account)
  for ctrl in ['mean_context','corr3_F','full_direct_AB','full_direct_ABC','full_teacher_bridge','fullval_wrong_F','fullval_random_F','fullval_free_F','double_zero_F']:
   delta=np.array([r['arms']['fullval_corr_F']['IoU']-r['arms'][ctrl]['IoU'] for r in rows]);conc.append({'cohort':co,'method':'fullval_corr_F','control':ctrl,'n':len(rows),'gain_pp':float(delta.mean()*100),'better':int((delta>1e-12).sum()),'worse':int((delta< -1e-12).sum()),'same':int((abs(delta)<=1e-12).sum()),'query_changes':sum(r['arms']['fullval_corr_F']['query_PRIVATE']!=r['arms'][ctrl]['query_PRIVATE'] for r in rows),'largest_positive_single_input_pp':float(max(0,delta.max())*100/len(rows)),'largest_negative_single_input_pp':float(min(0,delta.min())*100/len(rows))})
   for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
    for kind,sgn in [('good',1),('failure',-1)]:
     part=[r for r in rows if r['source']==src and r['target']==tgt and sgn*(r['arms']['fullval_corr_F']['IoU']-r['arms'][ctrl]['IoU'])>1e-12]
     for r in sorted(part,key=lambda r:-abs(r['arms']['fullval_corr_F']['IoU']-r['arms'][ctrl]['IoU']))[:1]:cases.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':r['id_PRIVATE'],'kind':kind,'method':'fullval_corr_F','control':ctrl,'delta_IoU':r['arms']['fullval_corr_F']['IoU']-r['arms'][ctrl]['IoU'],'zero_IoU':r['arms']['mean_context']['IoU'],'new_IoU':r['arms']['fullval_corr_F']['IoU'],'identity':'UNKNOWN_without_private_visual_review'})
 dump(H/'PROCESS_ROWS_PRIVATE.json',process);dump(H/'CONCENTRATION.json',conc);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);dump(H/'VALIDATOR_COMPONENT_SUMMARY.json',components)
 lines=['# R126 — 完整多模态读出验收现成pointAdapter更新','','本轮在读取任何R125任务分数之前锁定并生成。它不改R125梯度、模型、框或主项，不增加GPU训练，而是检验一个明确接口差异：R125的提案/回溯只看native点云—文字分布，实际输出还包含RGB late与竞争判别。能否用完整读出的一致性，从同一批现成更新中保留有益变化？','','## 只有验收改变','','对R125 source-zero A/B的h/U，用现有sourceA RGB late及竞争残差rA构造完整候选分布，冻结均值Tfull。对每个现成端点，用同一late/rA重算A/B完整匹配分布，目标为平均KL(Tfull||studentA/B)加原单位点特征保持。所有信息来自同一个已准备的3EED/R91；没有外部模型、CLIP缓存、规则颜色、GT或新source训练。','','主项仅在{零步、原R125三步}之间选择；错配、随机和自由特征对照也各只有{零步、三步}两个选项。选择最小目标，且改善须超过max(1e−8,1e−6×|零步目标|)，否则回退零步。{0,1,3}正确提案早停只作为预设次项，不升级主项。没有新增步幅网格或根据本轮GT选强度。','','完整教师仍是模型自己的预测，不是实例身份真值。RGB/竞争读出只参与本轮验收，**没有重新生成完整读出梯度**；现成native提案可能已经丢失有益方向或大步，本轮不能否定所有完整任务梯度。','','所有A/B损失与状态选择确定后，才读取C相关最终读出；C不参与选择。选中输出逐值复制原R125实际写权重后的封存预测，不是额外改分或假写参数。部署时可使用所选原R125权重端点；零步不计为有效参数更新。C评价仍使用A256及A标注，不是新的独立场景/新帧GT。','','## 必须保留的强直接读出','','包含完整教师桥接logTfull+rF−rA；在相同最终late/rF下的完整AB概率平均；完整ABC概率平均。它们均无网络更新，后者直接使用相同三份观测。另保留原冻结均值、head均值、中位数、原生ABC、lateOnly等全部强基线，以及正确/错配/随机/自由变量相同验收。全部负比较在表内。','','验收和新直接读出采用float64匹配MLP/归一化/logsumexp，以NumPy64独立复算Torch64生产；所选参数更新结果仍是原R125 GPU输出。另列相同float64零步F/C，避免将CPU读出精度变化误认作适应。','','## 全体条件与结果','','所有2304历史曝光输入、154保护别名、未知和无B/C在分母；原三source100/R91、A256、六层位置、完整文字和相机条件保持。别名不解码新特征，只沿用preparedA。R125原主项与所有27臂完整保留，R126新增15臂合计42臂96768任务读出值，非独立样本数。候选预测先封存、独立审计后再评分。','','| 范围 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']=='all_six_equal':lines.append(f"| {x['cohort']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 目标平台','','| 范围 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']!='all_six_equal' and x['arm'] in ['original_source','prepared_A','mean_context','corr3_F','fullval_corr_F','fullval_corr013_F','full_direct_AB','full_direct_ABC','full_teacher_bridge','fullval_wrong_F','fullval_random_F','fullval_free_F','double_zero_F']:lines.append(f"| {x['cohort']} | {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 原场景频数后分层','','| 范围 | 方法 | 最小支持覆盖 | 后分层mIoU% |','|---|---|---:|---:|']
 for x in post:
  if x['group']=='all_six_equal':lines.append(f"| {x['cohort']} | {x['arm']} | {x['min_covered_full_mass']:.6f} | {x['mIoU_percent']:.6f} |")
 for name in ['DECISION','HELDOUT_C_TASK_ROWS','VALIDATOR_COMPONENT_SUMMARY','CONCENTRATION','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT']:
  lines+=['',f'## {name}','','```json',json.dumps(rd(H/(name+'.json')),indent=2),'```']
 lines+=['','## 裁决与边界','',('预设开发门通过，仅建立当前历史曝光范围上的有限候选，不等于独立确认或生产部署。' if dec['development_gate_pass'] else '预设开发门未通过，停止这份完整读出验收配方，不事后升级0/1/3次项、平台特例或更多阈值。'),'','组件表事后按真实IoU将原提案分成有益/有害/无变化，统计验收保留和拒绝；这些标签从不进入验收，也不自动证明物体身份。总体高于受损R125、却未超过冻结或同信息直接读出时，不能算TTA成立。','','原场景频数仅来自既有全量清单，EXP缺少4个Waymo场景，只报告共同支持；同一物理输入两个incoming source不能当成独立场景。所有案例与负例保留，未经私有图像复核的身份保持UNKNOWN。没有FINAL、原始媒体、PointNet或外部专家访问。','','本轮seconds是额外CPU多臂验证/读出开销，不包括生成R125更新及原RGB/context/PointNet缓存，不能包装成整个TTA端到端成本。原始媒体、原文、ID、坐标、特征、权重与梯度不公开。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dec,indent=2),flush=True)
if __name__=='__main__':main()
