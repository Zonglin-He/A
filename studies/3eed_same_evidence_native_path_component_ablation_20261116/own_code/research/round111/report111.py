"""Frozen-path signal verdict, source-preparation component diagnostics and all negatives."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round111';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;agg=rd(O/'AGGREGATE_ROWS.json');aa={r['arm']:r for r in agg if r['group']=='all_six_equal'};v=lambda k:aa[k]['mIoU_percent'];p='native_real_fusedABC';controls=['original_source','original_source_matchedcoverage','prepared_frozen','native_real_ensemble','native_randomA_fusedABC','native_randomA_ensemble','native_wrong_fusedABC','real_frozen_fusedABC','real_frozen_ensemble','real_TTA_fusedABC','direct_three_geometric'];rows=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];conc=[];sub=[];cases=[]
 for c in controls:
  d=np.array([r['arms'][p]['IoU']-r['arms'][c]['IoU'] for r in rows]);conc.append({'control':c,'n':len(d),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,d.max())*100/768),'largest_negative_single_input_pp':float(min(0,d.min())*100/768)})
 for label in ['all','nonalias','actual_C_available','no_actual_C_nonalias','protected_alias']:
  rr=[r for r in rows if label=='all' or (label=='nonalias' and not r['status']['protected_alias_fallback']) or (label=='actual_C_available' and r['status']['families'].get('real',{}).get('C_available',False)) or (label=='no_actual_C_nonalias' and not r['status']['protected_alias_fallback'] and not r['status']['families']['real']['C_available']) or (label=='protected_alias' and r['status']['protected_alias_fallback'])]
  if rr:
   for k in [p]+controls:sub.append({'subset':label,'n':len(rr),'arm':k,'mIoU_percent':float(np.mean([r['arms'][k]['IoU'] for r in rr])*100)})
 for direction in rd(R/'crossplatform/round111/MATRIX_SEALED.json')['directions']:
  rr=[r for r in rows if r['source']+'_to_'+r['target']==direction]
  for control in ['real_frozen_fusedABC','native_real_ensemble']:
   for kind,sign in [('good',1),('failure',-1)]:
    part=[r for r in rr if sign*(r['arms'][p]['IoU']-r['arms'][control]['IoU'])>1e-12]
    if part:
     r=max(part,key=lambda r:abs(r['arms'][p]['IoU']-r['arms'][control]['IoU']));cases.append({'source':r['source'],'target':r['target'],'row_PRIVATE':r['id_PRIVATE'],'kind':kind,'control':control,'delta_IoU':r['arms'][p]['IoU']-r['arms'][control]['IoU'],'postseal_only':True})
 factor={'source_native_projection':v('original_source_matchedcoverage'),'source_prepared_projection':v('sourceA_prepared_head'),'preparedA_native_projection_base':v('preparedA_native_head_base'),'preparedA_prepared_projection_base':v('preparedA_prepared_head_base'),'preparedF_native_projection_base':v('preparedF_native_head_base'),'preparedF_prepared_projection_base':v('preparedF_prepared_head_base'),'preparedA_full':v('prepared_frozen'),'preparedF_full':v('real_frozen_fusedABC'),'head_comparison_h_and_text_fixed_within_input_system':True,'base_rows_exclude_late_RGB_and_competition_residual':True,'latent_system_comparison_includes_its_own_conditioned_text':True,'not_a_deployment_GT_selector':True}
 gate=all(v(p)>v(c) for c in controls);decision={'primary':p,'mIoU_percent':v(p),'gains_pp':{c:v(p)-v(c) for c in controls},'passes_frozen_signal_gate':gate,'TTA_run':False,'robust_TTA_adoption':False,'decision':'lock_expansion_before_further_method_if_signal_passes' if gate else 'stop_original_native_fusion_recipe_no_TTA_from_this_signal','n':768,'primary_not_switched':True,'FINAL_access':False}
 for name,x in [('DECISION',decision),('CONCENTRATION_SUMMARY',conc),('SUBSET_RESULTS',sub),('HEAD_FACTORIAL_DIAGNOSTIC',factor),('GOOD_FAILURE_CASES_PRIVATE',cases)]:dump(H/(name+'.json'),x)
 lines=['# R111 — 同一实测点云证据的原始路径撤回对照','','R110统计适应失败后，先区分新增观测不足与source-prepared路径干扰。此轮没有TTA、梯度、更新或新监督；没有替换生产主方法。原始3EED和既有R91双向模型均保留。原A256/六层source A位置/三source100/物理配准/16384点预算/语句/UNKNOWN策略固定。全部768输入，44保护别名不重读而以prepared冻结回退计入。','','## 最小改变与强对照','','唯一主干差异是把与R106完全相同的已对齐B/C PointNet特征送入原始cross-encoder/decoder和原始匹配头，撤回R91新增分支。各层位置仍取source A，原框不回归；不是换一组候选。真实/同A随机重采样/错配三组都比较B、C、早期特征融合和最终概率均值。相同信息的native概率均值是必要强对照。其余58已有读出及评分严格复用，仅新增17读出。没有因为旧CE配方失败而再拟合其输出。','','原始A重放的h/query/tokens与R110全部位等。首次正式尝试在第73条触发1e-7概率校验：批量矩乘读出与规范二维矩乘相差1.4685e-7，h/query/tokens不变。独立探针确认二维读出误差为0后，在GT读取之前修正；前72条复用隐藏表示重放读出，没有重跑PointNet/decoder。失败版本与封存哈希保留。12工程输入再次通过规范读出、身份融合、物理采样哈希、源指纹和独立CPU匹配投影验证。不是提高误差容忍来绕过断言。','','## 原始与prepared匹配头的受控互换','','O表示原始h与其文字表示，P表示prepared h与其文字表示。各输入系统内部只替换投影层；基础项均不含late RGB与竞争残差。完整prepared行保留作为应用系统参照。跨系统hidden即使维数相同也不假定语义基相同，互换是干预，不是把某一头当正确教师。','','```json',json.dumps(factor,indent=2),'```','','## 全体/平台结果','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 全部配对结果','','| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---|---:|---:|']
 for x in rd(O/'ALL_PAIRED_ROWS.json'):lines.append(f"| {x['source']} | {x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 lines+=['','## 裁决','','```json',json.dumps(decision,indent=2),'```','','此处冻结融合增量不是参数适应贡献。DEV历史暴露、同物理输入两incoming source相关、Drone/Quad场景很少，逐场景完整保留，不作独立输入置信区间。actual C与A回退单列，不能把所有C都宣称为新观测泛化。负例、错配、未知、失败尝试与44别名均不筛掉。未过门槛则停止这个冻结证据配方，不继续扫强度或训练新接口；通过也只能支持后续固定扩展验证，不直接支持新的TTA损失。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
