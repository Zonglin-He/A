"""Locked primary and factorial, all development negatives retained."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round107';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;a=rd(O/'AGGREGATE_ROWS.json');aa={x['arm']:x for x in a if x['group']=='all_six_equal'};v=lambda k:aa[k]['mIoU_percent'];p='real_mean_keep_fusedABC';controls=['original_source','prepared_frozen','real_frozen_fusedABC','real_frozen_AB','real_frozen_AB_C','real_frozen_ensemble4','real_mean_direct_matched_C','real_native_direct_matched_C','real_TTA_fusedABC','norm_old_direction_fusedABC','randomA_mean_keep_fusedABC','wrong_mean_keep_fusedABC','teacher_shuffle_keep_fusedABC'];rows=[x for f in O.glob('*_ALL_ROWS_PRIVATE.json') for x in rd(f)];conc=[];fits=[];sub=[]
 for c in controls:
  d=np.array([r['arms'][p]['IoU']-r['arms'][c]['IoU'] for r in rows]);conc.append({'control':c,'n':len(d),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,d.max())*100/768),'largest_negative_single_input_pp':float(min(0,d.min())*100/768)})
 for label in ['real_mean_keep','real_native2','real_native_keep','randomA_mean_keep','wrong_mean_keep','teacher_shuffle_keep']:
  f=[r['fits'][label] for r in rows if label in r['fits']];fits.append({'fit':label,'n':len(f),**{k:float(np.mean([x[k] for x in f])) for k in ['initial_error','final_error','relative_error','accepted_steps','attempted_steps','parameter_L2']}})
 for group in ['all','enabled','B_available_C_missing','protected_alias']:
  rr=[r for r in rows if group=='all' or (group=='enabled' and r['status']['updated_output_enabled']) or (group=='B_available_C_missing' and r['fits'] and not r['status']['updated_output_enabled']) or (group=='protected_alias' and r['status']['protected_alias_fallback'])]
  if rr:
   for k in [p]+controls+['real_native2_fusedABC','real_native_keep_fusedABC']:sub.append({'subset':group,'n':len(rr),'arm':k,'mIoU_percent':float(np.mean([r['arms'][k]['IoU'] for r in rr])*100)})
 factorial=[]
 for endpoint in ['fusedABC','C']:
  mean2=v('real_TTA_'+endpoint);mk=v('real_mean_keep_'+endpoint);native2=v('real_native2_'+endpoint);nk=v('real_native_keep_'+endpoint);factorial.append({'endpoint':endpoint,'mean2':mean2,'mean_keep':mk,'native2':native2,'native_keep':nk,'preservation_effect_at_mean_pp':mk-mean2,'supervision_effect_without_keep_pp':native2-mean2,'supervision_effect_with_keep_pp':nk-mk,'interaction_pp':nk-native2-mk+mean2})
 gates=['real_frozen_fusedABC','real_frozen_AB_C','real_frozen_ensemble4','norm_old_direction_fusedABC','randomA_mean_keep_fusedABC','wrong_mean_keep_fusedABC','teacher_shuffle_keep_fusedABC','real_mean_direct_matched_C','real_native_direct_matched_C'];passed=all(v(p)>v(c) for c in gates);decision={'primary':p,'mIoU_percent':v(p),'gains_pp':{c:v(p)-v(c) for c in controls},'passes_development_gates':passed,'robust_adoption':False,'decision':'lock_before_expansion_required' if passed else 'do_not_adopt_fused_state_preservation_recipe','n':768,'primary_not_switched':True,'FINAL_access':False,'source_retraining':False,'external_experts':False}
 for name,data in [('DECISION',decision),('CONCENTRATION_SUMMARY',conc),('FIT_SUMMARY',fits),('SUBSET_RESULTS',sub),('FACTORIAL_EFFECTS',factorial)]:dump(H/(name+'.json'),data)
 lines=['# R107 — 固定实测点观测的监督来源与融合状态保持','','沿用R106真实注册B/C点云、同帧randomA和错配支持，原A RGB、坐标、文本视角、1024空间seed及256最终框不变。冻结三个source100与R91双向结构；只更新18496参数末端匹配投影，不增加专家或source训练。','','## 锁定对照','','R106原始均值目标两视图拟合原样复用。主实验保持相同目标q=(pA+pB)/2，在A/B拟合之外，以source原生fusedAB分布为第三视图的完整输出保持目标。两个诊断分别改为q=pAB两视图拟合，以及q=pAB+第三视图保持；等视图权重、LM10×6不变。R30已有输出保持思想，本轮不将保持项宣称为新贡献。','','额外AB源状态的计算属于新增信息/成本，完整提供冻结AB、AB+C、四分布平均、直接q及覆盖匹配q+C。R106旧方向按新主实验的参数L2范数缩放，独立检查效果是否仅来自减弱更新；只使用权重，不用C或GT。随机A/错配/打乱目标均保留。','','C/fusedABC缓存首次进入本轮程序在所有拟合和范数控制之后，事件顺序经独立CPU核验。但C和DEV已历史暴露，不能称独立确认。新增原生AB提取的B采样、物理点哈希、注册哈希与R106逐条一致；12条完整原生写参数前向验证，不依赖重放自洽而已。','','## 全体结果','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in a:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 锁定主方法与全部强对照','','| 对照 | Δ pp | 改善/恶化/不变 |','|---|---:|---:|']
 for x in conc:lines.append(f"| {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 lines+=['','## 2×2机制结果','',json.dumps(factorial,indent=2),'','## 范数与方向','',json.dumps(rd(H/'NORM_CONTROL_SUMMARY.json'),indent=2),'','## 拟合','',json.dumps(fits,indent=2),'','## 裁决','',json.dumps(decision,indent=2),'','## 证据边界','','55臂×768=42240读出，全部负例、UNKNOWN、保护媒体别名回退及C缺失回退均留在总体。258条执行拟合，178条实际使用更新输出；这些不能代表全体。目标分布和直接读出在评分前封存，GT只用于事后测量。相同场景/开发标签重复暴露，不能用最佳诊断臂代替预先主方法，也不能以某一平台个别病例决定门控。原source作为系统质量参照，FINAL从未读取。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
