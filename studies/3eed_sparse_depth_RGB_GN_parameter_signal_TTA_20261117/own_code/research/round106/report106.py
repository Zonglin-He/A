"""All task results, matched sampling controls and complete positive/negative tails."""
from pathlib import Path
import json,collections,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round106';B=R/'crossplatform/round106';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;aggregate=rd(O/'AGGREGATE_ROWS.json');a={x['arm']:x for x in aggregate if x['group']=='all_six_equal'};v=lambda k:a[k]['mIoU_percent'];main='real_TTA_fusedABC';controls=['prepared_frozen','real_frozen_fusedABC','real_frozen_ensemble','randomA_TTA_fusedABC','wrong_TTA_fusedABC','teacher_shuffle_fusedABC','joint3_FC'];gains={k:v(main)-v(k) for k in controls};rows=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];cases=[];conc=[];fit=[];subset=[]
 for c in controls:
  d=np.array([r['arms'][main]['IoU']-r['arms'][c]['IoU'] for r in rows]);conc.append({'control':c,'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'net_pp':float(d.mean()*100),'largest_positive_single_input_pp':float(max(0,d.max())*100/768),'largest_negative_single_input_pp':float(min(0,d.min())*100/768)})
 for label in ['real','randomA','wrong','teacher_shuffle']:
  x=[r['fits'][label] for r in rows if label in r['fits']];fit.append({'fit':label,'n':len(x),**{k:float(np.mean([v[k] for v in x])) if x else None for k in ['initial_error','final_error','relative_error','accepted_steps','attempted_steps','parameter_L2']}})
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=[r for r in rows if r['source']+'_to_'+r['target']==name]
  for label in ['all','real_enabled','protected_alias']:
   part=[r for r in rr if label=='all' or (label=='protected_alias' and r['status']['protected_alias_fallback']) or (label=='real_enabled' and r['status'].get('families',{}).get('real',{}).get('updated_output_enabled',False))]
   if part:
    for arm in ['prepared_frozen','real_frozen_fusedABC','real_frozen_ensemble',main,'randomA_TTA_fusedABC','wrong_TTA_fusedABC']:subset.append({'direction':name,'subset':label,'n':len(part),'arm':arm,'mIoU_percent':float(np.mean([r['arms'][arm]['IoU'] for r in part])*100)})
  for c in ['prepared_frozen','real_frozen_fusedABC','randomA_TTA_fusedABC']:
   for sign in [-1,1]:
    part=sorted([r for r in rr if (r['arms'][main]['IoU']-r['arms'][c]['IoU'])*sign>1e-12],key=lambda r:abs(r['arms'][main]['IoU']-r['arms'][c]['IoU']),reverse=True)
    for j in sorted({0,min(1,len(part)-1),len(part)//2}) if part else []:
     r=part[j];cases.append({'source':r['source'],'target':r['target'],'row_PRIVATE':r['id_PRIVATE'],'control':c,'kind':'good' if sign>0 else 'failure','delta_IoU':r['arms'][main]['IoU']-r['arms'][c]['IoU'],'all_arms':r['arms'],'status':r['status'],'fits':r['fits'],'selection':'two extremes and median per direction/control/sign; postseal only'})
 gate=all(gains[k]>0 for k in controls[:6]);decision={'primary':main,'mIoU_percent':v(main),'gains_pp':gains,'passes_development_gates':gate,'robust_adoption':False,'decision':'candidate_requires_fixed_expansion' if gate else 'do_not_adopt_this_actual_LiDAR_interpolation_TTA_recipe','n':768,'no_experts':True,'source_retraining':False,'FINAL_access':False,'primary_not_switched':True};dump(H/'DECISION.json',decision);dump(H/'FIT_SUMMARY.json',fit);dump(H/'CONCENTRATION_SUMMARY.json',conc);dump(H/'SUBSET_RESULTS.json',subset);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases)
 lines=['# R106 — 实际邻观测点特征的原生匹配适应','','保留三份source100、source准备的R91双向结构、原A RGB/标定图、A文字坐标视角、1024个A空间seed和原A256。改变的是3D观测来源：真实注册B/C点云在A坐标中经过同一个冻结PointNet，再把原生pre-language特征对齐到固定A位置。未知区域保留A特征，没有新增专家网络或目标身份真值。','','每个额外输入16384点；同帧randomA对照使用两次独立16384点采样和同样前向次数。额外真实点不必比同帧重采样更好，新的体素可能来自噪声或配准误差。额外RGB bank没有与本轮混用。','', '## 观测前提','',json.dumps(rd(H/'COVERAGE_SUMMARY.json'),indent=2),'','## 固定方法与实施','', '3NN反距离插值；需要A位置与B实测回波距离≤.25m、第三邻seed≤1.6m（实际source SA2尺度）。已知零距离自对应使用one-hot，避免依赖无关邻点。首个工程检查暴露原半径条件误拒孤立自对应，attempt完整保留；在任何正式模型/任务评分之前按预定自对应含义修复，不改变非零距离插值。12条真实原生验证通过自对应、未知回退、原始A输出、实际写参数后完整前向和Jacobian。','','q=(p_A+p_B)/2来自同一冻结source准备模型；在A/B两视图拟合相同停止梯度目标，沿用18496参数的末端匹配投影和L2-LM10×6预算。各视图完整竞争残差冻结，因为匹配投影不改变它。C的PointNet/神经预测在所有拟合及验收后计算。C不足时记录拟合成本，最终TTA读出回到相同信息的冻结版本。候选ID始终是A锚点，不跨独立query编号硬对应。','','## 全体结果','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in aggregate:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 六方向改善与恶化','','| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---|---:|---:|']
 for x in rd(O/'ALL_PAIRED_ROWS.json'):lines.append(f"| {x['source']} | {x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 lines+=['','## 裁决','',json.dumps(decision,indent=2),'','## 边界','','全部768条保留，不能只报配准或更新成功子集。真正新增3D观测的贡献需胜过同帧重采样；参数更新需胜过同信息冻结特征融合和概率集成，也需检查错配。IoU变化不作为物体身份标签，案例只解释机制、不筛分母。不是Latte复现；共享场景、离线候选邻帧、历史开发暴露均限制泛化主张，见LITERATURE.md。FINAL未访问。若没过门槛，不通过扩大步数或调插值半径挽救。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
