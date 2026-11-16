"""Fixed update-scope result, including visibility and all positive/negative changes."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round104';S=H/'scores_attempt001';B=R/'crossplatform/round104';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;agg=rd(S/'AGGREGATE_ROWS.json');a={x['arm']:x for x in agg if x['group']=='all_six_equal'};v=lambda k:a[k]['mIoU_percent'];controls=['prepared_frozen','frozen_FC','direct_FC','direct_four','joint3_FC','late_shuffle_FC'];gains={k:v('late_FC')-v(k) for k in controls};gates=rd(H/'experiment/DESIGN_LOCK.json')['primary_gates'];dec={'primary':'late_FC','primary_mIoU_percent':v('late_FC'),'gains_pp':gains,'passes_prelocked_primary_gates':all(gains[k]>0 for k in gates),'n':768,'parameters':4160,'query_projection_parameters_in_control':18496,'scope_only_change_vs_R100':True,'robust_adoption':False,'development_exposed':True,'FINAL_access':False,'no_experts':True,'no_source_training':True,'primary_not_switched_to_F':True};(H/'DECISION.json').write_text(json.dumps(dec,indent=2)+'\n');lines=['# R104 — 几何绑定的RGB末端投影适应','','固定原R100均值G、三拟合视图、10次LM与每次6次阻尼验收、原A256及所有source状态，只把更新对象从18,496参数query投影换为已在source训练的4,160参数RGB末端投影。RGB编码器、双向交互、门控、query匹配、主干与竞争残差全部冻结。不是新增视觉网络，也不是引入专家。参数数目不同属于本更新结构差异，不能归为严格等参数量因果实验。','','先用12条真实样本验证原始和实际写参数后的完整模型重放、显式Jacobian、未映射候选每视图原始分数不变。新RGB输入通过原始点–像素对应重算，不反求源权重或猜像素。随后全部768条封存预测，独立审计后评分。C只在两种拟合完成后生成；是同物理观测的遮挡版本，不是新相机或独立场景。','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 pairs=rd(S/'ALL_PAIRED_ROWS.json');lines+=['','## 六方向固定比较','','| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---:|---:|']
 for x in pairs:lines.append(f"| {x['source']}→{x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 rr=[r for f in sorted(S.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];stats=[];case=[];coverage=[];switches=[]
 for r in rr:
  name=r['source']+'_to_'+r['target'];sid=r['id_PRIVATE']
  with np.load(B/name/'features_PRIVATE'/(sid+'.npz')) as z:known=z['0_known'];coverage.append(float(known.mean()))
  q0=r['arms']['prepared_frozen']['query_PRIVATE'];q1=r['arms']['late_FC']['query_PRIVATE']
  if q0!=q1:switches.append({'source':r['source'],'target':r['target'],'row_PRIVATE':sid,'before_known':bool(known[q0]),'after_known':bool(known[q1]),'delta_IoU':r['arms']['late_FC']['IoU']-r['arms']['prepared_frozen']['IoU']})
 for label in ['late','late_shuffle']:
  z=[r['fits'][label] for r in rr];stats.append({'fit':label,'n':len(z),**{k:float(np.mean([x[k] for x in z])) for k in ['initial_error','final_error','relative_error','parameter_L2','accepted_steps','attempted_steps']},'zero_update_n':sum(x['accepted_steps']==0 for x in z)})
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   for ctrl in ['prepared_frozen','direct_FC']:
    for sign in [-1,1]:
     z=sorted([r for r in rr if r['source']==src and r['target']==tgt and (r['arms']['late_FC']['IoU']-r['arms'][ctrl]['IoU'])*sign>1e-12],key=lambda r:abs(r['arms']['late_FC']['IoU']-r['arms'][ctrl]['IoU']),reverse=True)
     if z:
      for i in sorted({0,min(1,len(z)-1),len(z)//2}):
       r=z[i];case.append({'source':src,'target':tgt,'row_PRIVATE':r['id_PRIVATE'],'control':ctrl,'kind':'good' if sign>0 else 'failure','change_IoU':r['arms']['late_FC']['IoU']-r['arms'][ctrl]['IoU'],'all_arms':r['arms'],'fits':r['fits'],'selection':'two_extremes_and_median_per_direction_control_sign'})
 (H/'FIT_SUMMARY.json').write_text(json.dumps(stats,indent=2)+'\n');(H/'GOOD_FAILURE_CASES_PRIVATE.json').write_text(json.dumps(case,indent=2)+'\n');(H/'VISIBILITY_SWITCHES_PRIVATE.json').write_text(json.dumps(switches,indent=2)+'\n');vis={'mean_mapped_query_fraction':float(np.mean(coverage)),'top1_switches_vs_prepared_F':len(switches),'categories':[{'before_known':a,'after_known':b,'n':sum(x['before_known']==a and x['after_known']==b for x in switches),'better':sum(x['before_known']==a and x['after_known']==b and x['delta_IoU']>1e-12 for x in switches),'worse':sum(x['before_known']==a and x['after_known']==b and x['delta_IoU']< -1e-12 for x in switches)} for a in [False,True] for b in [False,True]],'invariant_is_raw_per_view_score_not_FC_ranking':True};(H/'VISIBILITY_SUMMARY.json').write_text(json.dumps(vis,indent=2)+'\n');lines+=['','## 机制与边界','',json.dumps(vis,indent=2),'','未映射候选的每视图原始分数保持，不等于最终FC排名保持：概率归一化及跨视图平均仍可引入间接变化。可见性不等于正确实例身份；IoU高低也不作为身份真值。所有未知、无RGB、零更新、负例仍计入总体。G、更新接受与停止均不使用GT。拟合下降、参数非零或保护不变量不是任务成功。','','## 裁决','',json.dumps(dec,indent=2),'','若不能超过同信息直接读出和打乱监督，不采纳本更新位置配方，也不按个例增加幅度或改变主端点。开发通过也仍需固定扩展验证，不能宣称稳定跨域TTA。文献边界见LITERATURE.md，本轮不是READ或xMUDA复现。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(dec)
if __name__=='__main__':main()
