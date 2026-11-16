"""Expose all aggregate, directional and changed-case results; never select task outputs."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round102';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 sections=[];decisions={};allcases=[]
 for label,D in [('DEV768',H),('FULL18620',H/'full')]:
  S=D/'scores_attempt001';assert rd(D/'SCALAR_AUDIT.json')['state'].startswith('PASS_');agg=rd(S/'AGGREGATE_ROWS.json');a={x['arm']:x for x in agg if x['group']=='all_six_equal'};v=lambda k,aggregate=a:aggregate[k]['mIoU_percent'];controls=['prepared_frozen','direct_FC','scaledmean_FC','shufflecommon_FC'];gains={k:v('signcommon_FC')-v(k) for k in controls};decisions[label]={'primary':'signcommon_FC','mIoU_percent':v('signcommon_FC'),'gains_pp':gains,'passes_primary_comparisons':all(x>0 for x in gains.values()),'robust_adoption':False};rows=rd(S/'ALL_METHOD_ROWS.json');pairs=rd(S/'ALL_PAIRED_ROWS.json');sections+=['','## '+label,'','| 方向 | n | 原始 | 冻结 | 旧G直接FC | 共识G直接FC | 同幅度旧G | 打乱共识G |','|---|---:|---:|---:|---:|---:|---:|---:|']
  for src in ['waymo','drone','quad']:
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    z={x['arm']:x for x in rows if x['source']==src and x['target']==tgt};sections.append('| '+src+'→'+tgt+' | '+str(z['signcommon_FC']['n'])+' | '+' | '.join(f"{z[k]['mIoU_percent']:.6f}" for k in ['original_source','prepared_frozen','direct_FC','signcommon_FC','scaledmean_FC','shufflecommon_FC'])+' |')
  sections+=['','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
  for x in agg:sections.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
  sections+=['','| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---:|---:|']
  for x in pairs:sections.append(f"| {x['source']}→{x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
  rr=[r for f in sorted(S.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];changed=[]
  for r in rr:
   a=r['arms'];diff=a['signcommon_FC']['IoU']-a['direct_FC']['IoU']
   if abs(diff)>1e-12:changed.append({'source':r['source'],'target':r['target'],'row_PRIVATE':r['id_PRIVATE'],'kind':'good' if diff>0 else 'failure','change_IoU':diff,'method_IoU':a['signcommon_FC']['IoU'],'control_IoU':a['direct_FC']['IoU'],'all_arms':a})
  case=[]
  for src in ['waymo','drone','quad']:
   for tgt in ['waymo','drone','quad']:
    for kind in ['good','failure']:
     z=sorted([r for r in changed if r['source']==src and r['target']==tgt and r['kind']==kind],key=lambda x:abs(x['change_IoU']),reverse=True)
     if z:
      for i in sorted(set([0,min(1,len(z)-1),len(z)//2])):case.append(z[i])
  for i,c in enumerate(case):c.update(case='CASE'+str(i+1).zfill(4),selection='two_extremes_and_median_per_direction_sign_descriptive_only',scope=label)
  dump(D/'GOOD_FAILURE_CASES_PRIVATE.json',case);allcases+=case
  contributions=[]
  for r in changed:
   count=next(x['n'] for x in rows if x['source']==r['source'] and x['target']==r['target'] and x['arm']=='signcommon_FC');contributions.append(r['change_IoU']*100/(6*count))
  dump(D/'CHANGE_SUMMARY.json',{'n':len(rr),'better_n':sum(r['change_IoU']>0 for r in changed),'worse_n':sum(r['change_IoU']<0 for r in changed),'same_n':len(rr)-len(changed),'largest_positive_macro_contribution_pp':max([0.]+contributions),'largest_negative_macro_contribution_pp':min([0.]+contributions),'net_macro_pp':v('signcommon_FC')-v('direct_FC')})
 stats=rd(H/'CONTRAST_STATS_PRIVATE.json');summary={k:float(np.mean([r[k] for r in stats])) for k in ['same_sign_fraction','common_to_mean_L2_ratio','contrast_cosine','single_source_top1_agrees']};dump(H/'CONTRAST_SUMMARY.json',{'state':'NO_GT_INPUT_CONTRAST_STATISTICS_DEV768','n':len(stats),**summary});decision={'state':'R102_FIXED_COUNTERFACTUAL_CONSENSUS_DIRECT_READOUT_EVALUATED','scopes':decisions,'robustly_adopted':False,'TTA_run':False,'FINAL_access':False,'external_experts':False,'source_training':False,'full_historically_exposed':True,'next_action':'consider_same_signal_parameter_comparison_only_if_full_primary_passes' if decisions['FULL18620']['passes_primary_comparisons'] else 'stop_this_consensus_recipe_no_parameter_fit_or_threshold_grid'};dump(H/'DECISION.json',decision)
 lines=['# R102 — 两次错配绑定差分的一致性检验','','R101更好拟合固定监督仍损害grounding，因此本轮只改变监督聚合，不修改模型、候选或更新器。每次错配绑定先对三视图真实–错配差分求平均，再对候选居中。两份差分同号时保留较小绝对值，异号记零，最后再居中。这是待验证的保守证据聚合，不把一致当身份真值。没有新的模型前向或参数更新。','','先锁定开发768；观察到正信号后，原样固定扩展到历史R98同18620有向输入。完整集合共9310目标表达，各进入两个非自身source方向；不是18620独立场景。FINAL始终排除。主端点始终FC，禁止看到全量后改选F/C。所有负例、UNKNOWN、失败和零IoU保留。','','同信息对照包括原G均值、两份单独差分、相同L2范数的原G整体缩放和打乱query对应。范数缩放排除“仅仅减小证据”的解释。开发均值差分余弦'+f"{summary['contrast_cosine']:.4f}"+'，同号比例'+f"{100*summary['same_sign_fraction']:.2f}%"+'，直接F第一名一致比例'+f"{100*summary['single_source_top1_agrees']:.2f}%"+'。这些只度量扰动稳定性，不是正确率。']+sections
 sp=rd(H/'full/scores_attempt001/STRATUM_PAIRED_ROWS.json');lines+=['','## 扩大部分单列','','| 部分 | 对照 | 等权 Δ pp | 改善/恶化 |','|---|---|---:|---:|']
 for part in ['DEV768','EXPANSION_ONLY']:
  for ctrl in ['prepared_frozen','direct_FC','scaledmean_FC','shufflecommon_FC','joint3_FC']:
   z=[r for r in sp if r['stratum']==part and r['control']==ctrl];lines.append(f"| {part} | {ctrl} | {np.mean([r['gain_pp'] for r in z]):+.6f} | {sum(r['better'] for r in z)}/{sum(r['worse'] for r in z)} |")
 lines+=['','## 文献边界','','[Reg-TTA3D原文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05971.pdf)用噪声稳定性筛伪框，但原任务是检测、监督定位回归。本轮只借鉴先检查稳定性的研究逻辑；两次反事实差分取同号最小值是这里的实验设计，不是其NPG复现。稳定和正确必须由本轮任务对照区分。','','## 裁决','',json.dumps(decision,indent=2,ensure_ascii=False),'','好坏变化按固定规则收集，IoU变化不等于身份标签；不得据它们筛测试输入。若全量未保持开发优势，不以换阈值、增减场景或选择更好的单错配结果挽救。本轮没有运行TTA；直接证据不成立时，不让优化器承担监督纠错。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(decision)
if __name__=='__main__':main()
