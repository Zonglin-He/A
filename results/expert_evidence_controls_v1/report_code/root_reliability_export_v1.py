"""ROOT interpretation of completed source-CV and sealed target gate statistics."""
from pathlib import Path
import os,sys,json,time,argparse
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/expert_evidence_controls_v1'))
from common import D,P,read,sha,atomic,identity,np
from reliability_source_worker import C,check_source
def bootstrap(values,parents):
 pp=sorted(set(parents));ix={p:i for i,p in enumerate(pp)};s=np.zeros(len(pp));count=np.zeros(len(pp))
 for v,p in zip(values,parents):s[ix[p]]+=v;count[ix[p]]+=1
 n=len(pp);w=np.random.default_rng(20261022).multinomial(n,np.ones(n)/n,10000);b=(w@s)/(w@count);return dict(mean=float(np.mean(values)),CI95=np.quantile(b,[.025,.975]).tolist())
def rank_auc(scores,labels):
 scores=np.asarray(scores);labels=np.asarray(labels,bool);p=int(labels.sum());n=len(labels)-p
 if not p or not n:return None
 order=np.argsort(scores,kind='stable');sortedscore=scores[order];rank=np.empty(len(scores),float);start=0
 while start<len(scores):
  end=start+1
  while end<len(scores)and sortedscore[end]==sortedscore[start]:end+=1
  rank[order[start:end]]=(start+1+end)/2;start=end
 return float((rank[labels].sum()-p*(p+1)/2)/(p*n))
def main():
 check_source();receipt=read(C/'reporting_receipt.json');allrows=read(C/'target_metrics.json')['domains'];selection=read(C/'source_selection.json');totalchecks=0;root={}
 for ds,rr in allrows.items():
  pub=P/'reliability'/ds;rep=read(pub/'report.json');assert receipt['reports'][ds]==sha(pub/'report.json');checks=0
  for arm,means in rep['means'].items():
   for field,value in means.items():assert float(np.mean([r['values'][arm][field]for r in rr]))==value;checks+=1
  for group,pairs in rep['paired'].items():
   subset=[r for r in rr if group=='all'or r['duration_group']==group]
   for name,paired in pairs.items():
    a,b=name.split('-')
    for field,outfield in [('v','delta_v_pp'),('t','delta_t_pp')]:
     x=bootstrap([100*(r['values'][a][field]-r['values'][b][field])for r in subset],[r['parent']for r in subset]);y=paired['statistics'][outfield];assert abs(x['mean']-y['mean'])<1e-11 and np.max(abs(np.array(x['CI95'])-y['CI95']))<1e-11;checks+=1
  for rule,audited in rep['harm_detection_AUROC'].items():
   scores=[r['risks'][rule]for r in rr]
   for name,y in audited.items():
    if name=='offline_evaluation_only':continue
    labels=[r['values']['Full']['v']<r['values']['Source']['v']-1e-12 for r in rr]if name=='any_v_harm'else[r['values']['Full']['v']-r['values']['Source']['v']< -.05 or r['values']['Full']['t']-r['values']['Source']['t']< -.05 for r in rr];x=rank_auc(scores,labels)
    if x is None:assert y['AUROC']is None
    else:assert abs(x-y['AUROC'])<1e-12
    checks+=1
  assert rep['risk_coverage']['RiskGate']['rejected_to_Source']==rep['risk_coverage']['RandomGate']['rejected_to_Source'];cfg=selection['domains'][ds];selected=next(x for x in rep['source_four_parentfold_OOF']if x['name']==cfg['selected_name']);dv=rep['paired']['all']['RiskGate-Full']['statistics']['delta_v_pp'];dt=rep['paired']['all']['RiskGate-Full']['statistics']['delta_t_pp'];random=rep['paired']['all']['RiskGate-RandomGate']['statistics']['delta_v_pp'];risk=rep['risk_coverage']['RiskGate'];baseline=rep['risk_coverage']['Full']
  if cfg['selected_name']=='alwaysFull':conclusion='Source-only selection retained alwaysFull: no usable query gate was established by these finite source rules. Target RiskGate and RandomGate are exactly Full; do not replace the selected rule after target scoring.'
  elif dv['CI95'][1]<0:conclusion='Selected source-only gate lowers target vIoU in this panel. Any reduction in severe corrections must be reported together with lost useful corrections and the negative mean trade-off; no target reselection.'
  elif dv['CI95'][0]>0:conclusion='Selected source-only gate increases target mean vIoU in this exposed panel. Assess severe corrections and matched random coverage jointly; this is bounded evidence for this query gate, not certified expert correctness or automatic production adoption.'
  else:conclusion='Selected source-only gate has an unresolved target mean vIoU direction under the paired CI. Report harm avoidance and forgone gains with the uncertainty; no strong reliable-correction claim.'
  text=[f'# ROOT reliability analysis: {ds}','',conclusion,'',f"Source calibration:36 queries/{cfg['source_parent_count']}parent clusters, four grouped folds; selected **{cfg['selected_name']}** at nominal source rejected fraction {cfg['selected_coverage']}. Selected OOF delta vs Full: {selected['OOF_delta_vs_Full']}.",'','|Policy|tIoU|vIoU|R.3|R.5|Rejected to Source|Severe t/v uniononce vs Source|','|---|---:|---:|---:|---:|---:|---:|']
  for a in ['Source','Full','RiskGate','RandomGate']:text.append('|'+a+'|'+'|'.join(f"{100*rep['means'][a][field]:.6f}"for field in ['t','v','R03','R05'])+f"|{rep['risk_coverage'][a]['rejected_to_Source']}|{rep['risk_coverage'][a]['severe_tv_union_once']}|")
  text+=['',f'RiskGate-Full delta vIoU {dv}; delta tIoU {dt}.',f'RiskGate-RandomGate delta vIoU {random}.',f"Full severe t/v uniononce {baseline['severe_tv_union_once']}→gate {risk['severe_tv_union_once']}; negative v corrections avoided {risk['negative_v_corrections_avoided']}; severe corrections averted {risk['severe_corrections_averted']}; beneficial severe rescues forgone {risk['beneficial_severe_rescues_foregone']}; positive v gain retained fraction {risk['positive_v_gain_retained_ratio']}.",'',f"Frozen signal harm detection (offline evaluation only): {json.dumps(rep['harm_detection_AUROC'],sort_keys=True)}"]
  for group in ['short_le5s','long_gt5s']:text.extend(['',f"{group}: {json.dumps(rep['paired'][group],sort_keys=True)}"])
  text+=['',f'ROOT independently recomputed {checks} means, paired parent CIs and rank-sum AUROC values. The independent all-array/input and normalized-pixel score audits precede this interpretation.','', 'GT is required for source calibration and offline target evaluation, not inference. Expert confidence/LOO consistency/temporal Source confidence are risk proxies; wrong but consistent expert identity and confident Source errors remain possible. These results do not identify which individual anchor is wrong. No target GT selected thresholds, rules, seed, confidence weights, or coverage. All nine predeclared source-fit controls are retained, including negative/null outputs.','', 'A gate selects the entire original Source or entire unchanged Full tube. This is not hard-anchor fixing, temporal-only correction, anchor filtering or action-semantic reasoning. Historical target panel exposure and opposite-direction source exposure are disclosed. Production and manuscript unchanged.','', 'Source72 teacher supplement preserves all input/native outputs. New Source/optimizer calls0; new ReferDINO logical queries72/observations576, one resident expert instance. Target reuses original experts and complete Source/Full. Cached signal wall and source experimental replay clocks include IO/checkpoints; they are not end-to-end deployment latency or proven savings.']
  (pub/'ROOT_ANALYSIS.md').write_text('\n'.join(text)+'\n');record=dict(status='ROOT_all_source_selection_target_signed_means_CI_AUC_shortlong_severe_rescues_and_nulls_reviewed',queries=128,parents=128,independent_checks=checks,selected_source_rule=cfg['selected_name'],report_sha256=sha(pub/'report.json'),reporting_receipt_sha256=sha(C/'reporting_receipt.json'),new_Source=0,new_GT=0,production_changed=False,publication_complete=False,worker=identity());atomic(pub/'ROOT_receipt.json',record);root[ds]=record;totalchecks+=checks
 costs=read(P/'reliability/all_actual_costs.json');assert len(costs['source_queries'])==72 and sum(x['new_ReferDINO_queries']for x in costs['source_queries'])==72;load={}
 for q in costs['source_queries']:
  key=str(q['teacher']['pid'])+':'+str(q['teacher']['startticks']);load[key]=max(load.get(key,0.),q['model_load_seconds'])
 summary=dict(status='both128_reliability_ROOT_complete_pending_publication',domains=root,independent_checks=totalchecks,unique_teacher_load_seconds=load,source_expert_logical_queries=72,source_expert_observations=576,target_queries=256,new_Source=0,new_optimizer=0,whole_propagation_also_ROOT_complete=all((D/'propagation'/ds/'ROOT_receipt.json').exists()for ds in root),cost_sha256=sha(P/'reliability/all_actual_costs.json'),old_Source_acquisition_not_zero=True,nested_costs_not_additive=True,not_deployment_latency=True,time=time.time());atomic(C/'ROOT_completion.json',summary);atomic(P/'reliability/ROOT_completion.json',summary);print(json.dumps({ds:dict(selected=x['selected_source_rule'],checks=x['independent_checks'])for ds,x in root.items()}),flush=True)
if __name__=='__main__':main()
