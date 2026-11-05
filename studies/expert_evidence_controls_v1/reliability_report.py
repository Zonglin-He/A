"""All signed source-CV/target risk-coverage reports; independent report audit."""
from common import *
from reliability_source_worker import C,check_source,source_cases
from reliability_study import POLICIES,RULES,COV,candidates,cutoff,reject,keyhash,risks
from report import stats
import argparse,csv
PUB=P/'reliability'
def pair(rr,a,b):
 out=[]
 for r in rr:
  x,y,s=r['values'][a],r['values'][b],r['values']['Source'];sx=(x['t']-s['t']< -.05 or x['v']-s['v']< -.05);sy=(y['t']-s['t']< -.05 or y['v']-s['v']< -.05);out.append(dict(parent=r['parent'],delta_v_pp=100*(x['v']-y['v']),delta_t_pp=100*(x['t']-y['t']),delta_severe_Source_relative_union_once_pp=100*(int(sx)-int(sy))))
 return stats(out,['delta_v_pp','delta_t_pp','delta_severe_Source_relative_union_once_pp'])
def auc(scores,labels,parents):
 labels=np.asarray(labels,bool);scores=np.asarray(scores,float);pos=np.flatnonzero(labels);neg=np.flatnonzero(~labels)
 if not len(pos)or not len(neg):return dict(AUROC=None,CI95=None,positives=len(pos),negatives=len(neg),reason='oneclass_unscorable')
 assert len(set(parents))==len(parents);kernel=(scores[pos,None]>scores[None,neg]).astype(float)+.5*(scores[pos,None]==scores[None,neg]);point=float(kernel.mean());n=len(scores);w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,10000);den=w[:,pos].sum(1)*w[:,neg].sum(1);num=((w[:,pos]@kernel)*w[:,neg]).sum(1);b=num[den>0]/den[den>0]
 return dict(AUROC=point,CI95=np.quantile(b,[.025,.975]).tolist(),positives=len(pos),negatives=len(neg),bootstrap_valid=len(b))
def derive_report():
 assert read(C/'target_score_audit.json')['metrics_sha256']==sha(C/'target_metrics.json');selection=read(C/'source_selection.json');source=read(C/'source_metrics.json')['domains'];target=read(C/'target_metrics.json')['domains'];out={}
 for ds,rr in target.items():
  assert len(rr)==128 and len({r['parent']for r in rr})==128;cfg=selection['domains'][ds];means={a:{f:float(np.mean([r['values'][a][f]for r in rr]))for f in ['v','t','R03','R05']}for a in POLICIES};pairs={g:{a+'-'+b:pair([r for r in rr if g=='all'or r['duration_group']==g],a,b)for a,b in [('RiskGate','Full'),('RiskGate','Source'),('RiskGate','RandomGate'),('Full','Source')]}for g in ['all','short_le5s','long_gt5s']};risksummary={}
  for a in POLICIES:
   harmed_v=severe_t=severe_v=severe_union=rescues=avoided=averted=lost=0;retained=totalgain=0.;rows_for_ci=[]
   for r in rr:
    s,f,x=r['values']['Source'],r['values']['Full'],r['values'][a];dt=x['t']-s['t'];dv=x['v']-s['v'];ft=f['t']-s['t'];fv=f['v']-s['v'];flag=r['reject_flags'][a];harmed_v+=dv< -1e-12;severe_t+=dt< -.05;severe_v+=dv< -.05;severe_union+=dt< -.05 or dv< -.05;rescues+=dt>.05 or dv>.05;avoided+=flag and fv< -1e-12;averted+=flag and(ft< -.05 or fv< -.05);lost+=flag and(ft>.05 or fv>.05);totalgain+=max(fv,0.);retained+=max(dv,0.);rows_for_ci.append(dict(parent=r['parent'],vIoU_pp=100*x['v'],tIoU_pp=100*x['t'],severe_union_percent=100.*(dt< -.05 or dv< -.05)))
   risksummary[a]=dict(rejected_to_Source=sum(r['reject_flags'][a]for r in rr),harmed_v_vs_Source=harmed_v,severe_t_vs_Source=severe_t,severe_v_vs_Source=severe_v,severe_tv_union_once=severe_union,rescue_tv_union_once=rescues,negative_v_corrections_avoided=avoided,severe_corrections_averted=averted,beneficial_severe_rescues_foregone=lost,positive_v_gain_retained_ratio=retained/totalgain if totalgain>0 else None,queryparentCI=stats(rows_for_ci,['vIoU_pp','tIoU_pp','severe_union_percent']))
  aurocs={}
  for rule in RULES:
   scores=[r['risks'][rule]for r in rr];parents=[r['parent']for r in rr];harm=[r['values']['Full']['v']<r['values']['Source']['v']-1e-12 for r in rr];severe=[r['values']['Full']['v']-r['values']['Source']['v']< -.05 or r['values']['Full']['t']-r['values']['Source']['t']< -.05 for r in rr];aurocs[rule]=dict(any_v_harm=auc(scores,harm,parents),severe_tv_union=auc(scores,severe,parents),offline_evaluation_only=True)
  sourcecfgs=[]
  for c in cfg['all_configs']:
   byid={r['id']:r for r in source[ds]};delta=[dict(parent=x['parent'],delta_v_pp=100*(x['values']['v']-byid[x['id']]['values']['Full']['v']),delta_t_pp=100*(x['values']['t']-byid[x['id']]['values']['Full']['t']))for x in c['OOF_cases']];sourcecfgs.append(dict(name=c['name'],mean_v=c['mean_v'],mean_t=c['mean_t'],rejected=c['rejected'],OOF_delta_vs_Full=stats(delta,['delta_v_pp','delta_t_pp'])))
  assert risksummary['RiskGate']['rejected_to_Source']==risksummary['RandomGate']['rejected_to_Source']
  out[ds]=dict(queries=128,parents=128,selected_source_rule=cfg['selected_name'],source_parents=cfg['source_parent_count'],source_queries=36,source_four_parentfold_OOF=sourcecfgs,source_rule_only=True,means=means,paired=pairs,risk_coverage=risksummary,harm_detection_AUROC=aurocs,CI_scope='queryequal10000parentclusters seed20261022 pointwise exposedcohort; targetlabels onlyafter all256decisions sealed; source labels calibration only',gain_retention_scope='sum of positive vIoU gains over Source; time/severe benefits separately reported; no causal error-quality certification')
 return out
def verify_source_selection():
 selection=read(C/'source_selection.json');source=read(C/'source_metrics.json')['domains'];checks=0
 for ds,ss in source.items():
  cfg=selection['domains'][ds];byid={r['id']:r for r in ss};parents=sorted({r['parent']for r in ss},key=lambda p:keyhash(ds,p,'source_fold'));fold={p:i%4 for i,p in enumerate(parents)};assert fold==cfg['fold_by_parent']
  for c in cfg['all_configs']:
   assert len(c['OOF_cases'])==36
   for x in c['OOF_cases']:
    r=byid[x['id']];train=[q for q in ss if fold[q['parent']]!=x['fold']];flag=reject(risks([q['signals']for q in train],r['signals'])[c['rule']],keyhash(r['dataset'],r['id']),cutoff(train,c['rule'],c['coverage']))if c['rule']else False;assert flag==x['reject_to_Source']and x['values']==r['values']['Source'if flag else'Full'];checks+=1
   assert c['mean_v']==float(np.mean([x['values']['v']for x in c['OOF_cases']]))
  best=max(cfg['all_configs'],key=lambda c:c['mean_v']);assert best['name']==cfg['selected_name'];assert cfg['cutoff']==(list(cutoff(ss,best['rule'],best['coverage']))if best['rule']else None)
 return checks
def export():
 check_source();torch.set_num_threads(1);checks=verify_source_selection();reports=derive_report()
 for ds,rep in reports.items():
  pub=PUB/ds;atomic(pub/'report.json',rep);rr=read(C/'target_metrics.json')['domains'][ds];pm={p:f'P{i:04}'for i,p in enumerate(sorted({r['parent']for r in rr}),1)}
  with(pub/'all_signed_scalars.tsv').open('w')as f:
   writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['query','parent','duration_group','policy','reject_to_Source','tIoU','vIoU','R03','R05','delta_t_vs_Source_pp','delta_v_vs_Source_pp','severe_tv_union_once'])
   for i,r in enumerate(rr,1):
    for a in POLICIES:
     x,s=r['values'][a],r['values']['Source'];dt=100*(x['t']-s['t']);dv=100*(x['v']-s['v']);writer.writerow([f'Q{i:05}',pm[r['parent']],r['duration_group'],a,r['reject_flags'][a],100*x['t'],100*x['v'],100*x['R03'],100*x['R05'],dt,dv,dt< -5 or dv< -5])
  selection=read(C/'source_selection.json')['domains'][ds];text=[f'# {ds}: source-only reliability gate, target128 sealed decisions','',f"Selected source-only rule: **{selection['selected_name']}**. Source36 queries/{selection['source_parent_count']}parents,4parentfold OOF; target128queries/parents. No target configuration selection.",'','|Policy|tIoU|vIoU|R.3|R.5|Reject to Source|Severe t/v uniononce vs Source|','|---|---:|---:|---:|---:|---:|---:|']
  for a in ['Source','Full','RiskGate','RandomGate']:text.append('|'+a+'|'+'|'.join(f"{100*rep['means'][a][f]:.6f}"for f in ['t','v','R03','R05'])+f"|{rep['risk_coverage'][a]['rejected_to_Source']}|{rep['risk_coverage'][a]['severe_tv_union_once']}|")
  text+=['',f"RiskGate-Full paired all: {rep['paired']['all']['RiskGate-Full']}",f"RiskGate-RandomGate paired all: {rep['paired']['all']['RiskGate-RandomGate']}",f"RiskGate retained/avoided/foregone: {rep['risk_coverage']['RiskGate']}",'','All nine prespecified source-cutoff coverage points and all signed short/long/negative/severe/rescue outcomes remain in report.json and scalars. AUROC is offline target evaluation of frozen risk signals, never a threshold selection criterion. An alwaysFull source selection means no gate is deployed; it is retained as a null outcome, not replaced after seeing target scores.','', 'This tests query-level rejection of the whole correction. It does not establish individual anchor filtering, time-only protection, action semantics, or calibrated expert correctness. Source confidence is temporal confidence, not correct boxes. Source data were previously exposed in the opposite transfer direction; target panel is historically exposed. Current softanchors and production unchanged.','', 'New72 source-domain ReferDINO calls use the correct saved Source checkpoint native eight-position policy; no new Source/optimizer. Target study reuses Source/Full arrays and experts. Signal/replay clocks are conditional cached experiment costs, not end-to-end deployment latency or demonstrated savings.']
  (pub/'ANALYSIS.md').write_text('\n'.join(text)+'\n')
 sourcefiles=[]
 for target in ['hc2','vidstg']:
  for i,(key,e)in enumerate(read(C/'source'/target/'sealed.json')['files'].items(),1):sourcefiles.append(dict(target_model=target,query=f'S{i:04}',teacher=e['teacher'],worker=e['worker'],expert_request_wall_seconds=e['request_wall_seconds'],native_expert_cost=e['native_expert_cost'],model_load_seconds=e['model_load_seconds'],query_active_wall_seconds=e['query_active_wall_seconds'],new_Source=0,new_ReferDINO_queries=1))
 targetfiles=[]
 for ds in ['hc2','vidstg']:
  for i,e in enumerate(read(C/'target'/ds/'sealed.json')['files'].values(),1):targetfiles.append(dict(dataset=ds,query=f'Q{i:05}',cached_signal_wall_seconds=e['cached_signal_wall_seconds'],new_Source=0,new_expert=0))
 atomic(PUB/'all_actual_costs.json',dict(source_queries=sourcefiles,target_signal_queries=targetfiles,source_expert_query_count=72,new_Source=0,new_optimizer=0,Source_and_old_targetexpert_acquisition_nonzero=True,model_load_counted_by_unique_PIDstart=True,nested_clocks_not_additive=True,cached_signal_wall_not_deployment_latency=True));atomic(C/'report_export_receipt.json',dict(status='all_source_CV_target_positive_negative_riskcoverage_AUROC_tails_cost_exported_pending_report_audit_ROOT',source_selection_checks=checks,reports={ds:sha(PUB/ds/'report.json')for ds in reports},cost_sha256=sha(PUB/'all_actual_costs.json'),worker=identity()));print('reliability all signed reports/costs exported',flush=True)
def audit():
 check_source();torch.set_num_threads(1);checks=verify_source_selection();reports=derive_report();receipt=read(C/'report_export_receipt.json')
 for ds,rep in reports.items():assert rep==read(PUB/ds/'report.json')and receipt['reports'][ds]==sha(PUB/ds/'report.json')
 costs=read(PUB/'all_actual_costs.json');assert len(costs['source_queries'])==72 and len(costs['target_signal_queries'])==256 and sum(x['new_ReferDINO_queries']for x in costs['source_queries'])==72
 ack=dict(status='sourcefold_selection_all_targetmeans_parentCI_riskcoverage_AUC_and_cost_independently_verified_PENDING_ROOT',source_selection_checks=checks,target_score_audit_sha256=sha(C/'target_score_audit.json'),source_score_audit_sha256=sha(C/'source_score_audit.json'),source_selection_sha256=sha(C/'source_selection.json'),target_decisions_seal_sha256=sha(C/'target_sealed.json'),reports={ds:sha(PUB/ds/'report.json')for ds in reports},export_receipt_sha256=sha(C/'report_export_receipt.json'),worker=identity());atomic(C/'reporting_receipt.json',ack);atomic(PUB/'reporting_receipt.json',ack);print('reliability source selection/target/stat all audited',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('task',choices=['export','audit']);x=a.parse_args();globals()[x.task]()
