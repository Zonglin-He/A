"""Root analysis after whole official score and independent array/mean audit."""
from common import *
import argparse
def export(ds):
 import numpy as np
 from full_score import cluster
 validate();plan=read(D/'root_analysis_plan_v1.json');assert plan['fingerprint']==digest({k:v for k,v in plan.items()if k!='fingerprint'})
 for f,h in plan['code_sha256'].items():assert sha(R/f)==h,f
 start=time.perf_counter();folder=D/ds;pub=P/ds;receipt=read(folder/'reporting_receipt.json');assert receipt['status']=='verified_whole_domain_all_arrays_parent_receipts_means_and_text_counts';assert sha(folder/'score_receipt.json')==receipt['score_sha256']and sha(folder/'metrics.json')==receipt['metrics_sha256'];rr=read(folder/'metrics.json')['private_rows'];assert len(rr)==COUNTS[ds]
 parents=sorted({r['parent']for r in rr});ix={p:i for i,p in enumerate(parents)};draw=np.random.default_rng(20261026).multinomial(len(parents),[1/len(parents)]*len(parents),size=10000)
 def group_delta(rows):
  sums=np.zeros(len(parents));counts=np.zeros(len(parents))
  for r in rows:j=ix[r['parent']];sums[j]+=100*(r['values']['FullMAP']['t']-r['values']['NoVisualUpdate']['t']);counts[j]+=1
  den=draw@counts;num=draw@sums;boot=np.divide(num,den,out=np.full(10000,np.nan),where=den>0);return sums.sum()/counts.sum()if counts.sum()else None,boot
 action=[r for r in rr if r['text_group']=='action_centric'];appearance=[r for r in rr if r['text_group']=='appearance_centric'];a,ba=group_delta(action);v,bv=group_delta(appearance);supported=all(len(x)>=plan['minimum_queries_per_semantic_group']and len({r['parent']for r in x})>=plan['minimum_parents_per_semantic_group']for x in [action,appearance]);both=np.isfinite(ba)&np.isfinite(bv);contrast=dict(status='supported_operational_text_strata'if supported else'insufficient_pure_appearance_or_action_support',action_queries=len(action),appearance_queries=len(appearance),difference_of_t_gain_pp=a-v if a is not None and v is not None else None,CI95=np.quantile((ba-bv)[both],[.025,.975]).tolist()if supported and both.any()else None,joint_parent_bootstrap=True,not_randomized_semantic_causal_inference=True)
 full_pairs={a+'-'+b:cluster(rr,a,b,[('t',100),('v',100),('vIoU_ge_03',100),('vIoU_ge_05',100),('boundary_MAE_seconds',1)])for a,b in PAIRS}
 with(pub/'all_text_signed_risk.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['anonymous_query','text_group','sequence_cues','timegroup','comparison','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','rescue_union_once','initial_joint_t','initial_native_t','final_t','prior_MAP_mass_increased'])
  for i,r in enumerate(rr,1):
   for a,b in PAIRS:
    dt=100*(r['values'][a]['t']-r['values'][b]['t']);dv=100*(r['values'][a]['v']-r['values'][b]['v']);w.writerow([f'Q{i:05d}',r['text_group'],r['explicit_sequence_and_multiple_action_cues'],r['duration_group'],a+'-'+b,dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5,dt>5 or dv>5,r['values']['NoVisualUpdate']['t'],r['values']['Source']['t'],r['values']['FullMAP']['t'],r['diagnostics']['prior_MAP_probability_increased']])
 catastrophic=dict(native_t_atleast05_to_final_zero=sum(r['values']['Source']['t']>=.5 and r['values']['FullMAP']['t']==0 for r in rr),joint_t_atleast05_to_final_zero=sum(r['values']['NoVisualUpdate']['t']>=.5 and r['values']['FullMAP']['t']==0 for r in rr),scope='Native-to-joint decoder changes isolated by NoVisualUpdate. Different counts do not identify cause of any single failure.')
 seal=read(folder/'sealed.json');cost={}
 for e in seal['files'].values():
  k=f"{e['worker']['pid']}:{e['worker']['startticks']}";seg=cost.setdefault(k,dict(queries=0,cached_conditional_elapsed_seconds=0.));seg['queries']+=1;seg['cached_conditional_elapsed_seconds']+=e['conditional_elapsed_seconds']
 report=read(pub/'report.json');summary=dict(status='root_temporal_risk_analysis_complete_pending_text_semantic_sensitivity_and_publication',queries=len(rr),operational_text_contrast=contrast,paired=full_pairs,catastrophic=catastrophic,cached_worker_cost=cost,Source_expert_inherited_cost_nonzero=True,new_Source=0,new_expert=0,new_optimizer=0,score_sha256=receipt['score_sha256'],audit_receipt_sha256=sha(folder/'reporting_receipt.json'))
 atomic(pub/'root_summary.json',summary);text=f'# Temporal feedback risk — {ds}\n\nAll {len(rr)} queries sealed before independent official scoring; all arrays and means independently rederived. Pure semantic groups are operational text-cue strata, not gold labels. All mixed/uncertain queries remain in overall results.\n\n|Temporal control|tIoU %|vIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|\n'
 for arm,m in report['means'].items():text+=f"|{arm}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|\n"
 text+='\nPrimary mechanistic contrast is FullMAP−NoVisualUpdate. All parent-paired t/v/boundary/R.3/R.5 pointwise95%CIs, all signed tails, short/long splits and initial-quality bins are retained in JSON/TSV. Pure-prior temperature must have identical MAP and boxes; it is an invariant calibration control. Fixed tau2/kernel3/mean evidence are not target-tuned.\n\n'
 text+=f"Action/appearance operational contrast: {contrast['status']}, action n={len(action)}, appearance n={len(appearance)}. HC appearance coverage cannot establish a two-category conclusion. Joint-parent group-delta CI only reported with at least30queries and10parents in each category. Query-stratum differences are observational and can reflect event length/initial difficulty.\n\n"
 text+=f"Severe native-good collapse count: {catastrophic['native_t_atleast05_to_final_zero']}; same-joint-decoder good-to-zero count: {catastrophic['joint_t_atleast05_to_final_zero']}. Native decoder shift and appearance feedback are different components. Prior mass/confidence increase associated with wrong intervals is not causal proof of self-reinforcement. Pre-text features need not encode the full language event; these contrasts also cannot prove they contain no action information.\n\nCached conditional replay elapsed is not process CPU or uncached deployment. Original Source/expert acquisition remains nonzero; no main method, paper, score-based parameter selection or paused task is changed.\n";(pub/'ANALYSIS.md').write_text(text)
 atomic(folder/'root_export_receipt_v1.json',dict(status=summary['status'],summary_sha256=sha(pub/'root_summary.json'),analysis_sha256=sha(pub/'ANALYSIS.md'),all_signed_text_SHA=sha(pub/'all_text_signed_risk.tsv'),root_analysis_plan_fingerprint=plan['fingerprint'],worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_GT_scoring=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);export(p.parse_args().dataset)
