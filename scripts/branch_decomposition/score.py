"""Independent CPU evaluation after cached branch predictions seal."""
import os,sys,json,csv
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,torch
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.branch_decomposition.run import OUT,PUBLIC,read,write,sha,validate,utility
from scripts.interval_pool.score import score_boxes

def auc(scores,labels):
 scores=np.asarray(scores);labels=np.asarray(labels,bool);pos=int(labels.sum());neg=len(labels)-pos
 return None if not pos or not neg else float((rankdata(scores)[labels].sum()-pos*(pos+1)/2)/(pos*neg))
def evaluate():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate(p);seal=read(OUT/'sealed.json');assert seal['plan_sha256']==sha(OUT/'plan.json')and seal['GT_read']is False and seal['complete448_samples']==1
 assert sha(OUT/'label_free_audit.json')==seal['audit_sha256'];old=read(ROOT/'results/tube_fusion_hc64/report.json');assert sha(ROOT/'results/tube_fusion_hc64/report.json')==p['old_report_sha256'];audit=read(OUT/'label_free_audit.json')['rows'];rows=[]
 for previous,a in zip(old['queries_anonymous'],audit):
  assert previous['query']==a['query'];B=previous['values']['Global6'];S=previous['values']['Strong_Reference'];dt=100*(S['t']-B['t']);dv=100*(S['v']-B['v']);row=dict(a,Global6_tIoU=B['t'],Fusion_tIoU=S['t'],delta_tIoU_pp=dt,delta_vIoU_pp=dv,delta_fixed_space_vIoU_pp=100*(previous['fixed_spatial']['Strong_Reference']['v']-previous['fixed_spatial']['Global6']['v']),start_error_change_seconds=S['start_error_seconds']-B['start_error_seconds'],end_error_change_seconds=S['end_error_seconds']-B['end_error_seconds'],length_change_seconds=S['interval_length_seconds']-B['interval_length_seconds'],rescued_half=B['t']<.5<=S['t'],harmed_half=S['t']<.5<=B['t'],GT_length_seconds=B['GT_length_seconds']);rows.append(row)
 labels=ROOT/p['labels']['path'];assert sha(labels)==p['labels']['sha256'];truth={v['id']:v for v in [json.loads(l)for l in labels.read_text().splitlines()if l.strip()]};records={v['id']:v for v in [json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()]};cases=[];checks=0;error=0.
 for entry in seal['files']:
  item=next(x for x in p['items']if x['anonymous']==entry['query']);assert sha(OUT/entry['path'])==entry['sha256']and sha(OUT/entry['posterior_path'])==entry['posterior_sha256'];z=np.load(OUT/entry['path']);qs=np.load(OUT/entry['posterior_path']);gt=truth[item['id']];rec=records[item['id']];boxes={int(k):v for k,v in gt['boxes'].items()};metrics={};fixed={};post={}
  for arm in ['Source','Only224six','Only448','Fusion']:
   metrics[arm],e=score_boxes(z[arm],rec,boxes,z[arm+'_interval'].tolist(),gt['interval'],z['positions8'].tolist());error=max(error,e);checks+=1;fixed[arm],e=score_boxes(z['FixedSpatial'],rec,boxes,z[arm+'_interval'].tolist(),gt['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
   if arm!='Source':
    q=qs[arm];i,j=np.nonzero(q);ids=qs['ids'];a,b=gt['interval'];overlap=np.maximum(0,np.minimum(b,ids[j]+1)-np.maximum(a,ids[i]));t=overlap/((b-a)+(ids[j]+1-ids[i])-overlap);n=q[q>0];post[arm]=dict(entropy=float(-(n*np.log(n)).sum()),posterior_mass_tIoU_ge_half=float(q[i,j][t>=.5].sum()),expected_tIoU_against_GT=utility(ids,q,gt['interval']))
  oldrow=next(r for r in old['queries_anonymous']if r['query']==entry['query'])
  for arm,before in [('Only224six','Global6'),('Fusion','Strong_Reference')]:
   for key in ['v','t','start_error_seconds','end_error_seconds']:assert abs(metrics[arm][key]-oldrow['values'][before][key])<2e-10
  # Actual448 utility contributions for this single recoverable case only.
  gspan=z['Only224six_interval'];hspan=z['Only448_interval'];sspan=z['Fusion_interval'];contribution={}
  for name,span in [('224_decision',gspan),('448_decision',hspan),('fusion_decision',sspan)]:contribution[name]={arm:utility(qs['ids'],qs[arm],span)for arm in ['Only224six','Only448','Fusion']}
  cases.append(dict(query=entry['query'],metrics=metrics,fixed_spatial=fixed,posterior_diagnostics=post,decision_utility_contributions=contribution))
 summaries={}
 for name,mask in dict(all=np.ones(64,bool),rescued_half=[r['rescued_half']for r in rows],harmed_half=[r['harmed_half']for r in rows],benefit_over1pp=[r['delta_tIoU_pp']>1 for r in rows],harm_over1pp=[r['delta_tIoU_pp'] < -1 for r in rows]).items():
  rr=[r for r,m in zip(rows,mask)if m];summaries[name]=dict(queries=len(rr),means={k:float(np.mean([r[k]for r in rr]))for k in ['delta_tIoU_pp','delta_vIoU_pp','native_MBR_utility','normalized_entropy','candidate_start_spread_fraction','candidate_end_spread_fraction','required448_relative_preference_lower_bound']})
 benefits=np.asarray([r['delta_tIoU_pp']>1 for r in rows]);harms=np.asarray([r['delta_tIoU_pp'] < -1 for r in rows]);extreme=benefits|harms
 diagnostic_auc={k:auc(np.asarray([r[k]for r in rows])[extreme],benefits[extreme])for k in ['normalized_entropy','candidate_start_spread_fraction','candidate_end_spread_fraction']}
 report=dict(status='completed_available_cache_decomposition_with_explicit_coverage_gap',coverage=dict(total=64,complete224_six=64,complete448=1,missing448=63,missing_reason='Old HC64 reference/student/prediction paths saved Strong intervals but not Strong/448 posterior or448 hidden/logits. A final MBR decision does not determine the distribution.',one_case_is_engineering_sample_not_representative=True),branch_means64=dict(Only224six=old['means']['Global6'],Fusion=old['means']['Strong_Reference'],Only448=None),one_case=cases,summary=summaries,rescued=[r for r in rows if r['rescued_half']],harmed=[r for r in rows if r['harmed_half']],all_signed=rows,uncertainty_diagnostic=dict(queries=int(extreme.sum()),AUC_benefit_over1pp_vs_harm_over1pp=diagnostic_auc,posthoc_only_no_gate_fitted=True,warning='Few rescue/harm cases; no threshold or call policy selected.'),proof=dict(all_new_predictions_sealed_before_labels=True,plan_sha256=sha(OUT/'plan.json'),seal_sha256=sha(OUT/'sealed.json'),official_dense_checks=checks,official_dense_maxerror=error,network_head_GPU_expert_updates=0,complete448_only_reconstructed_when_fullStrong_distribution_exists=True,full_run_unchanged=True),decision=dict(new_lightweight_algorithm_validated=False,full64_branch_question_complete=False,no_weight_sweep=True,no_production_change=True,reason='Cannot distinguish448-only capability from multiscale complementarity across64 with63 missing posteriors; do not fabricate a64-case comparison.'))
 validate(p);assert not torch.cuda.is_initialized();PUBLIC.mkdir(exist_ok=False,parents=True);write(PUBLIC/'report.json',report)
 with(PUBLIC/'signed_queries.tsv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 write(OUT/'score_receipt.json',dict(status='scored',report_sha256=sha(PUBLIC/'report.json'),new_single_query_scored=1));print(json.dumps(dict(one_case=[dict(query=c['query'],scores={a:dict(t=m['t'],v=m['v'],start_error=m['start_error_seconds'],end_error=m['end_error_seconds'])for a,m in c['metrics'].items()},posterior=c['posterior_diagnostics'])for c in cases],rescued=report['rescued'],harmed=report['harmed'],uncertainty=report['uncertainty_diagnostic']),indent=2))
if __name__=='__main__':evaluate()
