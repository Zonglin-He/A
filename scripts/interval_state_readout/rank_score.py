"""Independent evaluation of sealed structural state-readout outputs."""
import os,sys,json,csv
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.interval_state_readout.rank_run import OUT,PUBLIC,ARMS,NAMES,validate,read,write,sha
from scripts.interval_pool.score import score_boxes,summarize
from scripts.joint_boundary_readout.score import paired,stats
from scripts.joint_boundary_readout.core import iou

def evaluate():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate(p);seal=read(OUT/'sealed.json');assert seal['GT_read']is False and len(seal['files'])==128 and seal['plan_sha256']==sha(OUT/'plan.json')
 for r in seal['files']:assert sha(OUT/r['path'])==r['sha256']
 records={r['id']:r for r in [json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()]};selection=read(ROOT/'outputs/interval_pool/confirmation_selection.json')['development_recheck_selection']
 for value in selection.values():records.update({x['query_id']:x['record']for x in value})
 labels={}
 for cohort,x in p['labels'].items():
  assert sha(ROOT/x['path'])==x['sha256'];labels[cohort]={r['id']:r for r in [json.loads(l)for l in (ROOT/x['path']).read_text().splitlines()if l.strip()]}
 rows=[];changes=[];cases=[];checks=0;error=0.
 for item,se in zip(p['items'],seal['files']):
  gt=labels[item['cohort']][item['id']];record=records[item['id']];boxes={int(k):v for k,v in gt['boxes'].items()};z=np.load(OUT/se['path']);values={};fixed={}
  for arm in NAMES:
   span=z[arm+'_interval'].tolist();values[arm],e=score_boxes(z[arm],record,boxes,span,gt['interval'],z['positions8'].tolist());checks+=1;error=max(error,e);fixed[arm],e=score_boxes(z['FixedSpatial'],record,boxes,span,gt['interval'],z['positions8'].tolist());checks+=1;error=max(error,e)
   assert fixed[arm]['fixed_actual_sIoU']==fixed['Source']['fixed_actual_sIoU']
  quality=iou(z['candidate_intervals'],gt['interval']);assert abs(quality[0]-values['Baseline']['t'])<2e-10;k=int(quality.argmax())
  row=dict(query=item['anonymous'],cohort=item['cohort'],parent=gt['parent'],values=values,fixed_spatial=fixed,changed={a:bool(np.any(z[a+'_interval']!=z['Baseline_interval']))for a in ARMS},candidate_best_tIoU=float(quality[k]));rows.append(row)
  if values['Baseline']['t']<.5<=quality[k]:
   L=z['score_ActionJoint'];U=z['candidate_utilities'];case=dict(query=item['anonymous'],cohort=item['cohort'],baseline_tIoU=values['Baseline']['t'],best_candidate_tIoU=float(quality[k]),correct_event_likelihood_better=bool(L[k]>L[0]),correct_candidate_in_risk_band=bool(U[k]>=U[0]-float(z['posterior_sd'])-1e-12),chosen_tIoU=values['ActionJoint']['t'],crossed_half=values['ActionJoint']['t']>=.5)
   cases.append(case)
  for arm in ARMS:
   a,b=values[arm],values['Baseline'];dv=100*(a['v']-b['v']);dt=100*(a['t']-b['t']);changes.append(dict(cohort=item['cohort'],query=item['anonymous'],method=arm,baseline_wrong_half=b['t']<.5,candidate_can_rescue_half=values['Baseline']['t']<.5<=quality[k],changed=row['changed'][arm],vIoU_delta_pp=dv,tIoU_delta_pp=dt,fixed_space_vIoU_delta_pp=100*(fixed[arm]['v']-fixed['Baseline']['v']),boundary_MAE_delta_seconds=a['boundary_MAE_seconds']-b['boundary_MAE_seconds'],start_error_delta_seconds=a['start_error_seconds']-b['start_error_seconds'],end_error_delta_seconds=a['end_error_seconds']-b['end_error_seconds'],recovered_half=b['t']<.5<=a['t'],lost_half=a['t']<.5<=b['t'],v_declined_over5pp=dv < -5))
 cohorts={}
 for cohort in ['HC64','HC32','Vid32']:
  rr=[r for r in rows if r['cohort']==cohort];assert len(rr)==len({r['parent']for r in rr});subsets={'all':rr,'baseline_wrong_half':[r for r in rr if r['values']['Baseline']['t']<.5],'baseline_correct_half':[r for r in rr if r['values']['Baseline']['t']>=.5],'baseline_wrong_and_rescuable':[r for r in rr if r['values']['Baseline']['t']<.5<=r['candidate_best_tIoU']]}
  cohorts[cohort]=dict(queries=len(rr),means=summarize(rr,NAMES,'values'),fixed_spatial_means=summarize(rr,NAMES,'fixed_spatial'),subsets={k:dict(queries=len(v),paired={a:paired(v,a)for a in ARMS})for k,v in subsets.items()},controls={a:paired(rr,'ActionJoint',a)for a in ARMS if a!='ActionJoint'},changed={a:sum(r['changed'][a]for r in rr)for a in ARMS},Source_relative_severe_v={a:sum(r['values'][a]['v']-r['values']['Source']['v'] < -.05 for r in rr)for a in ['Baseline',*ARMS]})
 dev=cohorts['HC64'];d=dev['subsets']['all']['paired']['ActionJoint'];gate=dict(t_positive=d['t']['mean']>0,full_v_positive=d['v']['mean']>0,fixed_v_positive=d['fixed_space_v']['mean']>0,better_t_than_hidden=dev['controls']['ActionOnly']['t']['mean']>0,better_t_than_permuted=dev['controls']['ActionPermuted']['t']['mean']>0,severe_source_v_increase_le2=dev['Source_relative_severe_v']['ActionJoint']<=dev['Source_relative_severe_v']['Baseline']+2);passing=all(gate.values());rechecks={};pooled={};adopt=False
 if passing:
  for c in ['HC32','Vid32']:
   g=cohorts[c];dd=g['subsets']['all']['paired']['ActionJoint'];rechecks[c]=dict(t_positive=dd['t']['mean']>0,full_v_nonnegative=dd['v']['mean']>=0,severe_source_v_increase_le2=g['Source_relative_severe_v']['ActionJoint']<=g['Source_relative_severe_v']['Baseline']+2)
  rng=np.random.default_rng(20261022)
  for field in ['t','v']:
   arrays=[np.asarray([r['values']['ActionJoint'][field]-r['values']['Baseline'][field]for r in rows if r['cohort']==c])for c in ['HC32','Vid32']];draw=.5*sum(a[rng.integers(0,32,(10000,32))].mean(1)for a in arrays);pooled[field]=dict(mean_pp=100*float(np.concatenate(arrays).mean()),CI95_pp=(100*np.quantile(draw,[.025,.975])).tolist())
  adopt=all(all(g.values())for g in rechecks.values())and pooled['t']['CI95_pp'][0]>0 and pooled['v']['CI95_pp'][0]>-.5
 decision=dict(development_gate=gate,development_passed=passing,recheck_gates=rechecks,pooled_recheck=pooled,validated_candidate=adopt,production_changed=False,reason='passes fixed recheck'if adopt else'no adoption: structural candidate failed fixed development/recheck gate')
 temporal_gate={};temporal_recheck={};temporal_validated=False
 td=dev['subsets']['all']['paired']['ActionOnly'];temporal_gate=dict(t_positive=td['t']['mean']>0,full_v_positive=td['v']['mean']>0,fixed_v_positive=td['fixed_space_v']['mean']>0,severe_source_v_increase_le2=dev['Source_relative_severe_v']['ActionOnly']<=dev['Source_relative_severe_v']['Baseline']+2)
 if all(temporal_gate.values()):
  for c in ['HC32','Vid32']:
   g=cohorts[c];d=g['subsets']['all']['paired']['ActionOnly'];temporal_recheck[c]=dict(t_positive=d['t']['mean']>0,full_v_nonnegative=d['v']['mean']>=0,severe_source_v_increase_le2=g['Source_relative_severe_v']['ActionOnly']<=g['Source_relative_severe_v']['Baseline']+2)
  temporal_validated=all(all(g.values())for g in temporal_recheck.values())
 decision.update(temporal_only_development_gate=temporal_gate,temporal_only_recheck=temporal_recheck,temporal_only_positive_directional_signal=temporal_validated,temporal_only_not_joint_contribution=True)
 public=dict(status='completed_calibration_free_rank_interval_readout',cohorts=cohorts,decision=decision,rescuable_cases=cases,development_prior_case_audit=read(ROOT/'outputs/interval_state_readout_v1/development_case_audit.json'),queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],proof=dict(plan_sha256=sha(OUT/'plan.json'),prediction_seal_sha256=sha(OUT/'sealed.json'),all128_new_predictions_before_new_GT_scores=True,HC64_prior_cases_used_for_design=True,recheck_cases_not_newly_inspected_before_lock=False,all_cohorts_reused_exposed_development=True,official_dense_checks=checks,official_dense_max_error=error,CPU_existing_action_head_calls=seal['CPU_frozen_head_calls'],heavy_network_calls=0,GPU_calls=0,parameter_updates=0,manual_float64_head_maxerror=seal['manual_float64_head_maxerror']),costs={c:{k:float(np.mean([r[k]for r in seal['files']if r['cohort']==c]))for k in ['input_IO_seconds','head_seconds','score_seconds']}for c in cohorts},limitations=['All data historically exposed; HC32/Vid32 are frozen development rechecks, not pristine confirmation.','GT candidate rescue cases are diagnostic only. No sample whitelist or GT interval enters inference.','One fixed structural revision; no weight/threshold sweep.','Motion feature state separation is not event probability, identity correctness or causal proof.'])
 validate(p);assert not torch.cuda.is_initialized();PUBLIC.mkdir(parents=True,exist_ok=True);assert not(PUBLIC/'report.json').exists();write(PUBLIC/'report.json',public);write(PUBLIC/'decision.json',decision)
 with open(PUBLIC/'signed_queries.tsv','w')as f:
  w=csv.DictWriter(f,fieldnames=list(changes[0]),delimiter='\t');w.writeheader();w.writerows(changes)
 write(OUT/'score_receipt.json',dict(report_sha256=sha(PUBLIC/'report.json'),signed_sha256=sha(PUBLIC/'signed_queries.tsv')));print(json.dumps(decision,indent=2))
if __name__=='__main__':evaluate()
