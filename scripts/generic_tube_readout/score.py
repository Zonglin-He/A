"""Independent evaluation of sealed generic expected tube-overlap outputs."""
import os,sys,json,csv
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.generic_tube_readout.run import OUT,PUBLIC,ARMS,NAMES,validate,read,write,sha
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
   chosen=se['selected']['ExpectedTube'];case=dict(query=item['anonymous'],cohort=item['cohort'],baseline_tIoU=values['Baseline']['t'],best_candidate_tIoU=float(quality[k]),chosen_tIoU=values['ExpectedTube']['t'],correct_score=float(z['score_ExpectedTube'][k]),baseline_score=float(z['score_ExpectedTube'][0]),chosen_score=float(z['score_ExpectedTube'][chosen]),correct_temporal_utility=float(z['candidate_utilities'][k]),baseline_temporal_utility=float(z['candidate_utilities'][0]),crossed_half=values['ExpectedTube']['t']>=.5)
   cases.append(case)
  for arm in ARMS:
   a,b=values[arm],values['Baseline'];dv=100*(a['v']-b['v']);dt=100*(a['t']-b['t']);changes.append(dict(cohort=item['cohort'],query=item['anonymous'],method=arm,baseline_wrong_half=b['t']<.5,candidate_can_rescue_half=values['Baseline']['t']<.5<=quality[k],changed=row['changed'][arm],vIoU_delta_pp=dv,tIoU_delta_pp=dt,fixed_space_vIoU_delta_pp=100*(fixed[arm]['v']-fixed['Baseline']['v']),boundary_MAE_delta_seconds=a['boundary_MAE_seconds']-b['boundary_MAE_seconds'],start_error_delta_seconds=a['start_error_seconds']-b['start_error_seconds'],end_error_delta_seconds=a['end_error_seconds']-b['end_error_seconds'],recovered_half=b['t']<.5<=a['t'],lost_half=a['t']<.5<=b['t'],v_declined_over5pp=dv < -5))
 cohorts={}
 for cohort in ['HC64','HC32','Vid32']:
  rr=[r for r in rows if r['cohort']==cohort];assert len(rr)==len({r['parent']for r in rr});subsets={'all':rr,'baseline_wrong_half':[r for r in rr if r['values']['Baseline']['t']<.5],'baseline_correct_half':[r for r in rr if r['values']['Baseline']['t']>=.5],'baseline_wrong_and_rescuable':[r for r in rr if r['values']['Baseline']['t']<.5<=r['candidate_best_tIoU']]}
  cohorts[cohort]=dict(queries=len(rr),means=summarize(rr,NAMES,'values'),fixed_spatial_means=summarize(rr,NAMES,'fixed_spatial'),subsets={k:dict(queries=len(v),paired={a:paired(v,a)for a in ARMS})for k,v in subsets.items()},controls={a:paired(rr,'ExpectedTube',a)for a in ARMS if a!='ExpectedTube'},changed={a:sum(r['changed'][a]for r in rr)for a in ARMS},Source_relative_severe_v={a:sum(r['values'][a]['v']-r['values']['Source']['v'] < -.05 for r in rr)for a in ['Baseline',*ARMS]})
 dev=cohorts['HC64'];d=dev['subsets']['all']['paired']['ExpectedTube'];gate=dict(t_nonnegative=d['t']['mean']>=0,full_v_positive=d['v']['mean']>0,fixed_v_positive=d['fixed_space_v']['mean']>0,better_t_than_shifted=dev['controls']['ShiftedQuality']['t']['mean']>0,severe_source_v_increase_le2=dev['Source_relative_severe_v']['ExpectedTube']<=dev['Source_relative_severe_v']['Baseline']+2);passing=all(gate.values());rechecks={};pooled={}
 for c in ['HC32','Vid32']:
  dd=cohorts[c]['subsets']['all']['paired']['ExpectedTube'];rechecks[c]=dict(t_nonnegative=dd['t']['mean']>=0,full_v_positive=dd['v']['mean']>0)
 rng=np.random.default_rng(20261022)
 for field in ['t','v']:
  arrays=[np.asarray([r['values']['ExpectedTube'][field]-r['values']['Baseline'][field]for r in rows if r['cohort']==c])for c in ['HC32','Vid32']];draw=.5*sum(a[rng.integers(0,32,(10000,32))].mean(1)for a in arrays);pooled[field]=dict(mean_pp=100*float(np.concatenate(arrays).mean()),CI95_pp=(100*np.quantile(draw,[.025,.975])).tolist())
 adopt=passing and all(all(g.values())for g in rechecks.values())and pooled['v']['CI95_pp'][0]>0
 decision=dict(development_gate=gate,development_passed=passing,recheck_gates=rechecks,pooled_recheck=pooled,validated_candidate=adopt,production_changed=False,reason='passes fixed development checks'if adopt else'no adoption: generic candidate failed fixed development/recheck gate')
 public=dict(status='completed_generic_box_posterior_readout',cohorts=cohorts,decision=decision,rescuable_cases=cases,queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],proof=dict(plan_sha256=sha(OUT/'plan.json'),prediction_seal_sha256=sha(OUT/'sealed.json'),all128_new_predictions_before_new_GT_scores=True,HC64_prior_cases_used_for_design=True,all_cohorts_previously_exposed=True,official_dense_checks=checks,official_dense_max_error=error,network_calls=0,GPU_calls=0,parameter_updates=0,constant_quality_temporal_utility_max_error=seal['constant_quality_temporal_utility_max_error']),costs={c:{k:float(np.mean([r[k]for r in seal['files']if r['cohort']==c]))for k in ['input_IO_seconds','candidate_spatial_DP_seconds','expected_overlap_seconds']}for c in cohorts},limitations=['All data historically exposed; HC32/Vid32 are frozen development rechecks, not pristine confirmation.','GT candidate rescue cases are diagnostic only. No sample whitelist or GT interval enters inference.','One fixed generic structural trial; no weight/threshold sweep or actionness/hidden/attention input.','Spatial hypothesis agreement is not calibrated true IoU, identity correctness or event probability.'])
 validate(p);assert not torch.cuda.is_initialized();PUBLIC.mkdir(parents=True,exist_ok=True);assert not(PUBLIC/'report.json').exists();write(PUBLIC/'report.json',public);write(PUBLIC/'decision.json',decision)
 with open(PUBLIC/'signed_queries.tsv','w')as f:
  w=csv.DictWriter(f,fieldnames=list(changes[0]),delimiter='\t');w.writeheader();w.writerows(changes)
 write(OUT/'score_receipt.json',dict(report_sha256=sha(PUBLIC/'report.json'),signed_sha256=sha(PUBLIC/'signed_queries.tsv')));print(json.dumps(decision,indent=2))
if __name__=='__main__':evaluate()
