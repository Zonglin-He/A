"""Independent CPU evaluation after the whole candidate-readout panel is sealed.

No label-selected sample or interval enters inference. Wrong/correct subsets and
candidate oracles are evaluation-only, never method inputs or acceptance rules.
"""
import os,sys,json,csv,itertools
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.joint_boundary_readout.v2_run import OUT,PUBLIC,NAMES,ARMS,read,write,sha,validate_plan
from scripts.joint_boundary_readout.core import iou
from scripts.interval_pool.score import score_boxes,summarize

def stats(x):
 x=np.asarray(x,float);n=len(x)
 if not n:return dict(queries=0,mean=None,CI95=None)
 ix=np.random.default_rng(20261022).integers(0,n,(10000,n));return dict(queries=n,mean=float(x.mean()),CI95=np.quantile(x[ix].mean(1),[.025,.975]).tolist(),increased=int((x>1e-10).sum()),decreased=int((x < -1e-10).sum()),unchanged=int((abs(x)<=1e-10).sum()))
def paired(rows,a,b='Baseline'):
 out={}
 for f,scale in [('v',100),('t',100),('boundary_MAE_seconds',1),('start_error_seconds',1),('end_error_seconds',1),('unobserved_actual_sIoU',100)]:
  d=[scale*(r['values'][a][f]-r['values'][b][f])for r in rows if r['values'][a][f]is not None and r['values'][b][f]is not None];out[f]=stats(d)
 out['fixed_space_v']=stats([100*(r['fixed_spatial'][a]['v']-r['fixed_spatial'][b]['v'])for r in rows])
 out['recovered_tIoU_half']=sum(r['values'][b]['t']<.5<=r['values'][a]['t']for r in rows);out['lost_tIoU_half']=sum(r['values'][a]['t']<.5<=r['values'][b]['t']for r in rows)
 out['t_declined_over5pp']=sum(r['values'][a]['t']-r['values'][b]['t'] < -.05 for r in rows);out['v_declined_over5pp']=sum(r['values'][a]['v']-r['values'][b]['v'] < -.05 for r in rows)
 out['start_rescued_within1s']=sum(r['values'][b]['start_error_seconds']>1>=r['values'][a]['start_error_seconds']for r in rows);out['end_rescued_within1s']=sum(r['values'][b]['end_error_seconds']>1>=r['values'][a]['end_error_seconds']for r in rows)
 out['start_harmed_beyond1s']=sum(r['values'][a]['start_error_seconds']>1>=r['values'][b]['start_error_seconds']for r in rows);out['end_harmed_beyond1s']=sum(r['values'][a]['end_error_seconds']>1>=r['values'][b]['end_error_seconds']for r in rows)
 return out

def evaluate():
 torch.set_num_threads(1);os.nice(15);p=read(OUT/'plan.json');validate_plan(p);seal=read(OUT/'sealed.json');assert seal['GT_read']is False and seal['plan_sha256']==sha(OUT/'plan.json');assert len(seal['files'])==128
 for r in seal['files']:assert sha(OUT/r['path'])==r['sha256']
 records=[json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()];records={r['id']:r for r in records};selection=read(ROOT/'outputs/interval_pool/confirmation_selection.json')['development_recheck_selection']
 for values in selection.values():records.update({v['query_id']:v['record']for v in values})
 labels={}
 for cohort,entry in p['labels'].items():
  assert sha(ROOT/entry['path'])==entry['sha256'];labels[cohort]={g['id']:g for g in [json.loads(l)for l in (ROOT/entry['path']).read_text().splitlines()if l.strip()]}
 rows=[];checks=0;maxerror=0.;signed=[]
 for item,rec in zip(p['items'],seal['files']):
  key=item['anonymous'];label=labels[item['cohort']][item['id']];record=records[item['id']];gt={int(k):v for k,v in label['boxes'].items()};assert item['fps']==record['metadata']['fps']
  z=np.load(OUT/rec['path'],allow_pickle=False);values={};fixed={}
  for name in NAMES:
   interval=z[name+'_interval'].tolist();values[name],err=score_boxes(z[name],record,gt,interval,label['interval'],z['positions8'].tolist());maxerror=max(maxerror,err);checks+=1
   fixed[name],err=score_boxes(z['FixedSpatial'],record,gt,interval,label['interval'],z['positions8'].tolist());maxerror=max(maxerror,err);checks+=1
   assert fixed[name]['fixed_actual_sIoU']==fixed['Source']['fixed_actual_sIoU']
  cand_t=iou(z['candidate_intervals'],label['interval']);baseline_t=values['Baseline']['t'];assert abs(cand_t[0]-baseline_t)<2e-10
  best=float(cand_t.max());r=dict(query=key,cohort=item['cohort'],parent=label['parent'],ambiguous=rec['ambiguous'],values=values,fixed_spatial=fixed,changed={a:bool(np.any(z[a+'_interval']!=z['Baseline_interval']))for a in ARMS},candidate_best_tIoU=best,candidate_oracle_tIoU_delta=best-baseline_t,candidates=rec['candidates'],base_expected_iou=rec['base_expected_iou'],view_agreement=rec['view_agreement']);rows.append(r)
  for a in ARMS:
   va,vb=values[a],values['Baseline'];dv=100*(va['v']-vb['v']);dt=100*(va['t']-vb['t'])
   signed.append(dict(cohort=item['cohort'],query=key,method=a,ambiguous=r['ambiguous'],baseline_wrong_tIoU_half=vb['t']<.5,Source_wrong_tIoU_half=values['Source']['t']<.5,interval_changed=r['changed'][a],baseline_tIoU=vb['t'],method_tIoU=va['t'],tIoU_delta_pp=dt,vIoU_delta_pp=dv,fixed_space_vIoU_delta_pp=100*(fixed[a]['v']-fixed['Baseline']['v']),boundary_MAE_delta_seconds=va['boundary_MAE_seconds']-vb['boundary_MAE_seconds'],start_error_delta_seconds=va['start_error_seconds']-vb['start_error_seconds'],end_error_delta_seconds=va['end_error_seconds']-vb['end_error_seconds'],recovered_tIoU_half=vb['t']<.5<=va['t'],lost_tIoU_half=va['t']<.5<=vb['t'],v_declined_over5pp=dv < -5,candidate_oracle_tIoU_delta_pp=100*(best-baseline_t)))
 cohorts={}
 for cohort in ['HC64','HC32','Vid32']:
  rr=[r for r in rows if r['cohort']==cohort];assert len(rr)==len({r['parent']for r in rr})
  subsets={'all':rr,'ambiguous':[r for r in rr if r['ambiguous']],'nonambiguous':[r for r in rr if not r['ambiguous']],'baseline_wrong_half':[r for r in rr if r['values']['Baseline']['t']<.5],'wrong_and_ambiguous':[r for r in rr if r['ambiguous']and r['values']['Baseline']['t']<.5],'baseline_correct_half':[r for r in rr if r['values']['Baseline']['t']>=.5],'source_wrong_half':[r for r in rr if r['values']['Source']['t']<.5]}
  subset_stats={k:dict(queries=len(v),paired={a:paired(v,a)for a in ARMS})for k,v in subsets.items()}
  controls={}
  for w in ['10','25']:
   for b in ['Hidden'+w,'Permuted'+w]:controls['Joint'+w+'-'+b]=paired(rr,'Joint'+w,b)
  cohorts[cohort]=dict(queries=len(rr),means=summarize(rr,NAMES,'values'),fixed_spatial_means=summarize(rr,NAMES,'fixed_spatial'),subsets=subset_stats,controls=controls,
   changed={a:sum(r['changed'][a]for r in rr)for a in ARMS},Source_relative_severe_v={a:sum(r['values'][a]['v']-r['values']['Source']['v'] < -.05 for r in rr)for a in ['Baseline',*ARMS]},
   candidate_oracle_delta_tIoU_pp=stats([100*r['candidate_oracle_tIoU_delta']for r in rr]),wrong_candidate_can_rescue_half=sum(r['values']['Baseline']['t']<.5<=r['candidate_best_tIoU']for r in rr),
   ambiguous_candidate_can_rescue_half=sum(r['ambiguous']and r['values']['Baseline']['t']<.5<=r['candidate_best_tIoU']for r in rr))
 dev=cohorts['HC64'];gates={}
 for suffix in ['10','25']:
  arm='Joint'+suffix;x=dev['subsets']['all']['paired'][arm];gates[arm]=dict(delta_t_ge_025pp=x['t']['mean']>=.25,delta_full_v_positive=x['v']['mean']>0,delta_fixed_v_positive=x['fixed_space_v']['mean']>0,better_t_than_hidden=dev['controls'][arm+'-Hidden'+suffix]['t']['mean']>0,better_t_than_permuted=dev['controls'][arm+'-Permuted'+suffix]['t']['mean']>0,source_severe_increase_le2=dev['Source_relative_severe_v'][arm]<=dev['Source_relative_severe_v']['Baseline']+2)
 passing=[a for a,g in gates.items()if all(g.values())];selected=max(passing,key=lambda a:dev['subsets']['all']['paired'][a]['t']['mean'])if passing else None;rechecks={};adopt=False;pooled={}
 if selected:
  for c in ['HC32','Vid32']:
   g=cohorts[c];d=g['subsets']['all']['paired'][selected];rechecks[c]=dict(delta_t_positive=d['t']['mean']>0,delta_v_nonnegative=d['v']['mean']>=0,source_severe_increase_le2=g['Source_relative_severe_v'][selected]<=g['Source_relative_severe_v']['Baseline']+2)
  rr=[r for r in rows if r['cohort']in ['HC32','Vid32']];rng=np.random.default_rng(20261022)
  for field in ['v','t']:
   arrays=[np.asarray([r['values'][selected][field]-r['values']['Baseline'][field]for r in rows if r['cohort']==c])for c in ['HC32','Vid32']];draw=.5*sum(a[rng.integers(0,32,(10000,32))].mean(1)for a in arrays);pooled[field]=dict(mean_pp=100*float(np.mean(np.concatenate(arrays))),CI95_pp=(100*np.quantile(draw,[.025,.975])).tolist())
  adopt=all(all(g.values())for g in rechecks.values())and pooled['t']['CI95_pp'][0]>0 and pooled['v']['CI95_pp'][0]>-.5
 decision=dict(development_gates=gates,selected_from_HC64=selected,recheck_gates=rechecks,pooled_recheck=pooled,validated_lightweight_candidate=adopt,production_changed=False,reason='passed fixed two-domain recheck'if adopt else'no development candidate passed; stop this formulation'if not selected else'frozen two-domain recheck failed; retain original readout')
 costs={}
 for cohort in ['HC64','HC32','Vid32']:
  rr=[r for r in seal['files']if r['cohort']==cohort];costs[cohort]={k:dict(mean=float(np.mean([r[k]for r in rr])),median=float(np.median([r[k]for r in rr])))for k in ['input_IO_seconds','pool_seconds','feature_seconds','seconds']};costs[cohort]['arms']={a:{k:float(np.mean([r['costs'][a][k]for r in rr]))for k in ['rerank_seconds','actual_DP_seconds']}for a in ARMS}
 validate_plan(p);assert not torch.cuda.is_initialized()
 result=dict(status='completed_sealed_actual_boundary_readout_trial',cohorts=cohorts,decision=decision,costs=costs,queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],
 proof=dict(all128_predictions_before_GT=True,plan_sha256=sha(OUT/'plan.json'),seal_sha256=sha(OUT/'sealed.json'),official_dense_checks=checks,official_dense_max_error=maxerror,network_calls=0,GPU_calls=0,parameter_updates=0,formal_sources_unchanged=434),
 limitations=['Historically exposed development; this is not pristine independent confirmation.','GT wrong/correct subsets and candidate oracle are offline only; method does not know correctness.','Two fixed strengths selected only by HC64 gates; rechecks never choose a strength.','Added support uses native frozen attention on full-timeline spatial trajectory; no new semantic teacher.','Reported costs are incremental CPU cached readout and include candidate construction; no end-to-end speedup claim.'])
 PUBLIC.mkdir(parents=True,exist_ok=True);assert not(PUBLIC/'report.json').exists();write(PUBLIC/'report.json',result);write(PUBLIC/'decision.json',decision)
 with open(PUBLIC/'signed_queries.tsv','w')as f:
  w=csv.DictWriter(f,fieldnames=list(signed[0]),delimiter='\t');w.writeheader();w.writerows(signed)
 write(OUT/'score_receipt.json',dict(status='scored',report_sha256=sha(PUBLIC/'report.json'),signed_sha256=sha(PUBLIC/'signed_queries.tsv'),decision=decision));print(json.dumps(decision,indent=2))
if __name__=='__main__':evaluate()
