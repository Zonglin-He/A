"""Same frozen expert interpolation, independent native adapted temporal outputs."""
from common import *
import argparse,gc
from functools import lru_cache
fixed=lru_cache(None)(read)

def prepare():
 assert not(D/'plan.json').exists();rc.validate();inputs=[rc.D/'matched_cpu_plan.json',rc.selected.D/'plan.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',T/'plan.json']
 for ds,n in COUNTS.items():
  a=read(T/f'{ds}_reporting_receipt.json');assert a['status']=='verified_full_domain_Tent_SAR'and a['queries']==n and a['all_episodic_resets_exact']and a['all_native_joint_readouts_replayed_exact']
  assert a['score_sha256']==sha(T/f'{ds}_score_receipt.json')
  for f in [T/f'{ds}_sealed.json',T/f'{ds}_score_receipt.json',T/f'{ds}_reporting_receipt.json',rc.RUN/ds/'sealed.json',rc.selected.D/f'components_{ds}_sealed.json',rc.selected.D/f'components_{ds}_score_receipt.json',rc.selected.D/f'components_{ds}_reporting_receipt.json',rc.D/'matched_cpu'/ds/'sealed.json',rc.D/'matched_cpu'/ds/'reporting_receipt.json']:inputs.append(f)
 p=dict(schema='expert_matched_native_Tent_SAR_full_tubes_v1',question='Does structured inference retain an advantage when the frozen Tent/SAR STVG adapters receive the identical eight expert observations through a fully specified interpolation fusion?',cohort=COUNTS,arms=ARMS,pairs=PAIRS,primary='Ours versus TentExpertInterpolation and SARExpertInterpolation; independent native adapted MAP/hull intervals, never Ours interval',secondary='No-update native Source temporal readout with identical interpolation isolates adaptation from expert fusion; raw native Source retained for context.',expert='One original Source-selected positions8, same existing ReferDINO boxes/native mask payload SHA and original caption/RGB context; no new expert call. Same interpolation tensor for all three interpolation rows.',interpolation='Original stvg.paper_panel.interpolation center/logsize interpolation of same8 native expert boxes with original E outside anchor coverage; coordinates and outside rule byte-exact to the previously audited original interpolation. No competitor DP or Ours posterior.',temporal='Pinned official Tent/SAR lr1e-4/five steps/SARmargin.8/temporal-decoder LayerNorm-affine/episodic reset, exactly saved after-update logits/native per-offset legal MAP physical hull. No additional optimizer update or expert-supervised loss. Do not use Ours time posterior/MAP interval. Ours has independent own current selected event.',updates='Before each prediction commit verify SHA-bound trace five real steps, parameter delta finite and original trials/reset/official provenance. First actual two cases per domain additionally reload saved adapted logits and independently replay native readout exact. Whole-domain reset/readout/parameter-change diagnostics inherited and counted again from every receipt; SAR reliable-filter empty updates retained.',evaluation='Each whole domain seal before independent official CPU scoring of the three new combinations; raw Source/Ours reuse exact independently scored/audited metrics after array identity proofs. Independent cold all-array/SHA/means audit. t/v/s/R.3/R.5, all signed severe t/v union once/rescues/shortlong/parent10000bootstrap95CI. Fullsets previously exposed, no tuning or unseen confirmation.',cost='New cached CPU composition and scoring wall are not deployment. Existing Source/expert/optimizer costs remain nonzero and inherited once. Exact standalone end-to-end new combination timing requires a separate matched actual GPU plan after existing GPU queue; missing end-to-end combination latency must remain not measured, never sum nested costs or transplant Ours latency.',limits=dict(CPU_jobs=2,owned_compute_jobs=1,affinity=[14,15],nice=19,io='idle3',threads=1,CUDAhidden=True,min_RAM_bytes=8*2**30,min_disk_bytes=80*2**30,max_private_bytes=160*2**30,max_domain_seconds=259200,new_Source=0,new_expert=0,new_optimizer=0),failure='Hard fail and preserve original plan/source/log/all outputs; no skip/fallback/hash waiver or repeat acquisition.',code_sha256={str(f.relative_to(R)):sha(f)for f in S.glob('*.py')},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'])

def derive(ds,row):
 import numpy as np
 from stvg.paper_panel import interpolation
 pe,ce=rc.paths(ds,row['id']);te=fixed(T/f'{ds}_sealed.json')['files'][row['id']]
 assert not te['GT_read']and te['all_readouts_cold_exact'];assert sha(T/te['path'])==te['sha256']and sha(T/te['receipt_path'])==te['receipt_sha256']
 for m in ['Tent','SAR']:
  t=te['traces'][m];trial=t['trial'];assert t['reset_exact']and t['official_source_unmodified']and len(t['steps'])==5 and trial['method']==m and trial['lr']==.0001 and trial['steps']==5
  assert np.isfinite(t['parameter_delta'])and t['parameter_delta']>=0
  if m=='SAR':assert trial['margin_fraction']==.8
 with np.load(rc.RUN/ds/pe['path'])as z,np.load(rc.selected.D/ce['path'])as c,np.load(T/te['path'])as t:
  assert np.array_equal(z['Source'],c['Source'])and np.array_equal(z['Source'],t['Source'])and np.array_equal(z['frame_ids'],t['frame_ids'])and np.array_equal(z['positions8'],t['positions8'])
  inter=interpolation(z['SharedInit'],z['frame_ids'],z['positions8'],z['native_boxes']);assert np.array_equal(inter,z['Interpolation'])
  a=dict(Source=z['Source'].copy(),Source_interval=t['Source_interval'].copy(),Ours=c['FullMAP'].copy(),Ours_interval=c['FullMAP_interval'].copy(),frame_ids=z['frame_ids'].copy(),positions8=z['positions8'].copy())
  for arm,key in [('SourceExpertInterpolation','Source'),('TentExpertInterpolation','TentNative'),('SARExpertInterpolation','SARNative')]:a[arm]=inter.copy();a[arm+'_interval']=t[key+'_interval'].copy()
  assert all(np.isfinite(v).all()for v in a.values());assert all(np.array_equal(a[x],a['SourceExpertInterpolation'])for x in ['TentExpertInterpolation','SARExpertInterpolation'])
 proof=dict(parent_prediction_sha256=pe['sha256'],parent_receipt_sha256=pe['receipt_sha256'],current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],TTA_prediction_sha256=te['sha256'],TTA_receipt_sha256=te['receipt_sha256'],TTA_logits_sha256=te['logits_sha256'],same_expert_positions_and_Source=True,original_interpolation_exact=True,independent_native_adapted_intervals=True,competitor_DP=False,Ours_temporal_posterior_for_competitors=False,GT_read=False)
 return a,proof

def preflight():
 import numpy as np,torch
 from stvg.readout import prediction
 torch.set_num_threads(1);validate();rc.gt_guard();proof=[]
 for ds in COUNTS:
  for row in records(ds)[:2]:
   a,x=derive(ds,row);b,y=derive(ds,row);assert x==y and all(np.array_equal(v,b[k])for k,v in a.items())
   e=fixed(T/f'{ds}_sealed.json')['files'][row['id']];assert sha(T/e['logits_path'])==e['logits_sha256'];log=torch.load(T/e['logits_path'],map_location='cpu',weights_only=False,mmap=True)
   for m in ['Tent','SAR']:
    iv=prediction(log['adapted_logits'][m],torch.as_tensor(log['Source']),log['records'],log['frame_ids'])['physical_interval'];assert np.array_equal(a[m+'ExpertInterpolation_interval'],iv)
   proof.append(dict(dataset=ds,repeat_all_arrays_exact=True,saved_after_update_logits_native_readout_exact=True,traces_actual_updates_and_resets_checked=True,no_Ours_interval_for_competitors=True))
 assert not torch.cuda.is_initialized();rec=dict(status='passed',actual_cached_cases=4,cases=proof,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0);atomic(D/'preflight_receipt.json',rec);atomic(P/'preflight_receipt.json',rec);print('4 actual cases repeat exact/native readout verified')

def predict(ds):
 import numpy as np,torch
 torch.set_num_threads(1);validate();rc.gt_guard();assert read(D/'preflight_receipt.json')['status']=='passed';dest=D/ds;assert not(dest/'worker.json').exists();who=identity();atomic(dest/'worker.json',who);start=time.perf_counter();files={};stats={m:dict(queries=0,nonzero_parameter_delta=0,logits_changed=0,iterations=0)for m in ['Tent','SAR']}
 for i,row in enumerate(records(ds),1):
  assert resources()and time.perf_counter()-start<259200;a,proof=derive(ds,row);out=dest/'predictions'/(row['id']+'.npz');rp=dest/'receipts'/(row['id']+'.json');assert not out.exists()and not rp.exists();out.parent.mkdir(parents=True,exist_ok=True)
  with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(out.with_suffix('.tmp'),out)
  with np.load(out)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(out.relative_to(D)),sha256=sha(out),provenance=proof,worker=who,fingerprint=read(D/'plan.json')['fingerprint'],GT_read=False);atomic(rp,rec);files[row['id']]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
  te=fixed(T/f'{ds}_sealed.json')['files'][row['id']]
  for m,t in te['traces'].items():stats[m]['queries']+=1;stats[m]['nonzero_parameter_delta']+=int(t['parameter_delta']>0);stats[m]['logits_changed']+=int(t['logits_changed']);stats[m]['iterations']+=len(t['steps'])
  if i%25==0:atomic(dest/'progress.json',dict(completed=i,total=COUNTS[ds],elapsed_wall_seconds=time.perf_counter()-start,GT_read=False))
 for m in stats:assert stats[m]['queries']==COUNTS[ds]and stats[m]['iterations']==5*COUNTS[ds]
 atomic(dest/'sealed.json',dict(queries=len(files),files=files,worker=who,seconds=time.perf_counter()-start,update_diagnostics=stats,GT_read=False,fingerprint=read(D/'plan.json')['fingerprint']))

def score(ds):
 import numpy as np,torch
 from full_score import metric
 torch.set_num_threads(1);validate();dest=D/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=rc.selected.labels(ds);old={r['id']:r for r in read(rc.selected.D/'components'/ds/'metrics.json')['private_rows']};rows=[];checks=0;maxerr=0.;start=time.perf_counter()
 for row in records(ds):
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={'Source':old[row['id']]['values']['Source'],'Ours':old[row['id']]['values']['FullMAP']}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ['SourceExpertInterpolation','TentExpertInterpolation','SARExpertInterpolation']:
    vals[arm],err=metric(z[arm],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err);assert err<2e-10
  rows.append(dict(id=row['id'],parent=str(g['parent']),duration_group=old[row['id']]['duration_group'],values=vals))
 res=rc.selected.suite().parent_report(rows,ARMS,PAIRS,P/ds);rc.selected.suite().signed_tails(rows,PAIRS,P/ds);atomic(dest/'metrics.json',dict(private_rows=rows,report=res));rec=dict(status='scored',queries=len(rows),checks=checks,max_error=maxerr,whole_domain_sealed_before_GT=True,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),inherited_official_scored_arms=['Source','Ours']);atomic(dest/'score_receipt.json',rec);atomic(P/ds/'score_receipt.json',rec)

def audit(ds):
 import numpy as np,torch
 torch.set_num_threads(1);validate();rc.gt_guard();dest=D/ds;seal=read(dest/'sealed.json');score=read(dest/'score_receipt.json');assert score['seal_sha256']==sha(dest/'sealed.json')and score['metrics_sha256']==sha(dest/'metrics.json');start=time.perf_counter();checks=0
 for row in records(ds):
  a,proof=derive(ds,row);e=seal['files'][row['id']];assert e['provenance']==proof and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
  with np.load(D/e['path'])as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
  checks+=2
 rows=read(dest/'metrics.json')['private_rows'];report=read(P/ds/'report.json');assert len(rows)==COUNTS[ds]
 for arm,mm in report['means'].items():
  for k,v in mm.items():
   x=[r['values'][arm][k]for r in rows if r['values'][arm][k]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 rec=dict(status='verified_whole_domain',queries=len(rows),prediction_receipt_SHA_checks=checks,all_arrays_cold_rederived_exact=True,all_means_recomputed=True,competitor_intervals_from_own_native_adapted_logits=True,expert_interpolation_same_tensor_all_three_rows=True,whole_domain_official_scoring=True,score_sha256=sha(dest/'score_receipt.json'),seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),update_diagnostics=seal['update_diagnostics'],worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_optimizer=0,root_analysis_publication_pending=True);atomic(dest/'reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec);print(ds,'whole prediction score audit passed')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','predict','score','audit']);a.add_argument('--dataset',choices=COUNTS);v=a.parse_args();globals()[v.command](*([]if v.command in ['prepare','preflight']else[v.dataset]))
