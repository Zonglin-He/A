from common import *
import argparse,csv,subprocess,traceback

def prepare():
 selected.validate();assert not(D/'plan.json').exists();inputs=[R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',selected.D/'plan.json',STRUCT/'plan.json']
 for ds,n in COUNTS.items():
  for p in [selected.D/f'components_{ds}_reporting_receipt.json',STRUCT/f'spatial_{ds}_reporting_receipt.json',MATCH/ds/'reporting_receipt.json']:
   a=read(p);assert a['status']=='verified_whole_domain'and a['queries']==n
  inputs += [selected.RUN/ds/'sealed.json',selected.PREV/'spatial'/f'{ds}_sealed.json',selected.D/f'components_{ds}_sealed.json',selected.D/f'components_{ds}_reporting_receipt.json',selected.D/'components'/ds/'metrics.json',STRUCT/f'spatial_{ds}_sealed.json',STRUCT/f'spatial_{ds}_reporting_receipt.json',STRUCT/'spatial'/ds/'metrics.json',MATCH/ds/'sealed.json',MATCH/ds/'reporting_receipt.json',MATCH/ds/'metrics.json']
 p=dict(schema='native_Source_time_full_tube_controls_v1',authorization='2026-10-26 direct user requests complete cached computation, independent scoring and GitHub publication of whole-temporal-refinement deletion plus same-expert simple full-tube fusion.',cohorts=COUNTS,arms=ARMS,pairs=PAIRS,NoTemporalRefinement='Identical MAIN geometry/candidate tensors/content scores/link coefficients and same eight observations; bypass all temporal refinement and joint readout. Use original native per-offset Source MAP/hull event, then rerun interval-conditioned final spatial DP. Interval-dependent eligibility and observed/unobserved/edge normalization use the original code and fixed parameters.',EightReplacementNative='Copy raw Source trajectory, replace exactly eight Source-selected observed positions with the same native expert boxes, all other boxes exactly Source; original native Source interval, no posterior/DP.',SharedCorrectionNative='Same eight expert boxes/masks and original shared-corrected E trajectory, original native Source interval, no posterior or trajectory DP.',ExpertInterpolationNative='Original center/logsize same-eight interpolation, E outside observation coverage, original native Source interval; additional no-update context matched to completed SourceExpertInterpolation.',FullMAP='Current frozen direct-shared-ROI kappa0 beta1 own posterior/legal joint-MAP interval/final spatial DP. First auxiliary DP remains deleted.',GT='Complete each domain prediction seal before separate official CPU scoring process; only Source and current Full inherit previously officially scored exact arrays. All FOUR controls independently scored and checked against available exact audited reference scalar rows. Separate all-array cold replay and mean audit.',report='tIoU/vIoU/sIoU/R.3/R.5, signed positive-negative outcomes, severe t/v union once >5pp/rescues, short<=5s/long, parent clustered10000bootstrap pointwise95CI. All target sets previously exposed; no tuning.',limits=dict(new_Source=0,new_expert=0,new_optimizer=0,new_GPU=0,CPU_jobs_for_this_study=1,CPU_total_max=2,CPU_affinity=[14,15],nice=19,IO='idle3',threads=1,CUDAhidden=True,min_RAM_GiB=8,min_disk_GiB=80,private_GiB=160,domain_hours=72),cost='Cached derivation/scoring/audit wall only, not deployment latency. Inherited Source/expert acquisition costs remain nonzero; Vid Source producer was shared with original TTA, never double-count.',code_sha256={str(f.relative_to(R)):sha(f)for f in [*S.glob('*.py'),R/'stvg/spatial.py',R/'stvg/prototype.py',R/'stvg/paper_panel.py',R/'stvg/readout.py',R/'stvg/prior.py',R/'stvg/metrics.py',R/'stvg/evaluation.py',R/'tools/full_score.py',R/'studies/map_paper_completion_v1/common.py',R/'studies/paper_followup_v1/common.py',R/'studies/shared_corrected_roi_v1/shared_common.py',R/'studies/no_shrink_main_v1/ns_common.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'})
 for f in S.glob('*.py'):
  q=D/'source_snapshot'/f.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
 print(p['fingerprint'],flush=True)

def preflight():
 import numpy as np,torch
 from core import derive
 torch.set_num_threads(1);validate();no_GT();proof=[]
 for ds in COUNTS:
  for row in records(ds)[:2]:
   a,x=derive(ds,row);b,y=derive(ds,row);assert x==y and all(np.array_equal(v,b[k])for k,v in a.items());proof.append(dict(dataset=ds,all_arrays_repeat_exact=True,native_time_exact=True,full_current_exact=True,audited_structural_NoTemporal_exact=True,original_OursS_exact=True,shared_geometry_immutable=True,same_eight_direct_replacement_exact=True))
 assert not torch.cuda.is_initialized();z=dict(status='passed_four_actual_cached_cases',cases=proof,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0);atomic(D/'preflight.json',z);atomic(P/'preflight.json',z);print(z['status'],flush=True)

def predict(ds):
 import numpy as np,torch
 from core import derive
 torch.set_num_threads(1);p=validate();no_GT();assert read(D/'preflight.json')['status']=='passed_four_actual_cached_cases';dest=D/ds;assert not(dest/'predict_worker.json').exists();atomic(dest/'predict_worker.json',identity());files={};start=time.perf_counter()
 for i,row in enumerate(records(ds),1):
  wait_resources(start);t=time.perf_counter();a,proof=derive(ds,row);out=dest/'predictions'/(row['id']+'.npz');rp=dest/'receipts'/(row['id']+'.json');assert not out.exists()and not rp.exists();out.parent.mkdir(parents=True,exist_ok=True)
  with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(out.with_suffix('.tmp'),out)
  with np.load(out)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(out.relative_to(D)),sha256=sha(out),provenance=proof,worker=identity(),cached_query_wall_seconds=time.perf_counter()-t,GT_read=False,fingerprint=p['fingerprint']);atomic(rp,rec);files[row['id']]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
  if i%25==0:atomic(dest/'progress.json',dict(stage='predict',completed=i,total=COUNTS[ds],wall_seconds=time.perf_counter()-start,GT_read=False))
 assert len(files)==COUNTS[ds];atomic(dest/'sealed.json',dict(queries=len(files),files=files,GT_read=False,worker=identity(),wall_seconds=time.perf_counter()-start,fingerprint=p['fingerprint']));print(ds,'complete prediction seal',flush=True)

def score(ds):
 import numpy as np,torch
 from full_score import metric
 torch.set_num_threads(1);validate();dest=D/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];assert not(dest/'score_receipt.json').exists();truth=selected.labels(ds);old={r['id']:r for r in read(selected.D/'components'/ds/'metrics.json')['private_rows']};struct={r['id']:r for r in read(STRUCT/'spatial'/ds/'metrics.json')['private_rows']};matched={r['id']:r for r in read(MATCH/ds/'metrics.json')['private_rows']};rows=[];checks=0;maxerr=0.;start=time.perf_counter();atomic(dest/'score_worker.json',identity())
 for i,row in enumerate(records(ds),1):
  wait_resources(start);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={a:old[row['id']]['values'][b]for a,b in [('Source','Source'),('FullMAP','FullMAP')]}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a in NEW_ARMS:
    vals[a],err=metric(z[a],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8']);checks+=1;maxerr=max(maxerr,err);assert err<2e-10
   for a,ref in [('NoTemporalRefinement',struct[row['id']]['values']['NoTemporalInference']),('ExpertInterpolationNative',matched[row['id']]['values']['SourceExpertInterpolation'])]:
    for k,v in vals[a].items():assert(v is None and ref[k]is None)or abs(v-ref[k])<1e-12,(ds,a,k,v,ref[k])
   assert all(vals[a]['t']==vals['Source']['t']for a in NEW_ARMS)
  rows.append(dict(id=row['id'],parent=str(g['parent']),duration_group=old[row['id']]['duration_group'],values=vals))
  if i%100==0:atomic(dest/'score_progress.json',dict(completed=i,total=COUNTS[ds],checks=checks,wall_seconds=time.perf_counter()-start))
 report=selected.suite().parent_report(rows,ARMS,PAIRS,P/ds);selected.suite().signed_tails(rows,PAIRS,P/ds);atomic(dest/'metrics.json',dict(private_rows=rows,report=report));rec=dict(status='scored_whole_domain',queries=len(rows),checks=checks,max_error=maxerr,whole_domain_sealed_before_GT=True,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),inherited_exact_scored_arms=['Source','FullMAP'],independently_scored_arms=NEW_ARMS,all_NoTemporal_metrics_match_prior_exact_control=True,all_interpolation_metrics_match_prior_full_tube_control=True,worker=identity(),wall_seconds=time.perf_counter()-start);atomic(dest/'score_receipt.json',rec);atomic(P/ds/'score_receipt.json',rec);print(ds,'official score complete',flush=True)

def audit(ds):
 import numpy as np,torch
 from core import derive
 torch.set_num_threads(1);validate();no_GT();dest=D/ds;seal=read(dest/'sealed.json');score=read(dest/'score_receipt.json');assert score['seal_sha256']==sha(dest/'sealed.json')and score['metrics_sha256']==sha(dest/'metrics.json');assert not(dest/'reporting_receipt.json').exists();start=time.perf_counter();segments={};atomic(dest/'audit_worker.json',identity())
 for i,row in enumerate(records(ds),1):
  wait_resources(start);a,proof=derive(ds,row);e=seal['files'][row['id']];assert proof==e['provenance']and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
  with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  key=f"{e['worker']['pid']}:{e['worker']['startticks']}";segments[key]=segments.get(key,0.)+e['cached_query_wall_seconds']
  if i%100==0:atomic(dest/'audit_progress.json',dict(completed=i,total=COUNTS[ds],wall_seconds=time.perf_counter()-start,GT_read=False))
 rows=read(dest/'metrics.json')['private_rows'];report=read(P/ds/'report.json');assert len(rows)==COUNTS[ds]
 for a,mm in report['means'].items():
  for k,v in mm.items():
   x=[r['values'][a][k]for r in rows if r['values'][a][k]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 rec=dict(status='verified_whole_domain',queries=len(rows),prediction_receipt_SHA_checks=2*len(rows),all_arrays_cold_rederived_exact=True,all_means_recomputed=True,same_MAIN_geometry_and_DP_parameters=True,all_four_controls_use_original_native_Source_interval=True,score_sha256=sha(dest/'score_receipt.json'),seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),new_Source=0,new_expert=0,new_optimizer=0,new_GPU=0,worker=identity(),wall_seconds=time.perf_counter()-start,cached_query_wall_segments=segments,cost_scope='Cached conditional wall, not end-to-end deployment; inherited Source/expert cost nonzero, shared Vid producer not double-counted.');atomic(dest/'reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec);print(ds,'complete independent cold audit',flush=True)

def all_domains():
 validate();assert not(D/'owner.json').exists();atomic(D/'owner.json',identity())
 try:
  for ds in COUNTS:
   for command in ['predict','score','audit']:
    atomic(D/'state.json',dict(stage=command,dataset=ds,owner=identity()));log=(D/(ds+'_'+command+'.log')).open('xb');child=subprocess.Popen([sys.executable,'-B',str(S/'run.py'),command,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/(ds+'_'+command+'_child.json'),identity(child.pid));rc=child.wait();log.close();atomic(D/(ds+'_'+command+'_exit.json'),dict(code=rc));assert rc==0,'No automatic retry, preserve all prior bytes'
  atomic(D/'state.json',dict(stage='both_domains_scored_audited_pending_root_analysis_publication',owner=identity()))
 except BaseException:atomic(D/'state.json',dict(stage='failed_preserved',error=traceback.format_exc(),owner=identity()));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','preflight','predict','score','audit','all_domains']);ap.add_argument('--dataset',choices=COUNTS);v=ap.parse_args();globals()[v.command](*([]if v.command in ['prepare','preflight','all_domains']else[v.dataset]))
