"""Whole-domain TTA baselines, independent prediction/official score/readout audit."""
from full_common import *
import argparse,traceback,gc
ARMS=['Source','TentNative','TentJointMAP','SARNative','SARJointMAP','SourceJointMAP','FullMAP']
PAIRS=[(a,'Source')for a in ['TentNative','TentJointMAP','SARNative','SARJointMAP']]+[('FullMAP',a)for a in ARMS if a!='FullMAP']
def prepare():
 assert not(D/'plan.json').exists();check();contexts_index={ds:{}for ds in COUNTS};reuse={ds:{}for ds in COUNTS};support=read(OLD/'supports_sealed.json');legacy=read(PREV/'published_TTA/sealed.json');entropy=read(R/'outputs/entropy_tta_v2/sealed.json');expected={ds:{r['id']:r for r in records(ds)}for ds in COUNTS}
 for ds,groups in support['files'].items():
  for stage,files in groups.items():
   for key,e in files.items():assert key in expected[ds];contexts_index[ds][key]=dict(kind='shared_native',path=str(OLD/e['shared_path']),sha256=e['shared_sha256'],origin_seal_sha256=sha(OLD/'supports_sealed.json'),stage=stage)
 for ds,files in entropy['files'].items():
  selected={r['id']:r for r in read(R/'outputs/entropy_tta_v2/selection.json')[ds]}
  for key,e in files.items():
   assert selected[key]==expected[ds][key]
   if key not in contexts_index[ds]:contexts_index[ds][key]=dict(kind='native_temporal',path=str(R/'outputs/entropy_tta_v2'/e['native_path']),sha256=e['native_sha256'],origin_seal_sha256=sha(R/'outputs/entropy_tta_v2/sealed.json'),parent_prediction_sha256=e['source_parent_SHA'])
 for ds,files in legacy['files'].items():
  for key,e in files.items():assert key in expected[ds]and key in contexts_index[ds];reuse[ds][key]=e
 atomic(D/'context_index.json',contexts_index);atomic(D/'reuse_index.json',reuse);baseline().official();inputs=[K0/'plan.json',R/'CURRENT_METHOD.json',R/'profiles/no_shrink_map_v1.json',RUN/'plan.json',PREV/'published_TTA_plan.json',PREV/'published_TTA/sealed.json',PREV/'published_TTA/score_receipt.json',PREV/'published_TTA/reporting_receipt.json',R/'outputs/paper_followup_v1/tta_selected.json',R/'outputs/paper_followup_v1/reference_vendor/audit.json',OLD/'supports_sealed.json',R/'outputs/entropy_tta_v2/sealed.json',D/'context_index.json',D/'reuse_index.json',PREV/'spatial_plan.json',PREV/'spatial_runtime_v2_plan.json']
 for ds in COUNTS:inputs += [RUN/ds/'sealed.json',PREV/f'temporal_v2_{ds}_sealed.json',PREV/f'temporal_v2_{ds}_reporting_receipt.json']
 code=[*S.glob('*.py'),R/'studies/no_shrink_main_v1/ns_common.py',R/'studies/no_shrink_main_v1/run.py',R/'studies/no_shrink_main_v1/timing.py',R/'studies/map_paper_completion_v1/tta.py',R/'studies/map_paper_completion_v1/spatial.py',R/'studies/map_paper_completion_v1/spatial_runtime_v2.py',R/'studies/map_paper_completion_v1/spatial_core.py',R/'studies/map_paper_completion_v1/common.py',R/'studies/entropy_tta_v2/run.py',R/'stvg/temporal_hidden.py',R/'stvg/native.py',R/'stvg/current.py',R/'stvg/readout.py',R/'tools/full_score.py',*[(R/'outputs/paper_followup_v1/reference_vendor/mr-eggplant_SAR')/(n+'.py')for n in ['tent','sar','sam']]]
 p=dict(schema='official_Tent_SAR_full_targets_v1',authorization='User explicitly requests full-target TTA baselines. Existing128query panels preserved. HC1902+Vid10303; exactlyTent/SAR same existing selected settings, no sweep.',cohort=COUNTS,config_results=24410,algorithms=dict(Tent='Pinned SAR-repository Tent implementation, SGD lr1e-4/momentum.9,5steps',SAR='Pinned SAR/SAMrho.05 lr1e-4/momentum.9,5steps/marginfraction.8/resetconstant.2'),scope='Temporal-decoder LayerNorm-affine STVG adapters only; cached early context/Source boxes remain fixed. Full target coverage is not full-network/canonicalBN/continual adaptation.0expert versus currentkappa0Ours8expert/0optimizer; disclose differing budgets.',readout='Primary original native per-offset legalMAP+physicalhull. Secondary jointMAP isolatesreadout. CurrentOurs is independentlyauditedkappa0FullMAP, noMBR comparison.',reset='Full decoder parameters/grads, optimizer/SARstate reset per query/method, native logits bitwise before adaptation and afterreset; actual selected affine weights/logits/entropy traces saved.',existing_contexts={ds:len(v)for ds,v in contexts_index.items()},existing_completed_queries={ds:len(v)for ds,v in reuse.items()},maximum_additional_Source={ds:COUNTS[ds]-len(contexts_index[ds])for ds in COUNTS},maximum_additional_Source_total=sum(COUNTS[ds]-len(contexts_index[ds])for ds in COUNTS),reuse='Exact existing256query512config NPZ/logits bytes+SHA+trial/reset/readout checks; inherited optimizer/source costs retained, not deployedforfree. Other existing contexts used after native input/layout/logit+parent inputbinding verification.',acquisition='No interruption of activeHCgridcollector. After its successful exit, run fullTTAHC then existingcurrenttiming; Vidgrid+TTA run as one versionedhookedworker. The original unmodified Sourceforward produces both actual grids and actual7temporalkwargs; context andSHA saved before updates; exact model reset/Torch4 numerical context restored before original geometry/repool. Missing contexts only use one additional originalSource224query; no teacher. No posthoc synthesis from spatial features or outputboxes.',compact='Only this new active acquired temporal input may be removed after committedNPZ+adaptedlogits+weights+receipt, readbackallreadouts and reset proof. Preserve originalSHA/disposalreceipt and unresolvedpaidattempts. Does not preserve arbitrary futureoptimizer/contextreplay. All old caches immutable.',evaluation='Whole dataset1902or10303seal before independentCPUofficiallabels; allpositive/negative/severeuniononce/rescue/shortlong/parentclusterCI/logitchange/reliablefilter/cost. All fullsets historically exposed; no selection/retuning from results.',limits=dict(CPU_jobs=2,CPU_affinity=[14,15],nice=19,io='idle3',minimum_RAM_bytes=8*2**30,minimum_disk_free_bytes=80*2**30,maximum_private_bytes=160*2**30,maximum_domain_seconds=259200,GPU_workers=1,new_expert=0),failure='Any native/reset/hash/nonfinite mismatch hardstops and preserves code/plan/logs/partialpred/paidSourcecontext/attempt. No autoretry/skip/Sourcefallback/hashwaiver.',code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});
 for f in code:q=D/'source_snapshot'/f.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
 atomic(D/'state.json',dict(status='prepared_full_targets',fingerprint=p['fingerprint']));print(p['fingerprint'],p['existing_contexts'],p['maximum_additional_Source_total'])
def preflight():
 import torch,numpy as np
 from stvg.runtime import guard
 from stvg.temporal import probabilities,mass_table
 from stvg.readout import prediction
 from stvg.current import map_interval
 sys.addaudithook(guard);torch.set_num_threads(1);p=validate();proof=[]
 for ds in COUNTS:
  lookup={r['id']:r for r in records(ds)};keys=list(fixed(D/'context_index.json')[ds])[:2]
  for key in keys:
   c,e=payload_context(ds,key);cap,par=parent(ds,lookup[key]);check_context(c,cap);proof.append(dict(dataset=ds,context_sha256=e['sha256'],parent=par,actual_seven_kwargs_layouts_native_logits_frames_exact=True))
  for key,e in fixed(D/'reuse_index.json')[ds].items():
   for f,h in [('path','sha256'),('logits_path','logits_sha256'),('receipt_path','receipt_sha256')]:assert sha(PREV/e[f])==e[h]
   logs=torch.load(PREV/e['logits_path'],map_location='cpu',weights_only=False,mmap=True)
   with np.load(PREV/e['path'])as z:
    for m in ['Tent','SAR']:
     assert e['traces'][m]['trial']==fixed(R/'outputs/paper_followup_v1/tta_selected.json')[ds][m]and e['traces'][m]['reset_exact'];iv=prediction(logs['adapted_logits'][m],torch.as_tensor(logs['Source']),logs['records'],logs['frame_ids'])['physical_interval'];_,q=mass_table(probabilities(logs['records'],logs['adapted_logits'][m]));assert np.array_equal(z[m+'Native_interval'],iv)and np.array_equal(z[m+'JointMAP_interval'],map_interval(logs['frame_ids'],q))
 assert not torch.cuda.is_initialized();rec=dict(status='passed',real_cached_input_cases=4,context_proof=proof,reused_completed_queries=256,reused_configs=512,all_reused_files_SHA_trials_resets_readouts_exact=True,new_Source=0,new_expert=0,new_GPU=0,new_optimizer=0,new_GT_scoring=0,full_real_optimizer_entry='First nonreused query must pass nativeGPU/actualupdates/resetbitwise before any furtherqueries; full GPU not started while activecollector owns lock.',fingerprint=p['fingerprint']);atomic(D/'preflight_receipt.json',rec);atomic(P/'preflight_receipt.json',{k:v for k,v in rec.items()if k!='context_proof'});print('FullTTA input/reuse preflight passed')
def predict(ds):
 import torch
 from stvg.runtime import setup,lease
 from engine import Engine
 validate();assert read(D/'preflight_receipt.json')['status']=='passed';assert not(D/f'{ds}_worker.json').exists();setup();leasefile=lease();atomic(D/f'{ds}_worker.json',identity());e=Engine(ds)
 try:
  for row in records(ds):e.process(row)
  seal(ds)
 except BaseException:atomic(D/f'{ds}_prediction_state.json',dict(status='failed_preserved',error=traceback.format_exc(),actual_counts=e.actual));raise
 finally:e.close();leasefile.close()
def score(ds):
 import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);validate();sp=D/f'{ds}_sealed.json';ss=read(sp);assert ss['queries']==COUNTS[ds]and not ss['GT_read'];truth=labels(ds);old={x['id']:x for x in read(PREV/'temporal_v2_v2'/ds/'metrics.json')['private_rows']};rr=[];checks=0;errmax=0.
 for row in records(ds):
  e=ss['files'][row['id']];assert existing_entry(ds,row['id'])==e;g=truth[row['id']];vals={'FullMAP':old[row['id']]['values']['NoShrink'],'SourceJointMAP':old[row['id']]['values']['SourceJointMAP']}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ARMS[:5]:vals[arm],err=metric(z[arm],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;checks+=1;errmax=max(errmax,err)
  assert vals['Source']==old[row['id']]['values']['Source'];rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group=old[row['id']]['duration_group'],values=vals))
 res=ns.suite().parent_report(rr,ARMS,PAIRS,P/ds);ns.suite().signed_tails(rr,PAIRS,P/ds);atomic(D/ds/'metrics.json',dict(private_rows=rr,report=res));rec=dict(status='scored',checks=checks,max_error=errmax,queries=len(rr),whole_domain_sealed_before_GT=True,seal_sha256=sha(sp));atomic(D/f'{ds}_score_receipt.json',rec);atomic(P/ds/'score_receipt.json',rec)
def audit(ds):
 import torch,numpy as np
 from stvg.runtime import guard
 from stvg.temporal import probabilities,mass_table
 from stvg.readout import prediction
 from stvg.current import map_interval
 sys.addaudithook(guard);torch.set_num_threads(1);validate();ss=read(D/f'{ds}_sealed.json');assert read(D/f'{ds}_score_receipt.json')['seal_sha256']==sha(D/f'{ds}_sealed.json');lookup={r['id']:r for r in records(ds)};costs={};stats={m:dict(nonzero_parameter_delta_queries=0,changed_native_queries=0,changed_jointMAP_queries=0,empty_first_SAR_iterations=0,empty_second_after_first_SAR_iterations=0)for m in ['Tent','SAR']}
 for key,e in ss['files'].items():
  assert existing_entry(ds,key)==e;logs=torch.load(D/e['logits_path'],map_location='cpu',weights_only=False,mmap=True);cap,par=parent(ds,lookup[key]);assert par==e['parent']and np.array_equal(logs['Source'],cap['Source'])
  with np.load(D/e['path'])as z:
   for method,after in logs['adapted_logits'].items():
    iv=prediction(after,torch.as_tensor(cap['Source']),cap['records'],cap['frame_ids'])['physical_interval'];_,q=mass_table(probabilities(cap['records'],after));assert np.array_equal(z[method+'Native_interval'],iv)and np.array_equal(z[method+'JointMAP_interval'],map_interval(cap['frame_ids'],q))and np.array_equal(z[method+'Native'],cap['Source']);tr=e['traces'][method];assert tr['reset_exact'];stats[method]['nonzero_parameter_delta_queries']+=int(tr['parameter_delta']>0);stats[method]['changed_native_queries']+=int(not np.array_equal(z[method+'Native_interval'],z['Source_interval']));stats[method]['changed_jointMAP_queries']+=int(not np.array_equal(z[method+'JointMAP_interval'],map_interval(cap['frame_ids'],cap['native_prior'])))
    if method=='SAR':
     import math
     margin=tr['trial']['margin_fraction']*math.log(max(x.shape[1]for x in after))
     for step in tr['steps']:
      f=step['forwards'];a=np.asarray(f[0]['entropy']);b=np.asarray(f[1]['entropy']);stats[method]['empty_first_SAR_iterations']+=int(not(a<margin).any());stats[method]['empty_second_after_first_SAR_iterations']+=int(not((a<margin)&(b<margin)).any())
  w=e['worker'];s=costs.setdefault(f"{w['pid']}:{w['startticks']}",dict(queries=0,reused_completed=0,new_Source=0,shared_Source=0,new_optimizer_iterations=0,adaptation_seconds=0.));s['queries']+=1;s['reused_completed']+=e['reused_completed'];s['new_Source']+=e['actual_new_counts']['new_Source_queries'];s['shared_Source']+=e['actual_new_counts']['shared_Source_queries'];s['new_optimizer_iterations']+=e['actual_new_counts']['optimizer_iterations'];s['adaptation_seconds']+=sum(t['seconds']for t in e['traces'].values())if not e['reused_completed']else 0.
 rr=read(D/ds/'metrics.json')['private_rows'];res=read(P/ds/'report.json')
 for arm,mm in res['means'].items():
  for f,v in mm.items():
   vv=[r['values'][arm][f]for r in rr if r['values'][arm][f]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 rec=dict(status='verified_full_domain_Tent_SAR',queries=COUNTS[ds],configs=2*COUNTS[ds],SHA_checks=3*COUNTS[ds],all_native_joint_readouts_replayed_exact=True,all_native_Source_parent_arrays_exact=True,all_episodic_resets_exact=True,all_means_recomputed=True,counts=ss['counts'],worker_cost_segments=costs,diagnostics=stats,score_sha256=sha(D/f'{ds}_score_receipt.json'),new_GT_scoring=0,new_expert=0,scope='Only disclosed temporal-decoderLN adapter, not canonicalBN/fullnetwork/continual. Prior acquisition/optimizer work reused with provenance, notzero deployment. Shared_Source charged once to producer; nested times not added.');atomic(D/f'{ds}_reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec);atomic(D/f'{ds}_state.json',dict(status='complete_scored_audited_pending_root_analysis_publication'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','predict','score','audit']);a.add_argument('--dataset',choices=COUNTS);z=a.parse_args();globals()[z.command]()if z.command in ['prepare','preflight']else globals()[z.command](z.dataset)
