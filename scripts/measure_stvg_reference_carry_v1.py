"""Independent saved state-chain arithmetic and postseal LN-stream metrics."""
import os,sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_reference_recovery_common_v1 import BASE,PARENT,read,write,sha,load_case
from scripts.measure_stvg_reference_recovery_v1 import fit_readback,verify

def root():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from scripts import stvg_target_teacher_student_v1 as core
 from scripts.run_stvg_table1_student004 import stable_fit_key
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.stvg_native_instance_common_v1 import guard
 verify(BASE);sys.addaudithook(guard)
 bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==96 and not bar['GT_read']
 rows={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']}
 expected={str((BASE/'student'/e['dataset']/e['order']/(c+'.pt')).relative_to(ROOT))for e in read(BASE/'ORDERS.json')['records']for c in e['case_ids']}
 assert {x['path']for x in bar['files']}==expected
 parentitems={x['path']:x for x in read(PARENT/'STUDENT_GLOBAL_BARRIER.json')['files']}
 pins=[];fit_calls=with_reference=steps=checks=changed_arrivals=changed_commits=first_equal=0;max_r=max_q=0.;chain=[]
 for entry in read(BASE/'ORDERS.json')['records']:
  ds,order=entry['dataset'],entry['order'];prev=prevsha=None;stream=[]
  spath=BASE/'student'/ds/order/'STREAM_BARRIER.json';sb=read(spath);assert sb['case_ids']==entry['case_ids'] and sb['count']==16
  for pos,cid in enumerate(entry['case_ids']):
   path=BASE/'student'/ds/order/(cid+'.pt');item=next(x for x in bar['files']if x['path']==str(path.relative_to(ROOT)));assert sha(path)==item['sha256']
   d=torch.load(path,map_location='cpu',weights_only=False);case,sourcepath=load_case(rows[cid]);fit=d['fit'];initial=d['initial_state'];source=case['initial_state'] if 'initial_state'in case else None
   # Source light state is frozen in the transition, checked independently
   # against the original parent fitter whenever that query has evidence.
   tr=d['state_transition'];source=tr['source_state_before_stream'];assert core._tree_equal(source,case['source_light_initial_state']);incoming=tr['incoming_state_before_fit']
   assert d['source_case_sha256']==case['source_case_sha256'] and d['pixel_sha256']==case['pixel_sha256'] and d['source_model_state_sha256']==case['source_model_state_sha256']
   assert d['order']==order and d['dataset']==ds and d['stream_position']==pos and d['source_restored'] and not d['GT_read']
   assert d['previous_result_sha256']==prevsha and core._tree_equal(initial,incoming) and core._tree_equal(initial,fit['initial_state'])
   assert torch.count_nonzero(initial['spatial.query_residual'])==0
   if prev is None:
    for n in initial:assert torch.equal(initial[n],torch.zeros_like(source[n])if n=='spatial.query_residual'else source[n])
   else:
    assert core._tree_equal(d['committed_before'],prev)
    for n in initial:assert torch.equal(initial[n],torch.zeros_like(source[n])if n=='spatial.query_residual'else prev[n])
   changed_arrivals+=int(any(not torch.equal(initial[n],source[n])for n in initial if n!='spatial.query_residual'))
   assert state_hash(initial)==d['initial_state_sha256'] and state_hash(d['next_committed'])==d['next_state_sha256']
   assert fit['profile']=='current5' and fit['config']==read(BASE/'DESIGN_LOCK.json')['configs'][ds]
   assert fit['seed_key']==stable_fit_key(ds,rows[cid]['query_key_sha256']) and d['reference_anchors']==refs[cid]['anchors']
   rr=fit_readback(fit,refs[cid]['anchors']);checks+=rr['reward_scalars'];max_r=max(max_r,rr['max_reward_error']);max_q=max(max_q,rr['max_q_error'])
   alpha=fit['config']['writeback']
   for n in initial:
    expectednext=torch.zeros_like(initial[n])if n=='spatial.query_residual'else initial[n]+alpha*(fit['final_state'][n]-initial[n])
    assert torch.equal(expectednext,d['next_committed'][n]),(ds,order,cid,n)
   assert core._tree_equal(d['next_committed'],tr['next_stream_state'])
   changed_commits+=int(any(not torch.equal(initial[n],d['next_committed'][n])for n in initial))
   for name,key in [('Before_Native','initial_cached_decoder_boxes'),('After_Native','final_cached_decoder_boxes')]:
    out=d['readouts'][name];assert torch.equal(out['boxes'].cpu(),fit[key]);b=out['boxes'].cpu().numpy().astype(float);xy=np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],axis=1);assert np.array_equal(xy,d['arms'][name]['boxes'])
   pp=PARENT/'student'/(cid+'.pt');assert sha(pp)==parentitems[str(pp.relative_to(ROOT))]['sha256']==d['temporal_reuse']['parent_prediction_sha256'];p=torch.load(pp,map_location='cpu',weights_only=False)
   for a,b in [('F','F'),('T','T'),('Episodic_Native','S_new_crop'),('Episodic_T','TS_new_crop')]:
    assert np.array_equal(d['arms'][a]['boxes'],p['arms'][b]['boxes']) and d['arms'][a]['interval']==p['arms'][b]['interval']
   if refs[cid]['anchors']:
    assert core._tree_equal(source,p['fits']['new_crop']['initial_state']);with_reference+=1
   if pos==0:
    assert np.array_equal(d['arms']['After_Native']['boxes'],d['arms']['Episodic_Native']['boxes']);first_equal+=1
   if not refs[cid]['anchors']:assert fit['optimizer_steps']==0 and torch.equal(fit['initial_cached_decoder_boxes'],fit['final_cached_decoder_boxes'])
   fit_calls+=1;steps+=fit['optimizer_steps'];prev=d['next_committed'];prevsha=sha(path)
   stream.append(dict(case_id=cid,incoming_sha256=d['initial_state_sha256'],next_sha256=d['next_state_sha256'],optimizer_steps=fit['optimizer_steps']))
   pins.extend([item,dict(path=str(sourcepath.relative_to(ROOT)),sha256=sha(sourcepath)),dict(path=str(pp.relative_to(ROOT)),sha256=sha(pp))])
  chain.append(dict(dataset=ds,order=order,count=16,state_chain=stream));pins.append(dict(path=str(spath.relative_to(ROOT)),sha256=sha(spath)))
 result=dict(status='pass',stream_count=6,arrival_count=96,independent_source_count=32,actual_fit_invocations=fit_calls,with_reference_fit_count=with_reference,empty_reference_count=fit_calls-with_reference,accepted_optimizer_steps=steps,actual_full_readout_calls=192,changed_incoming_LN_count=changed_arrivals,changed_commit_count=changed_commits,first_source_equals_episodic_count=first_equal,independent_sampled_reward_scalar_count=checks,max_reward_error=max_r,max_q_error=max_q,CPU_only=True,GT_read=False,scope='exact six state chains/Query reset/fresh Adam/original alpha/all seven tensors/saved rewards and full-readout agreement; no independent backward replay',records=chain)
 write(BASE/'ROOT_READBACK.json',result)
 for name in ['ROSTER_LOCK.json','REFERENCE_LOCK.json','DESIGN_LOCK.json','ORDERS.json','RUNTIME.json','ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json']:
  p=BASE/name;pins.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=96,independent_source_count=32,files=pins,scorer_sha256=sha(__file__),GT_read=False))
 print({k:v for k,v in result.items()if k!='records'})

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from scripts import review_stvg_sparse_r1_v2 as scorer
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 verify(BASE);seal=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert seal['status']=='complete' and seal['count']==96 and seal['scorer_sha256']==sha(__file__) and not seal['GT_read']
 for p in seal['files']:assert sha(ROOT/p['path'])==p['sha256']
 cases={r['probe_alias']:load_case(r)[0]for r in read(BASE/'ROSTER_LOCK.json')['records']}
 write(BASE/'GT_EXPOSURE.json',dict(status='after_all_six_stream_predictions_and_root',seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),GT_read=True,time=time.time()))
 truth,provenance=scorer._truths_after_seal(cases);official,independent,_=scorer._metric_functions();rows=[]
 for item in read(BASE/'STUDENT_GLOBAL_BARRIER.json')['files']:
  d=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False);cid=d['case_id'];support={a['position']for a in d['reference_anchors']}
  for arm,pred in d['arms'].items():
   values,audit=metric._score_student_arm(pred,cases[cid],truth[cid],support,official,independent);rows.append(dict(case_id=cid,dataset=d['dataset'],order=d['order'],stream_position=d['stream_position'],arm=arm,metrics=values,support=audit['support'],metric_audit=audit['metric_audit']))
 summary=[];paired=[]
 for ds in ['hc2','vidstg']:
  for order in ['order1','order2','order3']:
   rr=[r for r in rows if r['dataset']==ds and r['order']==order];ids=sorted({r['case_id']for r in rr});lookup={(r['case_id'],r['arm']):r for r in rr};assert len(ids)==16
   for arm in sorted({r['arm']for r in rr}):
    for key in rr[0]['metrics']:
     vals=[lookup[c,arm]['metrics'][key]for c in ids if lookup[c,arm]['metrics'][key]is not None];summary.append(dict(dataset=ds,order=order,arm=arm,metric=key,count=len(vals),mean=math.fsum(vals)/len(vals)if vals else None))
   for a,b in [('Before_Native','F'),('After_Native','Before_Native'),('After_Native','Episodic_Native'),('Before_T','T'),('After_T','Before_T'),('After_T','Episodic_T')]:
    for key in ['m_vIoU','m_sIoU_dense_GT']:
     if key not in rr[0]['metrics']:continue
     delta={c:100*(lookup[c,a]['metrics'][key]-lookup[c,b]['metrics'][key])for c in ids if lookup[c,a]['metrics'][key]is not None and lookup[c,b]['metrics'][key]is not None};v=list(delta.values());paired.append(dict(dataset=ds,order=order,left=a,right=b,metric=key,count=len(v),mean_delta_pp=math.fsum(v)/len(v)if v else None,positive=sum(x>1e-10 for x in v),negative=sum(x< -1e-10 for x in v),neutral=sum(abs(x)<=1e-10 for x in v),worst=min(v)if v else None,best=max(v)if v else None,per_source_delta_pp=delta))
 write(BASE/'POSTSEAL_ROWS.json',dict(status='complete',rows=rows,GT_read=True));write(BASE/'POSTSEAL_SUMMARY.json',dict(status='complete',summary=summary,GT_read=True));write(BASE/'PARENT_PAIRED_DELTAS.json',dict(status='complete',records=paired,GT_read=True,bootstrap_executed=False,independent_source_count=32,orders_are_repeated_sources=True))
 print('six orders scored',len(rows),len(summary),len(paired))
if __name__=='__main__':{'root':root,'score':score}[sys.argv[1]]()
