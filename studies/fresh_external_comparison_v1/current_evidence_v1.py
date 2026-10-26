"""Same original256 inputs; strict reuse or authorized actual current-expert call."""
from pathlib import Path
import os,sys,time,json,hashlib,gc,traceback
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'studies/expert_generalization_v1'))
from eg_common_v1 import *
import eg_math_core_v1 as math
from eg_pilot_v1 import validate_pilot,save
N=R/'outputs/fresh_external_comparison_v1';PUB=R/'results/fresh_external_comparison_v1'

def validate():
 validate_pilot();p=read(N/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p

def inputs(ds,row):
 import torch,numpy as np
 from stvg.decode import decode
 from stvg.teacher_bridge import evidence_key
 cap,origin=rc.selected.suite().load_parent(ds,row);e=fixed(R/'outputs/map_paper_completion_v1/raw_reuse_index.json')[ds][row['id']];assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);assert raw['parent_capture_sha256']==origin['capture_sha256']and raw['frame_ids']==cap['frame_ids']and not raw['GT_read'];frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];pos=cap['positions8'];key,context=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,fingerprint=fixed(rc.RUN/'plan.json')['provider_runtime_fingerprint'],dataset=ds)
 proof=dict(parent=origin,retained_raw_path=e['path'],retained_raw_sha256=e['sha256'],actual_native_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),actual_eight_RGB_sha256=hashlib.sha256(frames[pos].tobytes()).hexdigest(),evidence_key=key,context=context,positions8=pos,frame_ids=ids,GT_read=False)
 return cap,raw,frames,proof

def existing(proof):
 roots=[rc.PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache'];key=proof['evidence_key'];hits=[p for p in roots if(p/(key+'.json')).exists()]
 if not hits:return None
 tp=hits[0]/(key+'.pt');m=read(tp.with_suffix('.json'));assert m['key']==key and m['context']==proof['context']and sha(tp)==m['sha256'];return tp

def run():
 import torch,numpy as np
 from stvg.runtime import setup,lease
 from stvg.teacher_bridge import ResidentTeacher
 p=validate();setup();torch.set_num_threads(1);rc.gt_guard();assert not(N/'worker_v1.json').exists();who=identity();atomic(N/'worker_v1.json',who);start=time.perf_counter();lock=lease();teacher=ResidentTeacher(cache_dir=N/'new_current_expert_cache',fingerprint=fixed(rc.RUN/'plan.json')['provider_runtime_fingerprint']);counts=dict(committed=0,reused_current_expert=0,new_current_expert=0,new_Source=0,new_optimizer=0);fresh=[]
 try:
  for ds in ['hc2','vidstg']:
   files={};folder=N/ds
   for i,row in enumerate(pilot_rows(ds),1):
    resources(start);assert sum(f.stat().st_size for f in N.rglob('*')if f.is_file())<160*2**30;cap,raw,frames,proof=inputs(ds,row);key=row['id'];tick=time.perf_counter();attempt=folder/'attempts'/(key+'.json');assert not attempt.exists();old=existing(proof);new=old is None
    atomic(attempt,dict(status='before_original_expert_reuse_or_one_actual_call',new_current_expert_planned=new,input_binding=proof,worker=who,GT_read=False));call_start=time.perf_counter()
    if new:
     # ResidentTeacher checkpoints the original native payload and exact RGB
     # context receipt in this NEW namespace before returning to any ROI/math.
     counts['new_current_expert']+=1;assert counts['new_current_expert']<=p['limits']['max_new_current_expert_queries'];native=teacher(frames[cap['positions8']],row['metadata']['caption'],positions=cap['positions8'],frame_ids=cap['frame_ids'],dataset=ds);tp=N/'new_current_expert_cache'/(proof['evidence_key']+'.pt');assert native['query_forwards']==1
    else:
     tp=old;native=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);counts['reused_current_expert']+=1
    expert_request=time.perf_counter()-call_start;meta=read(tp.with_suffix('.json'));assert meta['key']==proof['evidence_key']and meta['context']==proof['context']and sha(tp)==meta['sha256'];checkpoint=dict(payload_path=str(tp),payload_sha256=sha(tp),receipt_path=str(tp.with_suffix('.json')),receipt_sha256=sha(tp.with_suffix('.json')),before_geometry_or_ROI=True,new_actual_current_expert=new)
    atomic(folder/'teacher_checkpoints'/(key+'.json'),checkpoint);t=time.perf_counter();arrays,compact=math.build(cap,raw,native,ds);finish=time.perf_counter()-t
    if not new:
     pe,ce=rc.paths(ds,key)
     with np.load(rc.RUN/ds/pe['path'])as a,np.load(rc.selected.D/ce['path'])as b:assert np.array_equal(arrays['Simple'],a['Interpolation'])and np.array_equal(arrays['Ours'],b['FullMAP'])and np.array_equal(arrays['Ours_interval'],b['FullMAP_interval'])
    cp=folder/'compact'/(key+'.pt');save(cp,compact);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=math.replay(cold);assert all(np.array_equal(v,again[k])for k,v in arrays.items());pred=folder/'predictions'/(key+'.npz');pred.parent.mkdir(parents=True,exist_ok=True)
    with pred.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
    os.replace(pred.with_suffix('.tmp'),pred)
    with np.load(pred)as z:assert all(np.array_equal(v,z[k])for k,v in arrays.items())
    rec=dict(path=str(pred.relative_to(N)),sha256=sha(pred),compact_path=str(cp.relative_to(N)),compact_sha256=sha(cp),input_binding=proof,current_teacher=checkpoint,new_current_expert=int(new),new_Source=0,new_optimizer=0,original_native_payload_cost=native.get('cost'),original_or_new_model_load_seconds=native.get('model_load_seconds'),actual_new_expert_process=identity(teacher.process.pid)if new else None,expert_request_elapsed_seconds=expert_request,matched_current_full_chain_seconds=finish,conditional_query_elapsed_seconds=time.perf_counter()-tick,all_arrays_cold_exact=True,reused_current_matches_published_selected_exact=not new,fresh_current_uses_genuinely_new_geometry_and_Epool=new,worker=who,GT_read=False,plan_fingerprint=p['fingerprint']);rp=folder/'receipts'/(key+'.json');atomic(rp,rec);e={**rec,'receipt_path':str(rp.relative_to(N)),'receipt_sha256':sha(rp)};files[key]=e;counts['committed']+=1;atomic(attempt,dict(status='committed',receipt_sha256=sha(rp)))
    if new:fresh.append(dict(dataset=ds,file=e))
    atomic(N/'progress.json',dict(dataset=ds,completed=i,total=128,counts=counts,worker=who,GT_read=False))
    if new and len(fresh)==2:
     wait=N/'first_two_new_current_expert_waiting.json';atomic(wait,dict(status='waiting_root_two_actual_new_current_expert_full_chain',cases=fresh,worker=who,GT_read=False));ack=N/'first_two_new_current_expert_bridge.json'
     while not ack.exists():resources(start);time.sleep(10)
     a=read(ack);assert a['status']=='passed_two_actual_new_current_expert_context_payload_fresh_math_cold_exact'and a['waiting_sha256']==sha(wait)
    del cap,raw,frames,native,arrays,compact,cold,again;gc.collect()
   assert len(files)==128;atomic(folder/'sealed.json',dict(status='all128_same_input_current_expert_baselines_prepared_no_GT',queries=128,files=files,counts=counts,worker=who,GT_read=False))
  atomic(N/'state_v1.json',dict(status='all256_current_expert_contexts_and_matched_baselines_prepared_pending_independent_audit',counts=counts,worker=who,elapsed_wall_seconds=time.perf_counter()-start,GT_read=False))
 except BaseException:atomic(N/'failure_v1.json',dict(error=traceback.format_exc(),counts=counts,worker=who,elapsed_wall_seconds=time.perf_counter()-start,GT_read=False));raise
 finally:teacher.close();lock.close()

def checked(ds,row,e):
 import torch,numpy as np
 cap,raw,frames,proof=inputs(ds,row);assert proof==e['input_binding'];t=e['current_teacher'];tp=Path(t['payload_path']);assert sha(tp)==t['payload_sha256']and sha(t['receipt_path'])==t['receipt_sha256'];meta=read(t['receipt_path']);assert meta['key']==proof['evidence_key']and meta['context']==proof['context'];native=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);a,_=math.build(cap,raw,native,ds);b,_=math.build(cap,raw,native,ds);assert all(np.array_equal(v,b[k])for k,v in a.items());assert sha(N/e['path'])==e['sha256']and sha(N/e['compact_path'])==e['compact_sha256'];compact=torch.load(N/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);cold=math.replay(compact)
 with np.load(N/e['path'])as z:assert all(np.array_equal(v,z[k])and np.array_equal(v,cold[k])for k,v in a.items())
 return dict(prediction_sha256=e['sha256'],teacher_payload_sha256=t['payload_sha256'],input_RGB_context_exact=True,actual_grid_Epool_newposterior_ownMAP_finalDP_two_repeats_cold_exact=True,new_Source=0,new_GT=0)

def bridge():
 import torch
 validate();torch.set_num_threads(1);rc.gt_guard();w=N/'first_two_new_current_expert_waiting.json';v=read(w);assert len(v['cases'])==2;start=time.perf_counter();proof=[]
 for item in v['cases']:
  ds=item['dataset'];e=item['file'];row=next(x for x in pilot_rows(ds)if x['id']==Path(e['path']).stem);assert e['new_current_expert']==1;proof.append(checked(ds,row,e))
 atomic(N/'first_two_new_current_expert_bridge.json',dict(status='passed_two_actual_new_current_expert_context_payload_fresh_math_cold_exact',waiting_sha256=sha(w),cases=proof,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_GT=0))

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('command',choices=['run','bridge']);z=a.parse_args();globals()[z.command]()
