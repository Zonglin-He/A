"""Actual decoder adaptation for each coarse/fine optimizer; GT is closed here."""
import sys,os,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE,read,put,sha,load,save,verify

def main(ds,phase):
 import numpy as np,torch
 from methods.stvg_single_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1.state import Replay
 from methods.stvg_referdino_opd_v1.policy import fit
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 setup();p=verify();lock=lease();start=time.perf_counter();prior=sum(x['wall_seconds']for x in read(D/'GPU_LEDGER.json')['invocations'])if(D/'GPU_LEDGER.json').exists()else 0.;count=updates=compat=0;files={};failure=None;m=None;cfgs=p['configs'][ds]if phase=='coarse'else read(D/ds/'FINE_PLAN_LOCK.json')['configs'];records=[r for r in p['records']if r['dataset']==ds]
 def check():resource_guard();assert prior+time.perf_counter()-start<p['cumulative_GPU_worker_wall_limit_seconds'],'bounded cumulative GPU worker wall'
 try:
  for f,h in p['checkpoints'].items():assert sha(ROOT/f)==h,f
  tick=time.perf_counter();m=load_source(p['configs'][ds][p['baseline']]);loading=time.perf_counter()-tick;mh=state_hash(m.state_dict());m.cpu();gc.collect();torch.cuda.empty_cache();m.ground_decoder.decoder.cuda()
  for rec in records:
   check();a=rec['alias'];inpfile=BASE/f'inputs/{ds}/{a}.pt';assert sha(inpfile)==p['source_files'][str(inpfile.relative_to(ROOT))];cache=load(inpfile);rp=Replay(m,cache,cache['Source'],cache['E'],cache['planned_positions'],check)
   try:
    for key,cfg in cfgs.items():
     dest=D/ds/'fits'/phase/key/f'{a}.pt'
     if phase=='coarse'and key==p['baseline']and ds=='hc2':
      old=BASE/f'traces/{ds}/{a}.pt';files[key+'/'+a]=dict(path=str(old.relative_to(ROOT)),sha256=sha(old),historical_reuse=True)
      if a==records[0]['alias']and not(D/ds/'BASELINE_COMPATIBILITY.json').exists():
       oldfit=load(old)['fit'];out=fit(rp,cache['teacher_native']['boxes'],a,'Live',41,cfg);updates+=out['actual_updates'];compat+=40
       assert all(torch.equal(x['state'],y['state'])and torch.equal(x['boxes'],y['boxes'])for x,y in zip(out['path'],oldfit['path']));put(D/ds/'BASELINE_COMPATIBILITY.json',dict(status='pass',all41_states_and_boxes_bitexact=True,new_updates=40));del out,oldfit
      continue
     if dest.exists():files[key+'/'+a]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),historical_reuse=False);continue
     out=fit(rp,cache['teacher_native']['boxes'],a,'Live',41,cfg);updates+=out['actual_updates'];assert out['actual_updates']==40
     if phase=='coarse'and key==p['baseline']and ds=='vidstg':
      oldfit=load(BASE/f'traces/{ds}/{a}.pt')['fit'];assert all(torch.equal(x['state'],y['state'])and torch.equal(x['boxes'],y['boxes'])for x,y in zip(out['path'][:31],oldfit['path']));del oldfit
     h=save(dest,dict(alias=a,config_id=key,fit=out,input_sha256=sha(inpfile),GT_read=False));files[key+'/'+a]=dict(path=str(dest.relative_to(ROOT)),sha256=h,historical_reuse=False);count+=1;print('TUNE_FIT',ds,phase,a,key,round(out['seconds'],3),flush=True);del out;gc.collect();torch.cuda.empty_cache()
    assert not rp.state.detach().any()
   finally:del rp,cache;gc.collect();torch.cuda.empty_cache()
   put(D/'STATE.json',dict(status='running',dataset=ds,phase=phase,last_alias=a,new_fits_this_invocation=count,new_updates_this_invocation=updates,pid=os.getpid(),new_expert_forwards=0))
  assert len(files)==len(cfgs)*128;m.cpu();assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());verify()
  put(D/ds/f'{phase.upper()}_FIT_BARRIER.json',dict(status='sealed',configs=cfgs,files=files,parents=128,dataset=ds,GT_read=False,epoch=time.time(),plan_sha256=sha(D/'PLAN_LOCK.json')));print('TUNE_PHASE_DONE',ds,phase,count,updates,flush=True)
 except BaseException as e:failure=repr(e);put(D/f'FAILURE_{ds}_{phase}_{time.time_ns()}.json',dict(error=failure,new_fits=count,updates=updates));raise
 finally:
  led=read(D/'GPU_LEDGER.json')if(D/'GPU_LEDGER.json').exists()else dict(invocations=[]);led['invocations'].append(dict(dataset=ds,phase=phase,wall_seconds=time.perf_counter()-start,new_primary_fits=count,new_updates=updates,compatibility_updates=compat,Source_load_seconds=locals().get('loading'),peak_GPU_bytes=torch.cuda.max_memory_allocated(),new_RGB_or_Source_encoding_forwards=0,new_expert_forwards=0,failure=failure));put(D/'GPU_LEDGER.json',led);lock.close()
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
