"""Actual state-only OPD from the changed initialization, no GT access."""
import os,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_initialization_coupled_common_v1 import ROOT,D,P,MAIN_D,BASE,read,put,sha,load,save,initial,verify

def main(ds,phase):
 import numpy as np,torch
 from methods.stvg_single_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1.state import Replay
 from methods.stvg_referdino_opd_v1.policy import fit
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 setup();p=verify();master=read(MAIN_D/'PLAN_LOCK.json');cfgs=p['datasets'][ds]['coarse_configs']if phase=='coarse'else read(D/ds/'FINE_PLAN_LOCK.json')['configs'];lock=lease();beg=time.perf_counter();prior=sum(v['wall_seconds']for v in read(MAIN_D/'GPU_LEDGER.json')['invocations']);count=updates=0;failure=None;files={};m=None
 def check():resource_guard();assert prior+time.perf_counter()-beg<7200,'shared GPU budget'
 try:
  t=time.perf_counter();m=load_source(next(iter(cfgs.values())));loading=time.perf_counter()-t;mh=state_hash(m.state_dict());m.cpu();gc.collect();torch.cuda.empty_cache();m.ground_decoder.decoder.cuda()
  for rec in [r for r in p['records']if r['dataset']==ds]:
   a=rec['alias'];f=BASE/f'inputs/{ds}/{a}.pt';assert sha(f)==master['source_files'][str(f.relative_to(ROOT))];cache=load(f)
   for cid,cfg in cfgs.items():
    dest=D/ds/'fits'/phase/cid/f'{a}.pt'
    if dest.exists():files[cid+'/'+a]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest));continue
    E=initial(cache['Source'],cache['E'],cfg['initialization_gamma']);rp=Replay(m,cache,cache['Source'],E,cache['planned_positions'],check)
    try:out=fit(rp,cache['teacher_native']['boxes'],a,'Live',41,cfg)
    finally:del rp
    assert out['actual_updates']==40 and np.array_equal(out['path'][0]['boxes'].numpy(),E);updates+=40;count+=1;h=save(dest,dict(alias=a,config_id=cid,fit=out,source_input_sha256=sha(f),initialization_gamma=cfg['initialization_gamma'],GT_read=False));files[cid+'/'+a]=dict(path=str(dest.relative_to(ROOT)),sha256=h);print('INIT_COUPLED_FIT',ds,phase,a,cid,cfg['initialization_gamma'],round(out['seconds'],3),flush=True);del out;gc.collect();torch.cuda.empty_cache()
   del cache;put(MAIN_D/'STATE.json',dict(status='running',dataset=ds,phase='initialization_'+phase,last_alias=a,new_fits_this_invocation=count,new_updates_this_invocation=updates,pid=os.getpid()))
  m.cpu();assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());verify();assert len(files)==len(cfgs)*128;put(D/ds/f'{phase.upper()}_FIT_BARRIER.json',dict(status='sealed',files=files,configs=cfgs,dataset=ds,parents=128,GT_read=False,epoch=time.time()));print('INIT_COUPLED_PHASE_DONE',ds,phase,count,updates,flush=True)
 except BaseException as e:failure=repr(e);put(MAIN_D/f'FAILURE_initialization_{ds}_{phase}_{time.time_ns()}.json',dict(error=failure,new_fits=count,new_updates=updates));raise
 finally:
  led=read(MAIN_D/'GPU_LEDGER.json');led['invocations'].append(dict(dataset=ds,phase='initialization_'+phase,wall_seconds=time.perf_counter()-beg,new_primary_fits=count,new_updates=updates,compatibility_updates=0,Source_load_seconds=locals().get('loading'),peak_GPU_bytes=torch.cuda.max_memory_allocated(),new_RGB_or_Source_encoding_forwards=0,new_expert_forwards=0,failure=failure));put(MAIN_D/'GPU_LEDGER.json',led);lock.close()
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
