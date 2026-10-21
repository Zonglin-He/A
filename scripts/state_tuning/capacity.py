"""HC64 saved-input geometry replay and state-capacity interventions; no GT."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,time,gc,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import prepare,read,sha,atomic,savept,OLD
from scripts.state_opd.core import content_basis,FrozenContentReplay,ProjectionEvidence,protected_utility,_brief
TR=ROOT/'outputs/state_capacity_hc64'
NAMES=['Full','FullR4','PCA4','PCA4R4']

def main():
 cfg,records,items,oldlock=prepare();records=records[:64];items=items[:64]
 TR.mkdir(parents=True,exist_ok=True)
 identity=dict(parent_fingerprint=oldlock['fingerprint'],source_sha256=sha(__file__),
               ids=[r['id']for r in records],interventions=NAMES,radius=4.,
               cap='closest-state radial projection to 4*(1-1e-6), inside old feasible open ball',
               GT_read=False,formal_state_sha256=oldlock['formal_state_sha256'])
 fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();identity['fingerprint']=fp
 if (TR/'lock.json').exists():assert read(TR/'lock.json')==identity
 else:atomic(TR/'lock.json',identity)
 assert not (TR/'sealed.json').exists(),'Already completed; do not overwrite'
 from stvg.runtime import setup,lease,load_source
 from stvg.state import Replay
 from stvg.policy import fit
 setup();leasefile=lease();model=None;files={};begin=time.perf_counter()
 atomic(TR/'run.json',dict(status='running',PID=os.getpid(),fingerprint=fp,GT_read=False))
 try:
  sc=read(OLD/'config.json')['datasets']['hc2']
  for rec,item in zip(records,items):
   key=rec['id'];dest=TR/'predictions'/f'{key}.npz';receipt=dest.with_suffix('.json')
   if receipt.exists():
    rr=read(receipt);assert rr['fingerprint']==fp and sha(dest)==rr['sha256']
    assert sha(TR/rr['trace_path'])==rr['trace_sha256'];files[key]=rr;continue
   start=time.perf_counter();cache=torch.load(OLD/'student_cache'/f'{key}.pt',map_location='cpu',weights_only=False)
   assert sha(OLD/'student_cache'/f'{key}.pt')==item['cache_sha256']
   with np.load(OLD/item['reference_path'])as f:gold={k:f[k].copy()for k in f.files}
   assert sha(OLD/item['reference_path'])==item['reference_sha256']
   if model is None:
    model=load_source(sc);model.cpu();gc.collect();torch.cuda.empty_cache()
    model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
   rp=Replay(model,cache,cache['Source'],cache['E'],cache['planned_positions'],lambda:None)
   geo=fit(rp,cache['teacher']['boxes'],item['key'],'Live',41,sc)
   assert np.array_equal(geo['final'].numpy(),gold['GeometryOPD']),'Original geometry replay mismatch'
   theta=geo['path'][-1]['state'].clone();states=torch.stack([p['state']for p in geo['path']])
   del geo,rp;gc.collect();torch.cuda.empty_cache()
   P,bi=content_basis(cache);project=(theta.double()@P@P.T).float()
   def cap(t):return t*min(1.,4.*(1.-1e-6)/float(t.norm())) if float(t.norm())else t.clone()
   variants=dict(Full=theta,FullR4=cap(theta),PCA4=project,PCA4R4=cap(project))
   replay=FrozenContentReplay(model,cache)
   pack=torch.load(OLD/'research/native_masks'/item['mask_path'],map_location='cpu',weights_only=False)
   assert sha(OLD/'research/native_masks'/item['mask_path'])==item['native_mask_sha256']
   ev=ProjectionEvidence(pack,cache,'cuda');result={k:gold[k]for k in ['frame_ids','positions8','Strong_interval']};utility={}
   with torch.no_grad():
    for name,state in variants.items():
     b=replay.boxes(state.cuda());result[name]=b.cpu().numpy();utility[name]=_brief(protected_utility(b,replay.E,ev))
   assert np.array_equal(result['Full'],gold['GeometryOPD'])
   result['ERefStrong']=gold['E_Ref'];result['GeometryOPD']=gold['GeometryOPD']
   trace=TR/'traces'/f'{key}.pt';savept(trace,dict(theta=theta,basis=P,variants=variants,state_path=states,utility=utility,basis_info=bi))
   dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.npz.tmp')
   with tmp.open('wb')as h:np.savez(h,**result)
   os.replace(tmp,dest)
   rr=dict(fingerprint=fp,sha256=sha(dest),trace_path=str(trace.relative_to(TR)),trace_sha256=sha(trace),
           reference_sha256=item['reference_sha256'],geometry_replay_bit_exact=True,
           theta_L2=float(theta.norm()),PCA4_L2=float(project.norm()),
           retained_state_energy=float(project.square().sum()/theta.square().sum()),
           seconds=time.perf_counter()-start,GT_read=False)
   atomic(receipt,rr);files[key]=rr
   atomic(TR/'progress.json',dict(completed=len(files),total=64,GT_read=False));print(json.dumps(dict(completed=len(files),norm=rr['theta_L2'],retained=rr['retained_state_energy'])),flush=True)
   del cache,pack,replay,ev,variants,theta,project,states;gc.collect();torch.cuda.empty_cache()
  assert len(files)==64 and sha(ROOT/'outputs/formal/state.json')==oldlock['formal_state_sha256']
  atomic(TR/'sealed.json',dict(fingerprint=fp,queries=64,files={k:dict(path='predictions/'+k+'.npz',sha256=v['sha256'],receipt_sha256=sha(TR/'predictions'/f'{k}.json'))for k,v in files.items()},lock_sha256=sha(TR/'lock.json'),GT_read=False))
  atomic(TR/'run.json',dict(status='completed_sealed',queries=64,seconds=time.perf_counter()-begin,GT_read=False))
 except BaseException:
  atomic(TR/'failure.json',dict(completed=len(files),traceback=traceback.format_exc(),GT_read=False));raise
 finally:
  del model;gc.collect();torch.cuda.empty_cache();leasefile.close()

if __name__=='__main__':main()
