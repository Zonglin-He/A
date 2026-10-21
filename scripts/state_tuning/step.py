"""HC64 real adaptation with a locked, label-free calibrated full-state policy."""
import os,sys,json,hashlib,time,gc,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import prepare,read,sha,atomic,savept,OLD
from scripts.state_tuning.step_core import fit
TR=ROOT/'outputs/state_step_hc64';HERE=Path(__file__).resolve().parent
ARMS=['StateOPD','Direct','DirectMC','DirectFullRef']

def main():
 _,records,items,oldlock=prepare();records=records[:64];items=items[:64]
 cal=read(TR/'calibration.json');assert cal['GT_read'] is False and cal['queries']==64
 cfg=cal['selected'];assert cfg['basis_dimensions']==256 and cfg['radius'] is None
 assert cfg['lr']==.04 and cfg['forward_budget']==1321 and cfg['seed']==41
 config=dict(status='locked_before_new_adaptation',queries=64,arms=ARMS,selected=cfg,
             reward='same native-mask xy-projection Dice and identical mean/max whole-trajectory protection',
             calibration_sha256=sha(TR/'calibration.json'),capacity_seal_sha256=cal['capacity_seal_sha256'],
             parent_fingerprint=oldlock['fingerprint'],formal_state_sha256=oldlock['formal_state_sha256'],
             sources={str(p.relative_to(ROOT)):sha(p)for p in [HERE/'step_core.py',HERE/'step.py']},
             ids=[r['id']for r in records],GT_read=False)
 fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['fingerprint']=fp
 if (TR/'lock.json').exists():assert read(TR/'lock.json')==config
 else:atomic(TR/'lock.json',config)
 assert not (TR/'sealed.json').exists(),'Already sealed; refuse overwrite'
 from stvg.runtime import setup,lease,load_source
 setup();leasefile=lease();model=None;files={};begin=time.perf_counter()
 atomic(TR/'run.json',dict(status='running',PID=os.getpid(),fingerprint=fp,GT_read=False))
 try:
  sc=read(OLD/'config.json')['datasets']['hc2']
  for rec,item in zip(records,items):
   key=rec['id'];dest=TR/'predictions'/f'{key}.npz';receipt=dest.with_suffix('.json')
   if receipt.exists():
    rr=read(receipt);assert rr['fingerprint']==fp and sha(dest)==rr['sha256'] and sha(TR/rr['trace_path'])==rr['trace_sha256'];files[key]=rr;continue
   start=time.perf_counter();cache=torch.load(OLD/'student_cache'/f'{key}.pt',map_location='cpu',weights_only=False)
   assert sha(OLD/'student_cache'/f'{key}.pt')==item['cache_sha256']
   pack=torch.load(OLD/'research/native_masks'/item['mask_path'],map_location='cpu',weights_only=False)
   assert sha(OLD/'research/native_masks'/item['mask_path'])==item['native_mask_sha256']
   assert sha(OLD/item['reference_path'])==item['reference_sha256']
   assert sha(ROOT/'outputs/formal/state.json')==oldlock['formal_state_sha256']
   with np.load(OLD/item['reference_path'])as f:gold={k:f[k].copy()for k in f.files}
   if model is None:
    model=load_source(sc);model.cpu();gc.collect();torch.cuda.empty_cache();model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
   result={k:gold[k]for k in ['frame_ids','positions8','Strong_interval']};traces={};costs={}
   for arm in ARMS:
    out=fit(model,cache,pack,item['key'],arm,cfg)
    result[arm]=out['final'].numpy();costs[arm]=out['costs'];traces[arm]=out
    atomic(TR/'progress.json',dict(completed=len(files),total=64,id=key,arm=arm,GT_read=False))
    print(json.dumps(dict(completed=len(files),arm=arm,cost=out['costs'])),flush=True)
   tp=TR/'traces'/f'{key}.pt';savept(tp,traces);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.npz.tmp')
   with tmp.open('wb')as h:np.savez(h,**result)
   os.replace(tmp,dest)
   rr=dict(fingerprint=fp,sha256=sha(dest),trace_path=str(tp.relative_to(TR)),trace_sha256=sha(tp),reference_sha256=item['reference_sha256'],
           cache_sha256=item['cache_sha256'],native_mask_sha256=item['native_mask_sha256'],costs=costs,total_seconds=time.perf_counter()-start,GT_read=False)
   atomic(receipt,rr);files[key]=rr
   atomic(TR/'progress.json',dict(completed=len(files),total=64,GT_read=False))
   print(json.dumps(dict(completed=len(files),seconds=rr['total_seconds'])),flush=True)
   del traces,out,result,cache,pack;gc.collect();torch.cuda.empty_cache()
  assert len(files)==64 and sha(ROOT/'outputs/formal/state.json')==oldlock['formal_state_sha256']
  assert all(sha(ROOT/p)==h for p,h in config['sources'].items())
  atomic(TR/'sealed.json',dict(fingerprint=fp,queries=64,files={k:dict(path='predictions/'+k+'.npz',sha256=v['sha256'],receipt_sha256=sha(TR/'predictions'/f'{k}.json'))for k,v in files.items()},lock_sha256=sha(TR/'lock.json'),GT_read=False))
  atomic(TR/'run.json',dict(status='completed_sealed',queries=64,seconds=time.perf_counter()-begin,GT_read=False))
 except BaseException:
  atomic(TR/'failure.json',dict(completed=len(files),traceback=traceback.format_exc(),GT_read=False));raise
 finally:
  del model;gc.collect();torch.cuda.empty_cache();leasefile.close()

if __name__=='__main__':main()
