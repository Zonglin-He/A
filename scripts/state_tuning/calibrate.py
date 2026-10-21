"""Label-free, global sampling/temperature calibration after the capacity replay."""
import os,sys,json,hashlib,time,gc
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import read,sha,atomic,savept,OLD
from scripts.state_opd.core import FrozenContentReplay,ProjectionEvidence,protected_utility
from scripts.state_tuning.core import candidate_noise
TR=ROOT/'outputs/state_tuning_hc64';CAP=ROOT/'outputs/state_capacity_hc64'
RULE=dict(ratios=[.5,1.,2.],scale_reference='median L2 of exact original geometric content / sqrt(2048)',selection='largest sigma with mean candidate guard-zero fraction >=0.8; else smallest',temperature='max(1e-6, median per-query std(candidate U)), two significant digits',seed=41,candidates=32,labels=False)
def sig(x):return float(f'{x:.2g}')
def main():
 assert (CAP/'sealed.json').exists();seal=read(CAP/'sealed.json');assert seal['queries']==64
 TR.mkdir(parents=True,exist_ok=True);atomic(TR/'calibration_rule.json',RULE)
 assert not (TR/'calibration.json').exists(),'Do not overwrite completed calibration'
 # No labels or scored results are opened.
 ids=list(seal['files']);norms=[read((CAP/seal['files'][k]['path']).with_suffix('.json'))['theta_L2']for k in ids]
 base=float(np.median(norms))/np.sqrt(2048);scales=[sig(base*r)for r in RULE['ratios']]
 atomic(TR/'calibration_plan.json',dict(rule=RULE,sigmas=scales,geometry_norm_median=float(np.median(norms)),capacity_seal_sha256=sha(CAP/'sealed.json'),GT_read=False))
 from stvg.runtime import setup,lease,load_source
 setup();leasefile=lease();model=None;allrows=[];sc=read(OLD/'config.json')['datasets']['hc2'];oldlock=read(ROOT/'outputs/state_opd_hc128/lock.json');items={x['id']:x for x in oldlock['inputs']}
 atomic(TR/'calibration_run.json',dict(status='running',PID=os.getpid(),GT_read=False))
 try:
  for key in ids:
   c=torch.load(OLD/'student_cache'/f'{key}.pt',map_location='cpu',weights_only=False);it=items[key]
   assert sha(OLD/'student_cache'/f'{key}.pt')==it['cache_sha256']
   pack=torch.load(OLD/'research/native_masks'/it['mask_path'],map_location='cpu',weights_only=False)
   assert sha(OLD/'research/native_masks'/it['mask_path'])==it['native_mask_sha256']
   if model is None:
    model=load_source(sc);model.cpu();gc.collect();torch.cuda.empty_cache();model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
   rp=FrozenContentReplay(model,c);ev=ProjectionEvidence(pack,c,'cuda');q=dict(seed=41,basis_dimensions=256)
   noise,seed=candidate_noise(it['key'],0,q);noise=noise.cuda();rr=[]
   with torch.no_grad():
    for sigma in scales:
     U=[];G=[]
     for eps in noise:
      b=rp.boxes(sigma*eps);u=protected_utility(b,rp.E,ev);U.append(float(u['U']));G.append(float(u['guard']))
     rr.append(dict(sigma=sigma,U=U,guard=G,std_U=float(np.std(U)),guard_zero_fraction=float(np.mean(np.asarray(G)==0))))
   allrows.append(dict(id=key,rows=rr,rng_seed=seed));atomic(TR/'calibration_progress.json',dict(completed=len(allrows),total=64,GT_read=False));print(json.dumps(dict(completed=len(allrows))),flush=True)
   del c,pack,rp,ev,noise;gc.collect();torch.cuda.empty_cache()
  summaries=[dict(sigma=s,median_std_U=float(np.median([r['rows'][i]['std_U']for r in allrows])),mean_guard_zero_fraction=float(np.mean([r['rows'][i]['guard_zero_fraction']for r in allrows])))for i,s in enumerate(scales)]
  eligible=[v for v in summaries if v['mean_guard_zero_fraction']>=.8];chosen=(eligible or summaries[:1])[-1];tau=sig(max(1e-6,chosen['median_std_U']))
  config=dict(seed=41,sigma=chosen['sigma'],tau=tau,basis_dimensions=256,radius=None,lr=.014,opd_rounds=40,direct_steps=660,forward_budget=1321,samples=32)
  savept(TR/'calibration_candidates.pt',allrows)
  atomic(TR/'calibration.json',dict(status='completed_no_GT_global_calibration',rule=RULE,summaries=summaries,selected=config,queries=64,candidate_trajectories=64*32*3,capacity_seal_sha256=sha(CAP/'sealed.json'),candidates_sha256=sha(TR/'calibration_candidates.pt'),GT_read=False))
  atomic(TR/'calibration_run.json',dict(status='completed',queries=64,GT_read=False));print(json.dumps(dict(selected=config,summaries=summaries)),flush=True)
 finally:
  del model;gc.collect();torch.cuda.empty_cache();leasefile.close()
if __name__=='__main__':main()
