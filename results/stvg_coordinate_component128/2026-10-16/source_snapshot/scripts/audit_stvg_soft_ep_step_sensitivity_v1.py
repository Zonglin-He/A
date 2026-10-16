"""Saved-fit process diagnostics; never opens labels or reruns a model."""
from pathlib import Path
import sys,statistics,math,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
def main(trial):
 import torch
 b=BASE/'trials'/trial;rt=read(b/'RUNTIME.json');root=read(b/'ROOT_READBACK.json');bar=read(b/'STUDENT_GLOBAL_BARRIER.json');assert root['status']=='pass'and root['count']==bar['count']==128
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 design=read(b/'DESIGN_LOCK.json');groups={};rows=[]
 for ds in ['hc2','vidstg']:
  for m in ['M0','M1','M2']:groups[ds,m]=dict(dataset=ds,mode=m,configured_lr=design['configs'][ds][m]['lr'],fit_count=0,new_fit_count=0,reused_fit_count=0,no_reference_count=0,proposals=0,accepted=0,skipped=0,actual_norms=[],proposal_norms=[],gradient_norms=[],accepted_scales=[],U_RMS=[],reward_stds=[])
 def norm(x):return float(x.detach().double().norm())
 for x in bar['files']:
  p=ROOT/x['path'];assert sha(p)==x['sha256'];d=torch.load(p,map_location='cpu',weights_only=False)
  for mode in ['M0','M1','M2']:
   g=groups[d['dataset'],mode];f=d['fits'][mode]
   if f is None:g['no_reference_count']+=1;continue
   g['fit_count']+=1;g['reused_fit_count'if'reused_fit_from_sha256'in f else'new_fit_count']+=1
   for z in f['path']:
    if z.get('skipped'):g['skipped']+=1;continue
    g['reward_stds'].append(float(z['rollout']['rewards'].detach().double().std(unbiased=False)))
    proposed=z.get('proposed',False)if mode=='M0'else True
    if not proposed:continue
    g['proposals']+=1;ok=bool(z['accepted']);g['accepted']+=ok
    before=z['state']if mode=='M0'else z['state_before'];after=z['update_state']if mode=='M0'else z['state_after'];delta=torch.cat([(after[n].detach().double()-v.detach().double()).reshape(-1)for n,v in before.items()]);actual=norm(delta)
    if not ok:assert actual==0
    raw=z['update']['raw']if mode=='M0'else z['original_adam_delta'];gradient=z['update']['gradient']if mode=='M0'else z['gradient'];g['proposal_norms'].append(norm(raw));g['gradient_norms'].append(norm(gradient))
    if ok:
     g['actual_norms'].append(actual);g['accepted_scales'].append(1. if mode=='M0'else z['accepted_scale'])
     if mode!='M0':g['U_RMS'].append(z['candidates'][-1]['U_RMS'])
 for g in groups.values():
  for key in ['actual_norms','proposal_norms','gradient_norms','accepted_scales','U_RMS','reward_stds']:
   vals=g.pop(key);assert all(math.isfinite(v)for v in vals);g[key]=dict(count=len(vals),mean=statistics.mean(vals)if vals else None,min=min(vals)if vals else None,max=max(vals)if vals else None)
  g['rejected']=g['proposals']-g['accepted'];g['acceptance_rate']=g['accepted']/g['proposals']if g['proposals']else None;rows.append(g)
 write(b/'STEP_SENSITIVITY_READBACK.json',dict(status='pass',scope='saved actual parameter displacements/proposals/backtracking/rollback, no GT or independent backward',count=128,rows=rows,source_barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),root_sha256=sha(b/'ROOT_READBACK.json'),CPU_only=True,new_GT_read=False,time=time.time()))
 print(trial,'step sensitivity readback pass')
if __name__=='__main__':main(sys.argv[1])
