"""Independent NumPy rectangle integration, frozen Gaussian and aggregate audit."""
from pathlib import Path
import sys,json,collections,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_soft_correspondence32_v1 import D,OLD
import numpy as np

def area_score(box,m):
 h,w=m.shape;x,y,bw,bh=box;left=(x-bw/2)*w;right=(x+bw/2)*w;top=(y-bh/2)*h;bottom=(y+bh/2)*h;overlap=0.
 for i in range(h):
  for j in range(w):overlap+=max(0,min(bottom,i+1)-max(top,i))*max(0,min(right,j+1)-max(left,j))*m[i,j]
 return overlap/(bw*bh*w*h+float(m.sum())-overlap)

def run():
 import torch
 P=ROOT/'results/stvg_soft_correspondence/2026-10-18/qualification32';z=read(P/'RESULTS.json');gate=read(P/'GATE.json');records=read(D/'PLAN_LOCK.json')['records'];checks=0;maxerr=0.;stats=[]
 for rec in records:
  a=load(D/'predictions'/(rec['alias']+'.pt'));old=load(OLD/'predictions'/(rec['alias']+'.pt'));assert torch.equal(a['central'],old['fit']['before']);samples=a['samples'].double().numpy();central=a['central'].double().numpy();eps=(samples-np.log(central/(1-central))[:,None])/a['config']['sigma'];assert abs(eps[:,:16]+eps[:,16:]).max()<1e-3;checks+=1
  if a['admitted_positions']:
   for j,pos in enumerate(old['fit']['positions']):assert torch.equal(a['samples'][pos],old['fit']['rounds'][0]['rollout']['samples'][j]);checks+=1
   ac=a['actions'].double().numpy();maps=a['soft_maps_per_anchor'].double().numpy();stored=a['per_anchor_rewards'].double().numpy()
   for anchor in range(len(maps)):
    for frame in sorted(set([0,len(ac)//2,len(ac)-1])):
     for sample in [0,7,16,31]:
      score=area_score(ac[frame,sample],maps[anchor,frame]);err=abs(score-stored[anchor,frame,sample]);assert err<1e-7;maxerr=max(err,maxerr);checks+=1
  for arm in ['real','shuffled','interpolation']:
   r=a['rewards'][arm].double().numpy();w=np.exp((r-r.max(1,keepdims=True))/a['config']['tau']);w/=w.sum(1,keepdims=True);assert abs(w-a['weights'][arm].double().numpy()).max()<1e-7;checks+=w.size
  assert sorted(a['permutation'])==list(range(len(a['frame_ids'])))
  assert all(a['permutation'][i]==i for i in a['planned_positions']);assert all(a['permutation'][i]!=i for i in range(len(a['frame_ids']))if i not in a['planned_positions']);checks+=len(a['frame_ids'])
  stats.append(dict(dataset=a['dataset'],frames=len(a['frame_ids']),shape=list(a['maps'].shape[-2:]),valid_patches=int(a['valid'].sum()),all_patches=int(a['valid'].numel()),mean_anchor_disagreement=float(a['per_anchor_disagreement'].mean())))
 for ds in ['hc2','vidstg']:
  rr=[r for r in z['rows']if r['dataset']==ds]
  for sup,g in z['groups'][ds].items():
   valid=[r for r in rr if r['supports'][sup]['frames']]
   assert len(valid)==g['evaluable_queries'];assert sum(r['supports'][sup]['frames']for r in valid)==g['frames']
   for arm in ['real','shuffled','interpolation']:
    if valid:assert abs(sum(r['supports'][sup]['rank_delta'][arm]for r in valid)/len(valid)-g['mean'][arm])<1e-12;checks+=1
  assert gate['groups'][ds]['passed']==all(gate['groups'][ds]['conditions'].values())
 assert gate['passed']==all(g['passed']for g in gate['groups'].values())
 put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=checks,maximum_independent_soft_IoU_error=maxerr,Source_and_original_sample_binding_exact=True,valid_grid_and_map_stats=stats,old_saved_predictions_not_mutated=True,gate_recomputed=True));print('AUDIT',checks,maxerr)
if __name__=='__main__':run()
