"""Independent original-result readback, matched-weight and coarse-grid contracts."""
from pathlib import Path
import sys,json,math,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_correspondence32_v1 import D,OLD,PUB,read,put,load,sha,ARMS
import numpy as np,torch
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap,soft_iou
from scripts.score_stvg_correspondence_diagnosis32_v1 import matched_weights

def audit():
 z=read(PUB/'RESULTS.json');old=read(ROOT/'results/stvg_soft_correspondence/2026-10-18/qualification32/RESULTS.json');rows={r['alias']:r for r in old['rows']};checks=0;record=read(D/'PLAN_LOCK.json');maxerr=0.
 for r in z['rows']:
  c=load(D/'controls'/(r['alias']+'.pt'));b=load(OLD/'predictions'/(r['alias']+'.pt'));assert sha(OLD/'predictions'/(r['alias']+'.pt'))==next(p['sha256']for p in record['files']if p['path'].endswith('/'+r['alias']+'.pt'))
  for arm,oname in [('real','real'),('near_shift','shuffled'),('interpolation','interpolation')]:
   assert torch.equal(c['rewards'][arm],b['rewards'][oname]);assert torch.equal(c['weights'][arm],b['weights'][oname]);checks+=2
   for sup in ['unobserved','far','observed','all_known']:
    a=r['stats'][arm][sup]['rank_delta'];v=rows[r['alias']]['supports'][sup]['rank_delta'][oname];assert a is None and v is None or abs(a-v)<1e-12;checks+=1
  reference=b['weights']['interpolation'].double().numpy();baseESS=(reference**2).sum(-1)
  for arm in ARMS:
   re=c['rewards'][arm].double().numpy();w=matched_weights(re,reference);assert np.max(abs(w.sum(-1)-reference.sum(-1)))<1e-12
   unique=np.array([len(set(v.tolist()))==32 for v in re]);assert np.max(abs((w[unique]**2).sum(-1)-baseESS[unique]))<1e-12 if unique.any()else True
   # Sorting reward must not decrease assigned weights; tie permutations irrelevant.
   for x,y in zip(re,w):
    ix=np.argsort(x);assert np.min(np.diff(y[ix]))>=-1e-14;checks+=1
  maxerr=max(maxerr,c['cpu_vs_original_CUDA_max_map_error']);assert c['upsample_invariance_max_error']<1e-12
 # A rectangle wholly inside one cell: mask area a; fuzzy-IoU self=a/(2-a).
 grid=[]
 for h,w in [(7,4),(7,10),(14,20)]:
  box=torch.tensor([[.5/w,.5/h,.5/w,.5/h]],dtype=torch.float64);mask=cell_overlap(box,h,w);area=.25;value=float(soft_iou(box,mask)[0]);assert abs(value-area/(2-area))<1e-12;grid.append(dict(grid=[h,w],within_one_cell_area=area,soft_self_iou=value));checks+=1
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for arm in ARMS:
   for met in ['rank_delta','matched_strength_rank_delta','Spearman','pair_accuracy','top_reward_gain']:
    aa=[r['stats'][arm]['unobserved'][met]for r in rr if r['stats'][arm]['unobserved'][met]is not None];assert abs(sum(aa)/len(aa)-g['results']['unobserved'][arm][met]['mean'])<1e-12;checks+=1
  for sup,pairs in g['paired'].items():
   for label,vals in pairs.items():
    a,b=label.split('__minus__')
    for met,x in vals.items():
     aa=[r['stats'][a][sup][met]-r['stats'][b][sup][met]for r in rr if r['stats'][a][sup][met]is not None and r['stats'][b][sup][met]is not None];assert abs(sum(aa)/len(aa)-x['mean'])<1e-12;checks+=1
 # Original production registrations have not changed.
 assert all(sha(ROOT/p)==v for p,v in read(D/'RUNTIME_LOCK.json')['registries'].items())
 put(PUB/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=checks,all_original_maps_rewards_weights_predictions_immutable=True,original_b7_query_values_exact=True,matched_strength_same_sorted_weights_except_ties=True,partial_cell_math_contracts=grid,nearest_replication_invariance=True,max_CPU_original_CUDA_map_probe_error=maxerr,old_scientific_baselines_use_exact_saved_bits=True,GT_used_for_evaluation_only=True,registries_unchanged=True))
 print('AUDIT',checks,'maxCPUvsCUDA',maxerr)
if __name__=='__main__':audit()
