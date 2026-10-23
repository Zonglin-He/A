import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json,collections
from pathlib import Path
sys.path[:0]=['.','tools']
import torch,numpy as np
from full import read,sha,atomic,check,ROOT,RUN
from stvg.evidence import SourceEvidence,extract_visual_grids
from stvg.spatial import SpatialGeometry,evidence_dp
import stvg.evidence as em
import stvg.spatial as sm
from stvg.prototype import one_step
from stvg.prior import from_capture
from stvg.temporal import final_mbr
from stvg.representation import extract
D=ROOT/'outputs/algorithm_simplification_v1';p=read(ROOT/'outputs/cycle_reliability_ablation_v1/plan.json');torch.set_num_threads(1);check();rows=[]
for n,item in enumerate(p['items']):
 cap=torch.load(item['capture_path'],map_location='cpu',weights_only=False);raw=torch.load(item['raw_path'],map_location='cpu',weights_only=False);pack=torch.load(item['mask_path'],map_location='cpu',weights_only=False);j=int(pack['selected_query']);masks=pack['raw_out']['pred_masks'][0,:,j].float();ids=cap['frame_ids'];pos=cap['positions8'];count=collections.Counter();dur=collections.defaultdict(float)
 def timer(stage,fn):
  def f(*a,**kw):
   s=time.perf_counter();out=fn(*a,**kw);dur[stage]+=time.perf_counter()-s;count[stage]+=1;return out
  return f
 original=(em.affinity,sm.affinity,sm.fit_axis_affine,sm.soft_dice)
 # Warm up on the first item only. It is excluded from all measured rows.
 if n==0:
  SourceEvidence.from_capture(raw['spatial_inputs'],ids,pos,masks,pack['input_shape'][-2:],selected_query=j)
 ev=SourceEvidence.from_capture(raw['spatial_inputs'],ids,pos,masks,pack['input_shape'][-2:],selected_query=j)
 q0=from_capture(cap)
 with np.load(item['prediction_path'],allow_pickle=False)as a:span=a['Full_interval'].copy();ref=a['Full'].copy()
 for rep in range(3):
  em.affinity=timer('adjacent_affinity',original[0]);sm.affinity=timer('anchor_affinity',original[1]);sm.fit_axis_affine=timer('affine_fit',original[2]);sm.soft_dice=timer('cycle_and_link_dice',original[3])
  try:
   s=time.perf_counter();features,valid,physical,shape,meta=extract_visual_grids(raw['spatial_inputs'],ids);dur['grid_extraction_validation']+=time.perf_counter()-s
   s=time.perf_counter();ee=SourceEvidence.from_native_masks(features,valid,ids,pos,masks,pack['input_shape'][-2:],shape,selected_query=j);dur['masks_and_adjacency']+=time.perf_counter()-s
   s=time.perf_counter();g=SpatialGeometry(ee,cap['E'],pack['raw_out']['pred_boxes'][0,:,j]);dur['geometry_total']+=time.perf_counter()-s
   s=time.perf_counter();aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]));dur['aux_DP']+=time.perf_counter()-s
   s=time.perf_counter();q,_=one_step(cap['content_features'],q0);dur['prototype']+=time.perf_counter()-s
   s=time.perf_counter();iv=final_mbr(ids,q)['interval'];dur['MBR']+=time.perf_counter()-s;assert iv==span.tolist()
   s=time.perf_counter();out=evidence_dp(g.for_interval(span));dur['final_DP']+=time.perf_counter()-s;assert np.array_equal(out['final'].numpy(),ref)
  finally:em.affinity,sm.affinity,sm.fit_axis_affine,sm.soft_dice=original
 r=dict(query=f'H{n+1:03d}',frames=len(ids),grid_cells=ev.height*ev.width,times_seconds={k:v/3 for k,v in dur.items()},calls_per_query={k:v/3 for k,v in count.items()});rows.append(r)
 if(n+1)%16==0:print('profiled',n+1,flush=True)
fields=rows[0]['times_seconds'];summary={k:dict(mean_seconds=float(np.mean([x['times_seconds'][k]for x in rows])),median_seconds=float(np.median([x['times_seconds'][k]for x in rows])))for k in fields}
assert not torch.cuda.is_initialized();atomic(D/'profile_current.json',dict(status='complete',queries=len(rows),repeats=3,one_warmup_excluded=True,source='cached original Source/masks; no network/RGB/teacher',times=summary,rows=rows,nested=['adjacent_affinity is INSIDE masks_and_adjacency','anchor_affinity/affine_fit/cycle_and_link_dice are INSIDE geometry_total'],production_unchanged=True,GT_read=False,new_models=0))
print(json.dumps(summary),flush=True)
