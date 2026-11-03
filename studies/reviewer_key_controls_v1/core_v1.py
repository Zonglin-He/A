from common_v1 import *
import numpy as np,torch
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.paper_panel import interpolation
from stvg.representation import extract,pool,weights

def derive(task,ds,row):
 c,proof=source(ds,row);ids=np.asarray(c['ids']);pos=np.asarray(c['pos']);q0=c['q0'];E=c['E'];src=c['Source']
 qe,de=one_step(c['features']['AuxE_ROI'],q0,kappa=0,beta=1);qs,ds0=one_step(c['features']['AuxSourceROI'],q0,kappa=0,beta=1)
 ie=np.asarray(map_interval(ids,qe));is0=np.asarray(map_interval(ids,qs));iv0=np.asarray(map_interval(ids,q0));g=SpatialGeometry.from_cached_geometry(c['geometries']['main'])
 ce=fixed(selected.D/f'components_{ds}_sealed.json')['files'][row['id']];assert sha(selected.D/ce['path'])==ce['sha256']and sha(selected.D/ce['receipt_path'])==ce['receipt_sha256']
 with np.load(selected.D/ce['path'])as old:
  assert np.array_equal(ids,old['frame_ids'])and np.array_equal(pos,old['positions8'])and np.array_equal(src,old['Source'])and np.array_equal(ie,old['FullMAP_interval'])
  current=old['FullMAP'].copy();sourceiv=old['Source_interval'].copy()
 arrays=dict(frame_ids=ids,positions8=pos,Source=src.copy(),Source_interval=sourceiv)
 if task=='temporal':
  native=np.asarray(c['geometries']['main']['native_boxes']);inter=interpolation(E,ids,pos,native)
  pe=fixed(selected.RUN/ds/'sealed.json')['files'][row['id']];assert sha(selected.RUN/ds/pe['path'])==pe['sha256']
  with np.load(selected.RUN/ds/pe['path'])as z:assert np.array_equal(inter,z['Interpolation'])
  full=evidence_dp(g.for_interval(ie))['final'].numpy();assert np.array_equal(full,current)
  for name,boxes,iv in [('SourceJointMAP',src,iv0),('SourceTR',src,is0),('ExpertInitTR',E,ie),('Ours',full,ie),('FixedE_SourceTR',E,is0),('Interpolation_SourceTR',inter,is0),('Interpolation_ExpertTR',inter,ie)]:arrays[name]=boxes.copy();arrays[name+'_interval']=iv.copy()
  arrays.update(evidence_SourceROI=ds0['L'],evidence_CorrectedROI=de['L'],q0=q0.copy(),prototype_separation=np.asarray([np.linalg.norm(d['prototypes']['event']-d['prototypes']['background'])for d in [ds0,de]]))
  proof.update(parent_interpolation_prediction_sha256=pe['sha256'],source_only_temporal_inputs=['SourceROI features','q0','frame ids'],SourceTR_expert_required=False,competing_interpolation_DP=False)
 else:
  reuse=fixed(selected.PREV/'raw_reuse_index.json')[ds][row['id']];assert sha(reuse['path'])==reuse['sha256'];raw=torch.load(reuse['path'],map_location='cpu',weights_only=False,mmap=True)
  assert raw['frame_ids']==list(map(int,ids))and raw['parent_capture_sha256']==c['parent_capture_sha256']and not raw['GT_read']
  fe,_=extract(raw,E);fs,_=extract(raw,src);assert np.array_equal(fe,c['features']['AuxE_ROI'])and np.array_equal(fs,c['features']['AuxSourceROI'])
  perm=sorted(range(len(E)),key=lambda i:(hashlib.sha256(f'20261103|{ds}|{row["id"]}|{i}'.encode()).digest(),i))
  if len(E)>1 and perm==list(range(len(E))):perm=perm[1:]+perm[:1]
  shuffled=E[perm];assert sorted(map(tuple,shuffled))==sorted(map(tuple,E));fsh,shstats=extract(raw,shuffled)
  fg=np.zeros_like(fe);fb=np.zeros_like(fe);empty=0
  for off,(grid,valid)in enumerate(raw['native_local_grid']):
   x=grid.permute(0,2,3,1).numpy()
   for j in range(len(x)):
    t=off+2*j;v=valid[j].numpy();w,_=weights(E[t],*v.shape,v);bg=v.astype(float)-w;assert np.all(bg>=-1e-15);bg=np.maximum(bg,0)
    if bg.sum()==0:bg=v.astype(float);empty+=1
    fg[t]=pool(x[j],v);fb[t]=pool(x[j],bg)
  features=dict(SourceROI=fs,CorrectedROI=fe,GlobalAppearance=fg,BackgroundAppearance=fb,ShuffledBoxes=fsh)
  for name,f in features.items():
   q,d=one_step(f,q0,kappa=0,beta=1);iv=np.asarray(map_interval(ids,q));arrays[name]=E.copy();arrays[name+'_interval']=iv;arrays[name+'_evidence']=d['L']
  assert np.array_equal(arrays['CorrectedROI_interval'],ie)
  arrays['permutation']=np.asarray(perm);proof.update(retained_raw_sha256=reuse['sha256'],actual_ROI_repool=True,all_output_spatial_paths_same_E=True,background_empty_support_frames=empty,shuffle_global_fallback_frames=sum(x['fallback']for x in shstats),whole_box_multiset_exact=True)
 proof.update(current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],GT_read=False)
 assert all(np.isfinite(v).all()for v in arrays.values())
 return arrays,proof

