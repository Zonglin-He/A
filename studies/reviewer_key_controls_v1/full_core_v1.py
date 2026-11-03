"""Shared acquisition for full ROI controls and two predeclared uniform strengths."""
from common_v1 import *
import numpy as np,torch
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.representation import extract,pool,weights
REGIONS=['SourceROI','CorrectedROI','GlobalAppearance','BackgroundAppearance','ShuffledBoxes']
GAMMAS=[('UniformGamma05',.5),('UniformGamma1',1.)]
ARMS=['Source',*REGIONS,*[n for n,v in GAMMAS]]
PAIRS=[('CorrectedROI',a)for a in REGIONS if a!='CorrectedROI']+[(a,'SourceROI')for a in REGIONS if a not in ['SourceROI','CorrectedROI']]+[(a,'Source')for a,v in GAMMAS]+[(a,'Current')for a,v in GAMMAS]
F=D/'full';Q=P/'full'
def region_features(raw,E,src,ds,key):
 fe,_=extract(raw,E);fs,_=extract(raw,src);perm=sorted(range(len(E)),key=lambda i:(hashlib.sha256(f'20261103|{ds}|{key}|{i}'.encode()).digest(),i))
 if len(E)>1 and perm==list(range(len(E))):perm=perm[1:]+perm[:1]
 shifted=E[perm];assert sorted(map(tuple,shifted))==sorted(map(tuple,E));fsh,ss=extract(raw,shifted);fg=np.empty_like(fe);fb=np.empty_like(fe);empty=0
 for off,(grid,valid)in enumerate(raw['native_local_grid']):
  xx=grid.permute(0,2,3,1).numpy()
  for j in range(len(xx)):
   t=off+2*j;v=valid[j].numpy();w,_=weights(E[t],*v.shape,v);bg=np.maximum(v.astype(float)-w,0)
   if bg.sum()==0:bg=v.astype(float);empty+=1
   fg[t]=pool(xx[j],v);fb[t]=pool(xx[j],bg)
 return dict(SourceROI=fs,CorrectedROI=fe,GlobalAppearance=fg,BackgroundAppearance=fb,ShuffledBoxes=fsh),dict(permutation=perm,empty_background_frames=empty,shuffled_fallback_frames=sum(x['fallback']for x in ss))
def replay(c):
 a=dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['Source_interval']),frame_ids=np.asarray(c['ids']),positions8=np.asarray(c['pos']))
 for name,features in c['region_features'].items():
  q,d=one_step(features,c['q0'],kappa=0,beta=1);iv=np.asarray(map_interval(c['ids'],q));a[name]=c['E'].copy();a[name+'_interval']=iv
 for name,item in c['gammas'].items():
  q,d=one_step(item['features'],c['q0'],kappa=0,beta=1);iv=np.asarray(map_interval(c['ids'],q));g=SpatialGeometry.from_cached_geometry(item['geometry']);a[name]=evidence_dp(g.for_interval(iv))['final'].numpy();a[name+'_interval']=iv
 a['Current']=c['current'].copy();a['Current_interval']=np.asarray(c['current_interval']);return a
def build(ds,row,parent,raw):
 c,proof=source(ds,row);assert list(raw['frame_ids'])==c['ids'];features,diag=region_features(raw,c['E'],c['Source'],ds,row['id']);assert np.array_equal(features['CorrectedROI'],c['features']['AuxE_ROI'])and np.array_equal(features['SourceROI'],c['features']['AuxSourceROI'])
 # Reconstruct every changed E candidate, appearance pool and posterior; side
 # propagation uses original post-fusion raw grids and original eight seed masks.
 ev=selected.spatial().raw_evidence(parent,raw);cfg=read(R/'configs/default.json')['datasets'][ds];native=np.asarray(parent['geometry']['native_boxes']);shared,_=initialize(parent['Source'],parent['positions8'],native,cfg['E_calibration']);gammas={}
 for name,gamma in GAMMAS:
  E=output_writeback(parent['Source'],shared,gamma);g=SpatialGeometry(ev,E,native);f,_=extract(raw,E);gammas[name]=dict(gamma=gamma,E=E,features=f,geometry=g.state_dict())
 ce=fixed(selected.D/f'components_{ds}_sealed.json')['files'][row['id']];assert sha(selected.D/ce['path'])==ce['sha256']
 with np.load(selected.D/ce['path'])as z:current=z['FullMAP'].copy();iv=z['FullMAP_interval'].copy();srciv=z['Source_interval'].copy()
 compact=dict(Source=c['Source'],Source_interval=srciv,ids=c['ids'],pos=c['pos'],E=c['E'],q0=c['q0'],region_features=features,gammas=gammas,current=current,current_interval=iv,GT_read=False)
 a=replay(compact);default='UniformGamma05'if ds=='hc2'else'UniformGamma1';assert np.array_equal(a[default],current)and np.array_equal(a[default+'_interval'],iv)
 # The regional experiment holds the output path fixed, including its expert
 # boxes. It is not a no-expert deployment ranking; Source-only is separate A.
 proof.update(actual_pretext_ROI_pools=True,original_default_gamma_full_array_exact=True,original_Source_ROI_and_E_ROI_features_exact=True,new_teacher=0,all_boxes_multiset_exact=True)
 return a,compact,proof,diag

