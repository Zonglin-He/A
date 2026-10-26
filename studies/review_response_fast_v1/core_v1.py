from common import *
import numpy as np,torch
from gamma_cached_core_v1 import replay
from stvg.representation import extract,pool
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.spatial import SpatialGeometry,evidence_dp

def derive(ds,row):
 key=row['id'];entry=read(G/ds/'sealed_retained_subset.json')['files'][key]
 for a,b in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(G/entry[a])==entry[b]
 c=torch.load(G/entry['compact_path'],map_location='cpu',weights_only=False,mmap=True);ga=replay(c)
 with np.load(G/entry['path'])as z:assert set(ga)==set(z.files)and all(np.array_equal(v,z[k])for k,v in ga.items())
 cap,origin=selected.suite().load_parent(ds,row);reuse=read(selected.PREV/'raw_reuse_index.json')[ds][key];assert sha(reuse['path'])==reuse['sha256']==entry['retained_raw_sha256']
 raw=torch.load(reuse['path'],map_location='cpu',weights_only=False,mmap=True);assert raw['parent_capture_sha256']==origin['capture_sha256']and raw['frame_ids']==cap['frame_ids']and not raw['GT_read']
 ce=read(selected.D/f'components_{ds}_sealed.json')['files'][key]
 for a,b in [('path','sha256'),('receipt_path','receipt_sha256')]:assert sha(selected.D/ce[a])==ce[b]
 with np.load(selected.D/ce['path'])as z:source=z['Source'].copy();iv=z['Source_interval'].copy();current=z['FullMAP'].copy();civ=z['FullMAP_interval'].copy()
 ga['Source']=source;ga['Source_interval']=iv
 default=entry['default_gamma_arm'];dc=c['arms'][default];assert np.array_equal(ga[default],current)and np.array_equal(ga[default+'_interval'],civ)
 E=np.asarray(dc['E']);features={};stats={}
 for name,boxes in [('CorrectedROI',E),('SourceROI',source)]:features[name],stats[name]=extract(raw,boxes)
 assert np.array_equal(features['CorrectedROI'],dc['features'])
 seed=hashlib.sha256((ds+':'+key+':roi-negative-v1').encode()).digest();shift=E.copy();shift[:,0]+= .25 if seed[0]%2 else -.25
 random=E.copy();random[:,0]=.1+.8*int.from_bytes(seed[1:9],'big')/(2**64-1);random[:,1]=.1+.8*int.from_bytes(seed[9:17],'big')/(2**64-1)
 for name,boxes in [('ShiftedROI',shift),('RandomROI',random)]:features[name],stats[name]=extract(raw,boxes)
 f=np.zeros_like(features['CorrectedROI']);seen=set()
 for off,(v,valid)in enumerate(raw['native_local_grid']):
  x=v.permute(0,2,3,1).numpy()
  for j in range(len(x)):
   t=off+2*j;f[t]=pool(x[j],valid[j].numpy());seen.add(t)
 assert seen==set(range(len(E)));features['GlobalAppearance']=f
 a=dict(Source=source,Source_interval=iv,frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']));g=SpatialGeometry.from_cached_geometry(dc['geometry']);posts={}
 for name in ROI[1:]:
  q,_=one_step(features[name],c['q0'],kappa=0,beta=1);interval=map_interval(c['frame_ids'],q);a[name]=evidence_dp(g.for_interval(interval))['final'].numpy();a[name+'_interval']=np.asarray(interval);posts[name]=q
 assert np.array_equal(a['CorrectedROI'],current)and np.array_equal(a['CorrectedROI_interval'],civ)
 proof=dict(parent=origin,original_gamma_prediction_sha256=entry['sha256'],original_gamma_compact_sha256=entry['compact_sha256'],original_gamma_receipt_sha256=entry['receipt_sha256'],raw_SHA256=reuse['sha256'],current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],same_main_geometry_sha256=digest(dc['geometry'])if isinstance(dc['geometry'],str)else sha(G/entry['compact_path']),actual_all_counterfactual_ROI_pool=True,own_posterior_interval_finalDP=True,new_Source=0,new_expert=0,GT_read=False)
 diag=dict(shift_x=.25 if seed[0]%2 else -.25,random_center=random[0,:2].tolist(),fallback_frames={n:sum(x['fallback']for x in ss)for n,ss in stats.items()},features_different_from_corrected={n:bool(not np.array_equal(v,features['CorrectedROI']))for n,v in features.items()},posterior_different_from_corrected={n:bool(not np.array_equal(v,posts['CorrectedROI']))for n,v in posts.items()})
 return {'gamma':ga,'roi':a},proof,diag
