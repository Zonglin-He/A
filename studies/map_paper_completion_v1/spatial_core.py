"""Current-MAP structural controls, raw-grid repooling and graph invariants."""
from common import *
import numpy as np,torch,inspect
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.evidence import SourceEvidence
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.readout import prediction
from map_core import map_interval
from supports import greedy
ARMS=['Source','SourceJointMAP','FullMAP','NoSharedCorrection','PastOnly','OneWayAffinity','AuxIndependent','AuxSourceROI','AuxE_ROI','NoSpatialTrajectory','NoTemporalInference','S0T0','S1T0','S0T1','S1T1']
def oneway_class():
 import stvg.spatial as original,textwrap
 code=textwrap.dedent(inspect.getsource(SpatialGeometry.__init__))
 old='fit_axis_affine(at*ta.T*seeds[j].flatten()[:,None],xy,xy)';new='fit_axis_affine(at*seeds[j].flatten()[:,None],xy,xy)';assert code.count(old)==1
 scope={**vars(original)};exec(compile(code.replace(old,new),'<pinned_original_spatial_constructor_oneway_affinity>','exec'),scope)
 return type('OneWayGeometry',(SpatialGeometry,),{'__init__':scope['__init__']})
def solve(g,iv):return evidence_dp(g.for_interval(iv))['final'].numpy()
def raw_evidence(c,raw):
 g=SpatialGeometry.from_cached_geometry(c['geometry']);seeds=g.side_maps[c['positions8'],0].clone()
 return SourceEvidence.from_arrays(raw['spatial_features'],raw['spatial_valid'],c['frame_ids'],c['positions8'],seeds)
def features(raw,boxes):return extract(dict(frame_ids=raw['frame_ids'],native_local_grid=raw['native_local_grid']),boxes)[0]
def derive(c,raw):
 ids=c['frame_ids'];pos=c['positions8'];q0=c['native_prior'];g=SpatialGeometry.from_cached_geometry(c['geometry']);span=[ids[0],ids[-1]+1];native=prediction(c['native_logits'],torch.as_tensor(c['Source']),c['records'],ids)['physical_interval']
 aux=solve(g,span);f=features(raw,aux);assert np.array_equal(f,c['content_features']),'Full raw-grid pool must exactly match parent'
 ev=raw_evidence(c,raw);gu=SpatialGeometry(ev,np.asarray(c['Source']),g.native_boxes.numpy());assert torch.equal(gu.E,torch.as_tensor(c['Source'],dtype=torch.float64))
 gp=SpatialGeometry.from_cached_geometry(c['geometry']);gp.side_available[:,1]=False;gp.source_anchors[:,1]=-1;gp._build_scores();assert not gp.side_available[:,1].any()
 go=oneway_class()(ev,np.asarray(c['E']),g.native_boxes.numpy())
 fs=dict(FullMAP=f,NoSharedCorrection=features(raw,solve(gu,span)),PastOnly=features(raw,solve(gp,span)),OneWayAffinity=features(raw,solve(go,span)),AuxIndependent=features(raw,greedy(g.for_interval(span))),AuxSourceROI=features(raw,np.asarray(c['Source'])),AuxE_ROI=features(raw,np.asarray(c['E'])))
 compact=dict(ids=ids,pos=pos,Source=np.asarray(c['Source']),E=np.asarray(c['E']),q0=q0,original_interval=native,geometries=dict(main=g.state_dict(),uncorrected=gu.state_dict(),past=gp.state_dict(),oneway=go.state_dict()),features=fs,GT_read=False)
 arrays=replay(compact);assert np.array_equal(arrays['NoSpatialTrajectory'],np.asarray(c['E']))and arrays['NoTemporalInference_interval'].tolist()==native
 return arrays,compact

def replay(c):
 ids=c['ids'];q0=c['q0'];gs={k:SpatialGeometry.from_cached_geometry(v)for k,v in c['geometries'].items()};g=gs['main'];iv0=map_interval(ids,q0);a=dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['original_interval']),SourceJointMAP=c['Source'].copy(),SourceJointMAP_interval=np.asarray(iv0),frame_ids=np.asarray(ids),positions8=np.asarray(c['pos']))
 for name,f in c['features'].items():
  q,_=one_step(f,q0);iv=map_interval(ids,q);geo=gs[dict(NoSharedCorrection='uncorrected',PastOnly='past',OneWayAffinity='oneway').get(name,'main')];a[name]=solve(geo,iv);a[name+'_interval']=np.asarray(iv)
 qe,_=one_step(c['features']['AuxE_ROI'],q0);ive=map_interval(ids,qe);a.update(NoSpatialTrajectory=c['E'].copy(),NoSpatialTrajectory_interval=np.asarray(ive),NoTemporalInference=solve(g,c['original_interval']),NoTemporalInference_interval=np.asarray(c['original_interval']),S0T0=c['E'].copy(),S0T0_interval=np.asarray(iv0),S1T0=solve(g,iv0),S1T0_interval=np.asarray(iv0),S0T1=c['E'].copy(),S0T1_interval=np.asarray(ive),S1T1=a['FullMAP'].copy(),S1T1_interval=a['FullMAP_interval'].copy())
 assert all(np.isfinite(a[k]).all()for k in a)
 return a
