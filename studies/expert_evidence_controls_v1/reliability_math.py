"""Unchanged Full math on correct-checkpoint saved Source and new raw expert8."""
from common import *
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.evidence import SourceEvidence,extract_visual_grids
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.prior import native_prior
from stvg.readout import prediction
def replay(v):
 q,_=one_step(v['features'],v['q0'],kappa=0,beta=1);iv=np.asarray(map_interval(v['frame_ids'],q));assert np.array_equal(q,v['posterior'])and np.array_equal(iv,v['Full_interval'])
 g=SpatialGeometry.from_cached_geometry(v['geometry']);out=dict(Source=v['Source'].copy(),Source_interval=v['Source_interval'].copy(),Full=evidence_dp(g.for_interval(iv))['final'].numpy(),Full_interval=iv,E=v['E'].copy(),frame_ids=np.asarray(v['frame_ids']),positions8=np.asarray(v['positions8']))
 assert all(np.isfinite(x).all()for x in out.values());return out
def build(native,teacher,target,positions,grid=None):
 ids=native['frame_ids'];pos=list(map(int,positions));cfg=fixed(R/'configs/default.json')['datasets'][target];src=np.asarray(native['Source']);boxes=np.asarray(teacher['boxes']);assert len(pos)==8 and boxes.shape==(8,4)and teacher['native_same_track_all8']and teacher['nested_mask_all_false']
 if grid is None:f,valid,ii,shape,_=extract_visual_grids(native['spatial_inputs'],ids);assert ii==ids
 else:f,valid,shape=grid
 shared,_=initialize(src,pos,boxes,cfg['E_calibration']);E=output_writeback(src,shared,cfg['initialization_gamma'])
 ev=SourceEvidence.from_native_masks(f,valid,ids,pos,teacher['native_mask_logits'],teacher['input_shape'][-2:],shape,selected_query=teacher['selected_query']);geometry=SpatialGeometry(ev,E,boxes)
 features,_=extract(native,E);q0=native_prior(native['records'],native['native_logits'],ids);q,_=one_step(features,q0,kappa=0,beta=1);iv=np.asarray(map_interval(ids,q));sourceiv=np.asarray(prediction(native['native_logits'],torch.as_tensor(src),native['records'],ids)['physical_interval'])
 v=dict(Source=src.copy(),Source_interval=sourceiv,E=E,frame_ids=ids,positions8=pos,features=features,q0=q0,posterior=q,Full_interval=iv,geometry=geometry.state_dict(),GT_read=False,first_aux_DP=False,final_DP=True)
 return replay(v),v
