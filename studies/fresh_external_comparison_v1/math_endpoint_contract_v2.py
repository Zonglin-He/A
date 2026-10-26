"""One expert payload, simple native-time fusion and current own-time inference."""
from eg_common_v1 import *
import numpy as np,torch
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.evidence import SourceEvidence
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval
from stvg.readout import prediction
from stvg.paper_panel import interpolation

def replay(compact):
 ids=compact['frame_ids'];pos=compact['positions8'];native=compact['native_boxes'];E=compact['E'];q,_=one_step(compact['features'],compact['q0'],kappa=0,beta=1);iv=map_interval(ids,q)
 assert np.array_equal(q,compact['posterior'])and np.array_equal(iv,compact['interval'])
 geometry=SpatialGeometry.from_cached_geometry(compact['geometry'])
 out=dict(Source=compact['Source'].copy(),Source_interval=np.asarray(compact['native_interval']),Simple=interpolation(E,ids,pos,native),Simple_interval=np.asarray(compact['native_interval']),Ours=evidence_dp(geometry.for_interval(iv))['final'].numpy(),Ours_interval=np.asarray(iv),E=E.copy(),frame_ids=np.asarray(ids),positions8=np.asarray(pos))
 assert all(np.isfinite(v).all()for v in out.values());expected=np.asarray(native,dtype=np.float64).copy();expected[:,2:]=np.exp(np.log(expected[:,2:]));assert np.array_equal(out['Simple'][pos],expected)
 return out

def build(cap,raw,teacher,ds):
 ids=cap['frame_ids'];pos=list(cap['positions8']);assert raw['frame_ids']==ids and not raw['GT_read'];assert raw['parent_capture_sha256']
 assert teacher['nested_mask_all_false']and teacher['native_same_track_all8'];assert list(teacher['input_shape'][:2])==[8,3]
 native=torch.as_tensor(teacher['boxes']).detach().cpu().numpy();assert native.shape==(8,4)and np.isfinite(native).all()and(native[:,2:]>0).all()
 cfg=fixed(R/'configs/default.json')['datasets'][ds];shared,_=initialize(cap['Source'],pos,native,cfg['E_calibration']);E=output_writeback(cap['Source'],shared,cfg['initialization_gamma'])
 shape=(cap['geometry']['height'],cap['geometry']['width']);ev=SourceEvidence.from_native_masks(raw['spatial_features'],raw['spatial_valid'],ids,pos,teacher['native_mask_logits'],teacher['input_shape'][-2:],shape,selected_query=teacher['selected_query'])
 geometry=SpatialGeometry(ev,E,native);features,_=extract(raw,E);q,_=one_step(features,cap['native_prior'],kappa=0,beta=1);iv=map_interval(ids,q)
 native_iv=prediction(cap['native_logits'],torch.as_tensor(cap['Source']),cap['records'],ids)['physical_interval']
 compact=dict(schema='one_referring_video_expert_current_structured_inference_v1',frame_ids=ids,positions8=pos,Source=cap['Source'].copy(),native_boxes=native.copy(),native_interval=native_iv,q0=cap['native_prior'].copy(),E=E,geometry=geometry.state_dict(),features=features,posterior=q,interval=np.asarray(iv),GT_read=False,expert_model=teacher.get('expert_model','original_ReferDINO'),first_aux_DP=False,final_DP=True)
 out=replay(compact);again=replay(compact);assert all(np.array_equal(out[k],again[k])for k in out)
 return out,compact
