"""One-factor posterior controls, each with its own interval and same final DP."""
from common import *
import numpy as np,torch
from stvg.prototype import one_step,evidence,membership,inclusive_scores,update
from stvg.current_shared_roi import map_interval
from stvg.spatial import SpatialGeometry,evidence_dp
def smooth(q):
 # K*P0*K^T: fixed [1,2,1]/4 independently along endpoint indices,
 # zero padding, original positive legal support mask, one renormalization.
 z=np.pad(q,((1,1),(0,0)));z=(z[:-2]+2*z[1:-1]+z[2:])/4
 z=np.pad(z,((0,0),(1,1)));z=(z[:,:-2]+2*z[:,1:-1]+z[:,2:])/4
 z=np.where(q>0,z,0.);return z/z.sum()
def derive(ds,row):
 c,proof=source(ds,row);ids=c['ids'];q0=np.asarray(c['q0']);local,stats=evidence(c['features']['AuxE_ROI'],q0,kappa=0);total=inclusive_scores(local);q1,_=one_step(c['features']['AuxE_ROI'],q0,kappa=0,beta=1);assert np.array_equal(q1,update(q0,total));n=len(ids);length=np.arange(n)[None,:]-np.arange(n)[:,None]+1;mean=np.divide(total,length,out=np.zeros_like(total),where=length>0)
 qt=np.sqrt(q0);qt/=qt.sum();posteriors=dict(NoVisualUpdate=q0,FullMAP=q1,LengthMeanEvidence=update(q0,mean),TemperatureEvidence2=update(q0,total/2),SmoothPrior3=smooth(q0),TemperaturePrior2=qt)
 g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);iv0=np.asarray(map_interval(ids,q0));a=dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['original_interval']),SourceJointMAP=c['Source'].copy(),SourceJointMAP_interval=iv0,frame_ids=np.asarray(ids),positions8=np.asarray(c['pos']))
 for name,q in posteriors.items():
  assert np.isfinite(q).all()and(q>=0).all()and abs(q.sum()-1)<1e-10 and not(q[q0==0]>0).any();iv=np.asarray(map_interval(ids,q));a[name]=evidence_dp(g.for_interval(iv))['final'].numpy();a[name+'_interval']=iv
 assert np.array_equal(a['TemperaturePrior2'],a['NoVisualUpdate'])and np.array_equal(a['TemperaturePrior2_interval'],iv0)
 e=fixed(selected.D/f'components_{ds}_sealed.json')['files'][row['id']];assert sha(selected.D/e['path'])==e['sha256']and sha(selected.D/e['receipt_path'])==e['receipt_sha256']
 with np.load(selected.D/e['path'])as z:
  assert np.array_equal(a['FullMAP'],z['FullMAP'])and np.array_equal(a['FullMAP_interval'],z['FullMAP_interval'])and np.array_equal(a['Source'],z['Source'])and np.array_equal(a['Source_interval'],z['Source_interval'])and np.array_equal(a['SourceJointMAP_interval'],z['SourceJointMAP_interval'])
 hi=np.unravel_index(int(q0.argmax()),q0.shape);w=membership(q0);outside=1-w
 diagnostics=dict(q0_MAP_probability=float(q0[hi]),q1_probability_at_q0_MAP=float(q1[hi]),prior_MAP_probability_increased=bool(q1[hi]>q0[hi]),q0_entropy=float(-(q0[q0>0]*np.log(q0[q0>0])).sum()),q1_entropy=float(-(q1[q1>0]*np.log(q1[q1>0])).sum()),mean_local_evidence_inside_q0_weight=float(local@w/w.sum())if w.sum()else None,mean_local_evidence_outside_q0_weight=float(local@outside/outside.sum())if outside.sum()else None,local_evidence_std=float(local.std()),prototype_self_inclusion=True,pre_text_fusion_features=True,one_update_no_iterative_refitting=True)
 proof.update(current_main_prediction_sha256=e['sha256'],current_main_receipt_sha256=e['receipt_sha256'],same_E_ROI_features_and_MAIN_geometry=True,no_first_aux_DP=True,new_Source=0,new_expert=0,new_optimizer=0,GT_read=False)
 assert all(np.isfinite(v).all()for v in a.values());return a,proof,diagnostics
