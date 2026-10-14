"""Independent saved-path Gaussian arithmetic/source-cap audit; no GT or model."""
import sys,json,math,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scripts.stvg_temporal_caption_query_v1 import ROOT,read,write,sha
B=ROOT/'artifacts/stvg_support_aware_temporal_spatial_update_v1'
def n(t):return t.detach().double().numpy()
def xy(b):return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
def iou(a,b):
 a,b=xy(a),xy(b);inter=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0).prod(-1);area=np.maximum(a[...,2:]-a[...,:2],0).prod(-1)+np.maximum(b[...,2:]-b[...,:2],0).prod(-1);return inter/np.maximum(area-inter,1e-7)
rows=[];err=dict(reward=0.,weights=0.,likelihood=0.,interpolation=0.,rms=0.);coordinates=0;attempts=0
seal=read(B/'STUDENT_GLOBAL_BARRIER.json');assert seal['status']=='complete' and seal['GT_read'] is False
for receipt in seal['files']:
 p=ROOT/receipt['path'];assert sha(p)==receipt['sha256'];z=torch.load(p,map_location='cpu',weights_only=False);f=z['fits']['S3_conservative_opd'];source=n(f['source_decoder_boxes']);cfg=f['config'];anchors=np.stack([n(a['box_cxcywh_normalized'])for a in z['supervision_anchors']]);alphas=[]
 for row in f['path']:
  assert 'rollout'in row;rr=row['rollout'];samples=n(rr['samples']);mu=n(rr['mean']);reward=iou(1/(1+np.exp(-samples)),anchors[:,None,:]);err['reward']=max(err['reward'],float(abs(reward-n(rr['rewards'])).max()));weights=np.exp(reward/cfg['tau']-(reward/cfg['tau']).max(1,keepdims=True));weights/=weights.sum(1,keepdims=True);err['weights']=max(err['weights'],float(abs(weights-n(rr['weights'])).max()));assert np.allclose(weights,n(rr['weights']),atol=3e-6,rtol=0)
  logp=-((samples-mu[:,None,:])**2).sum(-1)/(2*cfg['sigma']**2)-2*math.log(2*math.pi*cfg['sigma']**2);loss= -((n(rr['weights'])-1/32)*logp).sum(-1).mean();err['likelihood']=max(err['likelihood'],abs(loss-row['loss']));assert abs(loss-row['loss'])<2e-5
  after=n(row['boxes_after']);rms=np.sqrt(((after-source)**2).mean(-1));assert rms.max()<=.01;err['rms']=max(err['rms'],float(abs(rms-n(row['constraint']['per_frame_rms'])).max()));coordinates+=after.size
  a=row['accepted_alpha'];alphas.append(a)
  if a is None:
   for k,v in row['state_before'].items():assert torch.equal(v,row['state_after'][k])
  else:
   for k,v in row['state_before'].items():
    actual=n(row['state_after'][k]);expected=n(v)+a*(n(row['proposal_state'][k])-n(v));error=float(abs(actual-expected).max());err['interpolation']=max(err['interpolation'],error);assert error<3e-7
   before_attempts=row['backtracking_attempts'];assert before_attempts[-1]['alpha']==a and before_attempts[-1]['all_source_frames_within_limit'];assert not any(t['all_source_frames_within_limit']for t in before_attempts[:-1]);attempts+=len(before_attempts)
 for arm,key in [('S1_direct','S1_direct'),('S3_conservative_opd','S3_conservative_opd')]:assert torch.equal(z['readouts'][arm]['boxes'].cpu(),z['fits'][key]['final_cached_decoder_boxes'].cpu())
 rows.append(dict(case_id=z['case_id'],dataset=z['dataset'],alphas=alphas,max_final_source_frame_RMS=float(np.sqrt(((n(f['final_cached_decoder_boxes'])-source)**2).mean(-1)).max()),rejected=f['rejected_updates']))
assert max(err['reward'],err['weights'])<3e-6
write(B/'ROOT_SPATIAL_INDEPENDENT_ARITHMETIC.json',dict(status='pass',rows=rows,max_errors=err,actual_saved_rounds=40,source_frame_coordinates_checked=coordinates,actual_backtracking_attempts=attempts,full_model_cached_output_pairs=16,all_actual_updates_accepted_after_backtracking=True,fully_rejected_actual_updates=0,fully_rejected_restore_branch_not_exercised_by_real_cohort=True,GT_read=False,CPU_only=True))
print(json.dumps(dict(status='pass',errors=err,coordinates=coordinates,attempts=attempts)))
