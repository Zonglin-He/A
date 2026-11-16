"""Paired native-original/prepared feature moment-state adaptation of existingCE BN."""
from pathlib import Path
import sys
R110=Path(__file__).resolve().parents[2];H110=R110/'research/round110';B110=R110/'crossplatform/round110';sys.path.insert(0,str(R110/'research/round108'));from core108 import *
BNS110=[1,5];BUFFER_KEYS110=[f'net.{j}.{k}' for j in BNS110 for k in ['running_mean','running_var']];PRIMARY110='transport_geometric_fusedABC'
def putce110(head,S):head.load_state_dict({k:torch.as_tensor(v,device='cuda') for k,v in S.items()},strict=True);head.eval().requires_grad_(False)
def pre110(head,h,j):return head.net[:j](h.T[None])
def moments110(x):
 x=x.detach().double();return x.mean((0,2)),x.var((0,2),unbiased=False)
def transport110(head,O,P,S):
 putce110(head,S)
 with torch.no_grad():reference={j:moments110(pre110(head,O,j)) for j in BNS110}
 records=[]
 with torch.no_grad():
  for j in BNS110:
   bn=head.net[j];mo,vo=reference[j];mp,vp=moments110(pre110(head,P,j));ms=torch.as_tensor(S[f'net.{j}.running_mean'],device='cuda').double();vs=torch.as_tensor(S[f'net.{j}.running_var'],device='cuda').double();eps=bn.eps;a=(vp/vo.clamp_min(eps)).sqrt();mn=mp+a*(ms-mo);vn=a.square()*(vs+eps)-eps;ok=(vo>eps)&(vp>eps)&(vn>=0)&torch.isfinite(mn)&torch.isfinite(vn);mn=torch.where(ok,mn,ms);vn=torch.where(ok,vn,vs);bn.running_mean.copy_(mn.float());bn.running_var.copy_(vn.float());records.append({'layer':j,'eligible_channels':int(ok.sum()),'reference_mean':mo.cpu().numpy(),'reference_var':vo.cpu().numpy(),'input_mean':mp.cpu().numpy(),'input_var':vp.cpu().numpy(),'a':a.cpu().numpy(),'valid':ok.cpu().numpy(),'eps':eps})
 result=statece108(head);assert all(np.array_equal(result[k],v) for k,v in S.items() if k not in BUFFER_KEYS110);norm=float(np.sqrt(sum(np.square(result[k].astype(float)-S[k].astype(float)).sum() for k in BUFFER_KEYS110)));return result,records,{'eligible_channels_per_layer':[x['eligible_channels'] for x in records],'buffer_L2':norm,'changed_values':sum(int(np.count_nonzero(result[k]!=S[k])) for k in BUFFER_KEYS110),'calibrated_running_stat_values':1152,'learned_parameters_changed':0,'C_or_GT_used':False}
def functional110(head,h,S,records):
 putce110(head,S);x=h.T[None]
 for j in range(9):
  if j in BNS110:
   z=next(v for v in records if v['layer']==j);muP=torch.as_tensor(z['input_mean'],device=x.device)[None,:,None];muO=torch.as_tensor(z['reference_mean'],device=x.device)[None,:,None];a=torch.as_tensor(z['a'],device=x.device)[None,:,None];ok=torch.as_tensor(z['valid'],device=x.device)[None,:,None];x=torch.where(ok,(x.double()-muP)/a.clamp_min(1e-30)+muO,x.double()).float()
  x=head.net[j](x)
 return x[0].T

def captureO110(base,r,src):
 pts,ids,_=sample_points(r)
 if r['platform']=='waymo':
  from utils.transform_waymo import transform_to_front_view
  pts,_=transform_to_front_view(pts,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pts=np.ascontiguousarray(pts,np.float32)
 logits=[];hook=base.prediction_heads[-1].sem_cls_scores_head.register_forward_hook(lambda _m,_a,o:logits.append(o.detach().cpu().numpy()[0].T))
 try:
  with torch.no_grad():z,proof=capture(base,pts,ids,r['caption'])
 finally:hook.remove()
 assert len(logits)==1;po=native_scores(torch.as_tensor(z['query'],device='cuda'),torch.as_tensor(z['tokens'],device='cuda'),torch.as_tensor(z['token_mask'],device='cuda').bool()).double().softmax(0).cpu().numpy();head=base.prediction_heads[-1].sem_cls_scores_head
 with torch.no_grad():pce,lc=ce108(head,torch.as_tensor(z['h'],device='cuda'),torch.as_tensor(z['token_mask'],device='cuda'))
 assert np.array_equal(lc.cpu().numpy(),logits[0]);old=load(R110/'crossplatform/round106'/(src+'_to_'+r['platform'])/'samples_PRIVATE'/(r['id']+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(z['boxes'],old['source_boxes']) and abs(po-oldp['original_source']).max()<1e-7
 return {'h':z['h'],'mask':z['token_mask'],'p':po,'pce':pce.cpu().numpy(),'CE_logits':logits[0],'query':z['query'],'tokens':z['tokens'],'point_indices':z['point_indices'],'source_boxes':z['boxes']},{**proof,'source_checkpoint_sha256':source97(src)['checkpoint_sha256'],'source_probability_error':float(abs(po-oldp['original_source']).max()),'native_CE_logits_bitexact':True,'no_new_RGB_decode':True}
