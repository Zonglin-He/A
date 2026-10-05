"""Meaningful synthetic CPU contracts, never production/GT prediction scores."""
import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from vg_tta.decota_paper_controls_v1 import dino_refine,severity_observation
from vg_tta.decota_paper_baselines_20261005_v1 import OnlineBaseline,profile,pseudo_native_loss
from vg_tta.native_probability_interface_v1 import NativeOutput,offsets_entropy
from vg_tta.decota_paper_baseline_math_v1 import audit_updates

def run():
 torch.set_num_threads(1);checks=[]
 def ck(k,yes):assert yes,k;checks.append(k)
 ids=[0,3,11,23];boxes=np.ones((4,4),dtype=np.float32)*.3
 out,rc=dino_refine(boxes,ids,dict(anchors=dict(single4=[])))
 ck('empty-preserves-Frozen',np.array_equal(out,boxes))
 anchors=[dict(position=1,box=[.2,.3,.1,.2]),dict(position=3,box=[.6,.7,.3,.4])]
 out,rc=dino_refine(boxes,ids,dict(anchors=dict(single4=anchors)))
 ck('observed-anchors-exact',np.array_equal(out[1],np.float32(anchors[0]['box'])) and np.array_equal(out[3],np.float32(anchors[1]['box'])))
 ck('physical-not-index-interpolation',np.allclose(out[2],np.float32(anchors[0]['box'])+.4*(np.float32(anchors[1]['box'])-np.float32(anchors[0]['box']))))
 ck('nearest-left',np.array_equal(out[0],out[1]))
 row=dict(source='synthetic-PaperContract',frame_ids=list(range(100)),input=dict(frame_count=100,frame_ids=list(range(100))))
 frames=np.arange(100*12*16*3,dtype=np.int64).reshape(100,12,16,3).astype(np.uint8)
 from scripts.run_tastvg_full_b1_experts_v1 import observation
 for family in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure']:
  ours,h,s=severity_observation(row,family,frames,5);old,oh,os=observation(row,family+'_5',frames)
  ck('5%-pixel-parity:'+family,np.array_equal(ours,old) and h==oh)
  ck('5%-burst-parity:'+family,all(s[k]==v for k,v in os.items()))
 for coverage,length in [(2.5,3),(5,5),(10,10)]:
  _,_,s=severity_observation(row,'frame_drop',frames,coverage);ck('float-coverage:'+str(coverage),s['length']==length and s['percentage']==coverage)
 def native(p):
  z=torch.stack([p[0]*torch.tensor([1.,0.,-1.,-.2]),p[1]*torch.tensor([-.5,.4,1.,0.])],1)
  a=p[0]*torch.tensor([1.,-.5,0.,2.]);return [NativeOutput(z,a,torch.ones(4,dtype=torch.bool),(0,2,7,12))]
 p=torch.nn.Parameter(torch.tensor([.6,-.4]));v=profile(native(p));ck('profile-detached-normalized',not v.requires_grad and torch.isclose(v.sum(),torch.tensor(1.)) and len(v)==66)
 ck('Fisher-surrogate-live',torch.isfinite(pseudo_native_loss(native(p))) and pseudo_native_loss(native(p)).requires_grad)
 p=torch.nn.Parameter(torch.tensor([.6,-.4]));ref=torch.nn.Parameter(p.detach().clone());refopt=torch.optim.Adam([ref],lr=.001)
 opt=OnlineBaseline(['p'],[p],'TENT');opt.audit_updates=True
 for _ in range(2):
  refopt.zero_grad();offsets_entropy(native(ref)).backward();refopt.step();_,_,ev=opt.arrive(lambda:native(p),lambda:p.detach().clone());ck('TENT-independent-NumPy:'+str(opt.arrivals),audit_updates(ev)['status']=='pass')
 ck('TENT-two-arrival-Adam-state',torch.equal(p,ref) and opt.arrivals==2)
 saved=opt.state_dict();opt.reset();ck('stream-reset-only',torch.equal(p,torch.tensor([.6,-.4])) and opt.arrivals==0)
 opt.load_state_dict(saved);ck('checkpoint-restores-online-state',torch.equal(p,ref) and opt.arrivals==2)
 try:OnlineBaseline(['p'],[p],'EATA');raise AssertionError('Missing Fisher accepted')
 except AssertionError as e:ck('EATA-no-Fisher-rejected','Fisher=None' in str(e))
 p=torch.nn.Parameter(torch.tensor([8.,8.]));fish=dict(names=['p'],values=[torch.ones_like(p)],source_parameters=[p.detach().clone()],source_inputs_only=True,source_queries=2000,target_labels_used=False)
 e=OnlineBaseline(['p'],[p],'EATA',fish,dict(lr=.00025,steps=1,optimizer='SGD',momentum=.9,margin_fraction=1.,d_margin=.05,fisher_alpha=2000.,profile_bins=32))
 e.audit_updates=True;_,_,t=e.arrive(lambda:native(p),lambda:p.detach().clone());ck('EATA-Fisher-and-first-update',t['counts']['optimizer_steps']==1 and e.probs is not None);ck('EATA-independent-SGD',audit_updates(t)['status']=='pass')
 _,_,t=e.arrive(lambda:native(p),lambda:p.detach().clone());ck('EATA-redundant-skip',t['trace'][0].get('skip')=='redundancy')
 p=torch.nn.Parameter(torch.tensor([.6,-.4]));s=OnlineBaseline(['p'],[p],'SAR',config=dict(lr=.001,steps=1,optimizer='SGD',momentum=.9,rho=.05,margin_fraction=1.,reset_entropy=0.))
 s.audit_updates=True;_,_,t=s.arrive(lambda:native(p),lambda:p.detach().clone());ck('SAR-two-live-backwards',t['counts']['backward_steps']==2 and t['counts']['optimizer_steps']==1);ck('SAR-independent-SGD',audit_updates(t)['status']=='pass')
 ck('SAR-no-perturbation-left',torch.linalg.vector_norm(p.detach()-torch.tensor([.6,-.4]))<.01)
 ck('CPU-only',not torch.cuda.is_initialized())
 print('PAPER_PORT_CPU_CONTRACTS_PASS',len(checks),'GPU_initialization',torch.cuda.is_initialized(),flush=True)
 from scripts.decota_paper_common_v1 import BASE,write
 from scripts.decota_paper_common_v1 import status
 status(BASE/'PORT_CPU_CONTRACTS_CURRENT.json',dict(status='pass',checks=checks,GT_read=False,model_forwards=False,CUDA_initialized=False,formal_baseline_qualified=False))
if __name__=='__main__':run()
