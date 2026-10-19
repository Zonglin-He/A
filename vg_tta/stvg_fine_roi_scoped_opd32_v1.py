"""Existing true layer3 ROI feedback, unchanged per-node OPD, pure tube scope."""
import numpy as np
import torch
from vg_tta.stvg_roi_layout_opd32_v1 import build,feedback,patches,fit,direct
ARMS=['E_MBR','Coarse_direct','Coarse_fixed','Coarse_live','Coarse_shuffle','Fine_direct','Fine_fixed','Fine_live','Fine_shuffle']
MODES={'Fine_fixed':'Fixed_dense','Fine_live':'On_policy_dense','Fine_shuffle':'Shuffled_OPD'}

def scope(frame_ids,planned,interval):
 ids=np.asarray(frame_ids);assert len(ids)>1 and np.all(np.diff(ids)>0)
 # Union of interpolation supports intersecting predicted [start,end).
 # Conservative endpoints exactly reproduce the preexisting postseal partition.
 left=np.r_[ids[0],ids[:-1]];right=np.r_[ids[1:],ids[-1]]
 relevant=(right>=interval[0])&(left<interval[1]);allowed=relevant.copy();allowed[planned]=False
 return torch.from_numpy(relevant),torch.from_numpy(allowed)

def restore(E,boxes,allowed):
 out=torch.as_tensor(boxes).clone();out[~allowed]=E[~allowed];return out

def scoped_fit(E,context,eps,cfg,planned,mode,permutation,allowed):
 # Independent block updates: excluding nodes cannot change any retained block.
 excluded=torch.where(~allowed)[0].tolist();f=fit(E,context,eps,cfg,excluded,mode,permutation)
 f.update(allowed=allowed.clone(),planned_positions=list(planned),no_tent_amplitude=True)
 return f

def scoped_direct(E,context,eps,cfg,planned,allowed):
 return direct(E,context,eps,cfg,torch.where(~allowed)[0].tolist())

def interpolation_equivalence(frame_ids,interval,a,b):
 ids=np.asarray(frame_ids);times=np.arange(int(interval[0]),int(interval[1]));inside=(times>=ids[0])&(times<=ids[-1]);times=times[inside]
 aa=np.column_stack([np.interp(times,ids,np.asarray(a)[:,j])for j in range(4)])
 bb=np.column_stack([np.interp(times,ids,np.asarray(b)[:,j])for j in range(4)])
 return dict(times=len(times),max_error=float(np.max(abs(aa-bb)))if len(times)else 0.,exact=bool(np.array_equal(aa,bb)))

def properties():
 from vg_tta.stvg_discriminative_dense32_v1 import noise
 torch.manual_seed(20261019);E=torch.tensor([[.5,.5,.3,.4]]*9,dtype=torch.float64);ids=[0,3,8,12,19,20,30,38,50];planned=[0,3,6,8];iv=[14,24];relevant,allowed=scope(ids,planned,iv);assert relevant.tolist()==[False,False,False,True,True,True,True,False,False]
 maps=torch.randn(9,12,7,10);valid=torch.ones(9,7,10,dtype=torch.bool);ctx=build(maps,valid,[0],E[:1]);ep=noise('fine-roi-scope-properties',9);cfg=dict(center_scale=.25,temperature=.1,step_size=.5)
 for mode in MODES.values():
  raw=fit(E,ctx,ep,cfg,planned,mode,list(range(9)));f=scoped_fit(E,ctx,ep,cfg,planned,mode,list(range(9)),allowed)
  for a,b in zip(raw['path'],f['path']):assert torch.equal(a['boxes'][allowed],b['boxes'][allowed])and torch.equal(b['boxes'][~allowed],E[~allowed]);assert interpolation_equivalence(ids,iv,a['boxes'],b['boxes'])['exact']
 no=build(maps,valid,[],[])
 for mode in MODES.values():assert torch.equal(scoped_fit(E,no,ep,cfg,planned,mode,list(range(9)),allowed)['final'],E)
 f=scoped_fit(E,ctx,ep,cfg,planned,'Fixed_dense',list(range(9)),allowed);l=scoped_fit(E,ctx,ep,cfg,planned,'On_policy_dense',list(range(9)),allowed);assert torch.equal(f['path'][1]['boxes'],l['path'][1]['boxes'])
 return dict(status='pass',pure_scope_not_amplitude=True,retained_blocks_exact_original=True,interpolation_inside_predicted_interval_exact=True,boundary_neighbors_retained=True,no_reference_exact_noop=True,first_fixed_live_exact=True)
