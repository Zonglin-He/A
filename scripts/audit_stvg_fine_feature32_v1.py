"""CPU audit of finer-grid correspondence, old candidate identity and full scores."""
from pathlib import Path
import os,sys,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_feature32_v1 import D,P,PRE,OLD,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 import torch.nn.functional as F
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
 from vg_tta.stvg_fine_feature32_v1 import LAYERS
 torch.set_num_threads(4);p=verify();z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxmap=0.;maxr=0.
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));f=load(D/'features'/(a+'.pt'));x=load(PRE/'features'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));E=x['E'];pos=x['planned_positions'];assert np.array_equal(o['outputs']['E_MBR'],E.numpy())and np.array_equal(o['outputs']['Coarse'],old['outputs']['Cosine_readout']);checks['exact_old_E_and_coarse_readout']+=1
  assert o['interval']==x['interval']and o['frame_ids']==x['frame_ids'];checks['exact_frozen_time_and_grid']+=1
  assert f['maps'].shape[1]==1024 and f['maps'].shape[-2]>x['maps'].shape[-2];checks['true_finer_backbone_capture']+=1
  dsmap=F.adaptive_avg_pool2d(f['maps'].float(),x['maps'].shape[-2:]);dv=F.interpolate(f['valid'][:,None].float(),size=x['maps'].shape[-2:],mode='nearest')[:,0].bool()
  # Fixed actual non-anchor frame and all admitted references: independent
  # normalized features / affinity / region map, CPU vs captured GPU numerics.
  t=next(i for i in range(len(E))if i not in pos)
  for layer,raw,valid in [('Fine_layer3',f['maps'],f['valid']),('Fine_downsampled',dsmap,dv)]:
   h,w=raw.shape[-2:];target=raw[t].float().flatten(1).T;target=target/target.norm(dim=-1,keepdim=True).clamp_min(1e-12)
   for j,(ap,box)in enumerate(zip(f['positions'],f['boxes'])):
    ref=raw[ap].float().flatten(1).T;ref=ref/ref.norm(dim=-1,keepdim=True).clamp_min(1e-12);logits=(target@ref.T)/.07;logits[:,~valid[ap].flatten()]=-torch.inf
    # Same fractional geometry, evaluated independently at the actual ROI.
    b=box.double();lo=b[:2]-b[2:]/2;hi=b[:2]+b[2:]/2;xx=torch.arange(w,dtype=torch.float64);yy=torch.arange(h,dtype=torch.float64)
    dx=(torch.minimum(hi[0]*w,xx+1)-torch.maximum(lo[0]*w,xx)).clamp(0,1);dy=(torch.minimum(hi[1]*h,yy+1)-torch.maximum(lo[1]*h,yy)).clamp(0,1);region=(dy[:,None]*dx[None]).float().flatten()*valid[ap].flatten();m=(logits.softmax(-1)@region).reshape(h,w)*valid[t]
    err=float(abs(m-o['soft_maps'][layer][j,t]).max());maxmap=max(maxmap,err);assert err<2e-5,(a,layer,err);checks['CPU_GPU_actual_fine_affinity_ROI_map']+=1
  for layer in LAYERS:
   rd=o['readouts'][layer];assert torch.equal(rd['candidates'],old['direct']['candidates']);b=rd['candidates'];ss=o['soft_maps'][layer];valid=o['valid'][layer];H,W=valid.shape[-2:];area=cell_overlap(b,H,W);ok=(area*valid[:,None]).sum((-1,-2))>1e-12;assert torch.equal(ok,rd['feedback']['valid'])
   if len(ss):
    inter=(area[None]*ss[:,:,None].double()).sum((-1,-2));den=b[None,...,2]*b[None,...,3]*H*W+ss.double().sum((-1,-2))[:,:,None]-inter;rr=torch.where(den>0,inter/den,torch.zeros_like(inter));err=float(abs(rr-rd['feedback']['per_anchor_rewards']).max());maxr=max(maxr,err);assert err<1e-12;reward=rr.mean(0);checks['independent_soft_IoU_reward_scalar']+=rr.numel()
   else:reward=torch.zeros_like(rd['feedback']['rewards'])
   eps=load(PRE/'feedback'/(a+'.pt'))['feedback']['noise'];ee=torch.cat((eps[0],eps[1]),1);lp=-.5*ee.square().sum(-1);safe=ok.clone();safe[~ok.any(1)]=True;lp[~safe]=-torch.inf;prior=lp.softmax(-1);pt=(lp+reward/.1).softmax(-1)
   informative=ok.any(1)&((reward.masked_fill(~ok,-torch.inf).amax(1)-reward.masked_fill(~ok,torch.inf).amin(1))>1e-12)if len(ss)else torch.zeros(len(E),dtype=torch.bool);pt[~informative]=prior[~informative];assert abs(pt-rd['feedback']['prob']).max()<1e-12;checks['same_posterior_formula_scalar']+=pt.numel()
   boxes=E.clone();use=informative.clone();use[pos]=False;best=pt.argmax(1);boxes[use]=b[torch.arange(len(E))[use],best[use]];assert np.array_equal(boxes.numpy(),o['outputs'][layer]);checks['actual_same_argmax_readout']+=1
   assert np.array_equal(boxes[pos],E[pos])and np.array_equal(boxes[:,2:],E[:,2:]);checks['size_and_planned_anchors_fixed']+=1
   if not f['positions']:assert torch.equal(boxes,E);checks['no_reference_noop']+=1
  independent_metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for name,ms in g['metrics'].items():
   for k,value in ms.items():
    v=[r['values'][name][k]for r in rr if r['values'][name][k]is not None];assert value is None if not v else abs(value-np.mean(v))<1e-12;checks['public_mean']+=1
  for c,stats in g['paired'].items():
   a,b=c.split('_minus_')
   for ep,t in stats.items():
    v=np.array([100*(r['values'][a][ep]-r['values'][b][ep])for r in rr if r['values'][a][ep]is not None and r['values'][b][ep]is not None]);ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));assert abs(v.mean()-t['mean'])<1e-12 and np.max(abs(np.quantile(v[ix].mean(1),[.025,.975])-t['CI95']))<1e-12;checks['independent_pair_mean_bootstrap']+=1
 verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_CPU_GPU_map_error=maxmap,max_reward_error=maxr,no_new_training=True,no_new_experts=True,all_outputs_sealed_before_NEW_GT=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maxmap,maxr,flush=True)

if __name__=='__main__':main()
