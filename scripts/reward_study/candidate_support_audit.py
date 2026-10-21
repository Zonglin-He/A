import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import torch,numpy as np,json
from pathlib import Path
torch.set_num_threads(1);ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();T=ROOT/'outputs/layout_reward_dev';O=T/'research'
rows=json.loads((T/'scope_hc2/private_case_mapping.json').read_text());total=0;anybad=0;allbad=0;badmass=0.;largeweight=[];badr=[];goodr=[];insideESS=[];comp=[]
u=torch.tensor([-1/3,0,1/3],dtype=torch.float64);y,x=torch.meshgrid(u,u,indexing='ij');delta=torch.stack((x.flatten(),y.flatten()),-1)
query=[]
for row in rows:
 z=torch.load(T/'traces'/f"{row['private_id']}.pt",map_location='cpu',weights_only=False)['LayoutOPD'];sums=[]
 for b in z['rounds']:
  rr=b['rollout'];bb=rr['boxes'].double();w=rr['weights'].double();r=rr['rewards'].double();xy=bb[:,:,:2,None].transpose(-1,-2)+bb[:,:,2:,None].transpose(-1,-2)*delta[None,None]
  invalid=((xy<0)|(xy>1)).any(-1);ib=invalid.any(-1);ab=invalid.all(-1)
  n=ib.numel();total+=n;anybad+=int(ib.sum());allbad+=int(ab.sum());badmass+=float((w*ib).sum())
  largeweight.extend(w[:,16:].sum(-1).tolist());badr.extend(r[ib].tolist());goodr.extend(r[~ib].tolist());sums.append(float(ib.double().mean()))
 query.append(dict(id=row['private_id'],public_id=row['public_id'],any_ROI_cell_outside_fraction=float(np.mean(sums))))
result=dict(candidate_count=total,candidate_any_ROI_cell_outside=anybad,candidate_any_ROI_cell_outside_fraction=anybad/total,candidate_all_ROI_cells_outside=allbad,candidate_all_ROI_cells_outside_fraction=allbad/total,weighted_probability_on_any_outside=badmass/(128*40*8),large_scale_weight_mean=float(np.mean(largeweight)),large_scale_weight_median=float(np.median(largeweight)),outside_reward_mean=float(np.mean(badr)),all_inside_reward_mean=float(np.mean(goodr)),query=query,note='Outside is physical normalized ROI grid coordinate only; zero-feature/padding invalidity may add exclusions. Each scale has exactly16 proposals; all remain in normalization.')
(O/'candidate_support_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='query'},indent=2))
