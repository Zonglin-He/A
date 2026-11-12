"""R86 fixed bases, conditional targets and discrimination-priority adaptation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round85'))
from core85 import *
from evidence86 import *
H86=ROOT/'research/round86';B86=ROOT/'crossplatform/round86'
BASELINES86=['source','frozen_bi','R82_BT','frozen_ABC','R59','R85_full1']
ARMS86=BASELINES86+[b+'_'+a for b in ['S','F'] for a in ['direct_raw','direct_guard','mismatch_direct_raw','mismatch_direct_guard']+[v+str(k)+'_'+rd for v,ks in [('candidate',[1,3]),('joint',[1,3]),('mismatch',[1,3]),('sum',[1])] for k in ks for rd in ['raw','guard']]]
def mask86(r,d,region,repeat):
 seed=int(hashlib.sha256(('R86globalcoverage|'+r['id']+'|'+str(repeat)).encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed);remaining=np.flatnonzero(d['depth_valid'].cpu().numpy());n=len(remaining)//4;orders=[]
 for _ in range(2):
  # First half is independent of all source/phrase rankings; local quota follows.
  global_cells=rng.permutation(remaining)[:int(np.ceil(n/2))];left=remaining[~np.isin(remaining,global_cells)];loc=rng.permutation(left[region['image'][left]])[:n-len(global_cells)];left=left[~np.isin(left,loc)];fill=rng.permutation(left)[:n-len(global_cells)-len(loc)];take=np.r_[global_cells,loc,fill];orders.append(take);remaining=remaining[~np.isin(remaining,take)]
 im=torch.zeros(1536,device='cuda',dtype=torch.bool);im[torch.tensor(orders[0],device='cuda')]=True;pm=d['visible']&torch.isin(d['cell'],torch.tensor(orders[1],device='cuda'));assert not im[d['cell'][pm]].any();return pm,im
def flat86(grad,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for g,p in zip(grad,ps)])
def objective86(m,base,z,d,s0,pi,baselines,pm,im,ws):return objective85(m,base,z,d,s0,pi,baselines,pm,im,ws)
def priority86(gc,gr):
 dot=gc@gr;projected=gr-torch.minimum(dot,dot.new_zeros(()))*gc/gc.square().sum().clamp_min(1e-20)
 return gc+projected,projected
def apply86(m,old,delta):
 off=0
 with torch.no_grad():
  for (_,p),v in zip(m.tta_parameters(),old):n=p.numel();p.copy_(v+delta[off:off+n].reshape_as(p));off+=n
 assert off==len(delta)
