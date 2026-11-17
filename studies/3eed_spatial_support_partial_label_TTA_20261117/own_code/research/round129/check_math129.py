"""Independent CPU checks for supervision scope, duplicates and direct controls."""
from pathlib import Path
import json,numpy as np,torch
from support129 import taskloss,glogits,collapse,keep_kl,direct
R=Path(__file__).resolve().parents[2];H=R/'research/round129';B=R/'crossplatform/round129'
def main():
 rows=json.load(open(H/'SCOPE_ROWS_PRIVATE.json'));checks=[]
 for e in [next(r for r in rows if r['source']==s and r['cohort']=='DEV' and r['partial_eligible']) for s in ['waymo','drone','quad']]:
  p=B/'scope_PRIVATE'/e['cohort']/(e['source']+'_to_'+e['target'])/(e['id_PRIVATE']+'.npz');f=np.load(p);groups=[np.flatnonzero(f['assignment']==j).tolist() for j in range(int(f['assignment'].max())+1)];S=torch.from_numpy(f['S']);C=torch.from_numpy(f['C']);T=torch.from_numpy(f['teacher_conditional']);lp=torch.from_numpy(f['pF']).log().requires_grad_(True)
  for kind in ['conditional','partial']:
   L=taskloss(lp.log_softmax(0),groups,S,C,T,kind);g=torch.autograd.grad(L,lp,retain_graph=True)[0];qS=S[torch.from_numpy(f['assignment'])];error=float(g[~qS].abs().max()) if (~qS).any() else 0.;assert error<1e-12
   checks.append({'source':e['source'],'kind':kind,'direct_logit_gradient_outside_S_max':error})
  zero=taskloss(lp.log_softmax(0),groups,S,S,T,'partial');zg=torch.autograd.grad(zero,lp,retain_graph=True)[0];assert float(abs(zero))==0 and torch.count_nonzero(zg)==0
  ref=collapse(lp.detach().softmax(0),qS)
  for kind in ['conditional','partial']:
   pp=direct(f['pF'],groups,S.numpy(),C.numpy(),T.numpy(),kind);assert abs(pp.sum()-1)<1e-12 and (pp>=0).all();u=~qS.numpy();assert np.max(abs(pp[u]-f['pF'][u]))<1e-12 if u.any() else True
   mass=float(keep_kl(ref,collapse(torch.from_numpy(pp),qS)));assert mass<1e-12
   loss=float(taskloss(torch.from_numpy(np.log(np.maximum(pp,1e-300))),groups,S,C,T,kind));assert abs(loss)<1e-10,(kind,loss)
   checks.append({'source':e['source'],'kind':kind,'direct_projection_target_loss':loss,'unknown_KL':mass})
 # Duplicate a member, unchanged score: the group's logmeanexp is invariant.
 l=torch.tensor([.2,.2,-.7],dtype=torch.float64);a=glogits(l,[[0],[2]]);b=glogits(l,[[0,1],[2]]);assert torch.allclose(a,b,atol=1e-15,rtol=0)
 out={'state':'PASS_GRADIENT_SCOPE_PARTIAL_ZERO_AND_DIRECT_PROJECTIONS','n_real_sources':3,'checks':checks,'duplicates_do_not_add_probability_votes':True,'task_GT_loaded':False};(H/'MATH_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
