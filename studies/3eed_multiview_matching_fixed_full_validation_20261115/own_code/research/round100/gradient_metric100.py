"""Does uniform-score versus distribution supervision change actual head gradient direction?"""
from pathlib import Path
import json,numpy as np,torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round100';D=H/'loss_mass_diagnostic';B=R/'crossplatform/round99';OLD=R/'crossplatform/round98';rd=lambda p:json.loads(Path(p).read_text())
def load(p):z=np.load(p,allow_pickle=False);return {k:z[k] for k in z.files}
def main():
 torch.set_num_threads(4);coll=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json');out=[]
 with torch.no_grad():
  for src in ['waymo','drone','quad']:
   S=load(R/coll['modules'][src]['state_path']);W=torch.as_tensor(S['match.4.weight']).double();b=torch.as_tensor(S['match.4.bias']).double()
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt;rows=rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')
    for r in rows:
     sid=r['id'];ff=load(B/name/'features_PRIVATE'/(sid+'.npz'));ev=load(OLD/name/'evidence_PRIVATE'/(sid+'.npz'));r0=torch.as_tensor(ev['r0']);delta=torch.as_tensor(ev['delta']);grad={k:{'weight':torch.zeros_like(W),'bias':torch.zeros_like(b)} for k in ['score_L2','distribution_KL']}
     for view in range(3):
      h=torch.as_tensor(ff[str(view)+'_h']);a=F.relu(F.linear(h,torch.as_tensor(S['match.0.weight']),torch.as_tensor(S['match.0.bias'])));a=F.relu(F.linear(a,torch.as_tensor(S['match.2.weight']),torch.as_tensor(S['match.2.bias']))).double();raw=F.linear(a,W,b)+torch.as_tensor(ff[str(view)+'_late']).double();norm=raw.norm(dim=-1,keepdim=True).clamp_min(1e-12);qvec=raw/norm;u=torch.as_tensor(ff[str(view)+'_u']).double();mask=torch.as_tensor(ff[str(view)+'_mask']).bool();logits=qvec@u.T/.07;g=(logits[:,mask].softmax(-1)@u[mask]-logits[:,~mask].softmax(-1)@u[~mask])/.07;g=(g-qvec*(g*qvec).sum(-1,keepdim=True))/norm;native=torch.as_tensor(ev['correct'][view])+r0;p=native.softmax(0);target=(native+delta).softmax(0)
      for kind,e in [('score_L2',delta/256),('distribution_KL',target-p)]:
       grad[kind]['weight']+=(e[:,None]*g).T@a/3;grad[kind]['bias']+=(e[:,None]*g).sum(0)/3
     vectors={k:torch.cat([x['weight'].flatten(),x['bias']]) for k,x in grad.items()};x,y=vectors.values();cos=float(F.cosine_similarity(x,y,dim=0));out.append({'source':src,'target':tgt,'id_PRIVATE':sid,'head_gradient_cosine':cos,'score_L2_gradient_norm':float(x.norm()),'distribution_KL_gradient_norm':float(y.norm())})
    print('GRADIENT_METRIC100',name,flush=True)
 assert len(out)==768;summary=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   z=[r['head_gradient_cosine'] for r in out if r['source']==src and r['target']==tgt]
   if z:summary.append({'source':src,'target':tgt,'n':len(z),'mean_cosine':float(np.mean(z)),'median_cosine':float(np.median(z)),'p10_cosine':float(np.quantile(z,.1)),'negative_cosines':sum(x<0 for x in z)})
 (D/'PARAMETER_GRADIENT_ROWS_PRIVATE.json').write_text(json.dumps(out,indent=2)+'\n');(D/'PARAMETER_GRADIENT_SUMMARY.json').write_text(json.dumps({'state':'DESCRIPTIVE_FROZEN_HEAD_GRADIENT_COMPARISON_NO_UPDATE','n':768,'equal_six_mean_cosine':float(np.mean([r['mean_cosine'] for r in summary])),'directions':summary,'GT_used':False,'no_proposal_applied':True},indent=2)+'\n');print('PARAM_GRAD_METRIC',float(np.mean([r['mean_cosine'] for r in summary])))
if __name__=='__main__':main()
