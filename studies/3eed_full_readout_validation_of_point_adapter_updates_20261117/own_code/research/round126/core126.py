"""Frozen full-readout validator of already generated R125 updates. No new gradient."""
from pathlib import Path
import json,hashlib,numpy as np,torch,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round126';B=R/'crossplatform/round126';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NEW=['fullval_'+s+'_'+v for s in ['corr','wrong','random','free','corr013'] for v in ['F','C']]+['full_teacher_bridge','full_direct_AB','full_direct_ABC','double_zero_F','double_zero_C'];assert len(NEW)==15
def head64(h,u,mask,late,res,S):
 x=torch.as_tensor(h,dtype=torch.float64)
 for k in ['0','2']:x=F.relu(F.linear(x,torch.as_tensor(S[k+'.weight'],dtype=torch.float64),torch.as_tensor(S[k+'.bias'],dtype=torch.float64)))
 raw=F.linear(x,torch.as_tensor(S['4.weight'],dtype=torch.float64),torch.as_tensor(S['4.bias'],dtype=torch.float64));q=F.normalize(raw+torch.as_tensor(late,dtype=torch.float64),dim=-1);s=q@torch.as_tensor(u,dtype=torch.float64).T/.07;m=torch.as_tensor(mask).bool();v=s[:,m].logsumexp(-1)-s[:,~m].logsumexp(-1)+torch.as_tensor(res,dtype=torch.float64);return v.log_softmax(0).numpy()
def Ares(e):
 co=e['cohort'];name=e['source']+'_to_'+e['target'];sid=e['row']['id'];p=R/('crossplatform/round113' if co=='DEV' else 'crossplatform/round109')/name/'features_PRIVATE'/(sid+'.npz');z=np.load(p);return z['frozen_full_A_r' if co=='DEV' else 'real_A_r'],{'path':str(p.relative_to(R)),'sha256':sha(p)}
def options():return {'corr':['zero','corr3'],'wrong':['zero','wrong3'],'random':['zero','random3'],'free':['zero','free3'],'corr013':['zero','corr1','corr3']}
def choose(loss,ks):
 i=min(range(len(ks)),key=lambda i:loss[ks[i]]);best=ks[i];tol=max(1e-8,1e-6*abs(loss['zero']));return best if loss['zero']-loss[best]>tol else 'zero'
def process(e,S):
 D=R/'crossplatform/round125/formal'/e['cohort']/(e['source']+'_to_'+e['target']);sid=e['row']['id'];samples=np.load(D/'samples_PRIVATE'/(sid+'.npz'))
 if e['protected_alias_fallback']:
  po=dict(zip(samples['arm_names'],samples['scores']));return {k:po['prepared_A'].copy() for k in NEW},{'protected_alias_fallback':True}
 f=np.load(D/'features_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));late=f['fixed_late'];rA,proof=Ares(e);mask=f['zero_F_mask'];lp={}
 # Only A/B fit tensors and source-A RGB/context are used for targets and choices.
 for setting in ['zero','corr1','corr3','wrong3','random3','free3']:
  lp[setting]=[head64(f[setting+'_fit_'+v+'_h'],f[setting+'_fit_'+v+'_u'],mask,late,rA,S) for v in ['A','B']]
 T=(np.exp(lp['zero'][0])+np.exp(lp['zero'][1]))/2;logT=np.log(T);loss={k:float((T*(logT-(a+b)/2)).sum())+rec['auxiliary'][k]['fit_keep'] for k,(a,b) in lp.items()};picked={k:choose(loss,ks) for k,ks in options().items()};choices_fixed=True
 # Only after all choices are fixed may existing C-dependent final readouts be read.
 po=dict(zip(samples['arm_names'],samples['scores']));rF=f['fixed_r'];out={}
 for kind,state in picked.items():
  for v in ['F','C']:out['fullval_'+kind+'_'+v]=po[('mean_context' if v=='F' else 'zero_C') if state=='zero' else state+'_'+v].copy()
 bridge=logT+rF.astype(float)-rA.astype(float);out['full_teacher_bridge']=np.exp(bridge-np.logaddexp.reduce(bridge));pABC=[]
 for v in ['A','B','C']:
  hp=f['teacher_'+v+'_h'] if v in ['A','B'] else f['zero_C_h'];up=f['teacher_'+v+'_u'] if v in ['A','B'] else f['zero_C_u'];pABC.append(np.exp(head64(hp,up,mask,late,rF,S)))
 out['full_direct_AB']=(pABC[0]+pABC[1])/2;out['full_direct_ABC']=sum(pABC)/3
 for v in ['F','C']:out['double_zero_'+v]=np.exp(head64(f['zero_'+v+'_h'],f['zero_'+v+'_u'],mask,late,rF,S))
 return out,{'protected_alias_fallback':False,'source_A_provider':proof,'choices':picked,'full_AB_objectives':loss,'full_teacher_entropy':float(-(T*logT).sum()),'choices_fixed_before_C_or_final_scores_used':choices_fixed,'B_active':rec['fit']['active'],'C_available':rec['C_available'],'no_new_gradients_or_model_updates':True,'source_GT_used':False}
