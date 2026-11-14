"""Re-linearized LM with gain-ratio acceptance, no source-relative weight cap."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round94'))
from solver94 import *
H95=ROOT/'research/round95';B95=ROOT/'crossplatform/round95';P94=B94/'prediction336_attempt001'
REFS95=['source','ABC','R59','R91_frozen','R94_direct','R94_mixture','R94_Adam','R94_GN']
DIRECT95=['frozen_C','direct_C','direct_FC','direct_four','mixture_FC','shuffle_FC']
NEW95=[n+'_'+v for n in ['lm','lm_mixture','lm_shuffle','lm_capped'] for v in ['F','C','FC']];ARMS95=REFS95+DIRECT95+NEW95

def score95(m,f,r0):
 # Exact existing last layer with float64 score reduction, matching original FP32 deployment.
 return read94(m,f,r0).log()
def error95(m,f,r0,target):
 s=score95(m,f,r0);e=target-s;e=e-e.mean();return e,s

def lm95(m,f,r0,target,S,kind,maxiter=10):
 put94(m,S);m.requires_grad_(False);records=[];states={};initial=None;damping=None;reason='max_iterations'
 for it in range(maxiter):
  with torch.no_grad():
   a,g,s=components_solve(m,f,r0);e,actual_s=error95(m,f,r0,target);before=float(e.square().mean()/2)
   if initial is None:initial=before
   if before<=max(1e-12,initial*1e-3):reason='relative_error';break
   G=(g@g.T)*(a@a.T+1);G=G-G.mean(0,keepdim=True)-G.mean(1,keepdim=True)+G.mean();trace=float(G.trace())
   if damping is None:damping=max(1e-12,.01*trace/256)
   old=patch94(m);accepted=False;trials=[]
   for tr in range(6):
    used=damping;v=torch.linalg.solve(G+used*torch.eye(256,device='cuda',dtype=torch.float64),e);v-=v.mean();dW=(v[:,None]*g).T@a;db=(v[:,None]*g).sum(0);delta={'weight':dW.float().cpu().numpy(),'bias':db.float().cpu().numpy()};norm=float(np.sqrt(sum(np.square(x.astype(np.float64)).sum() for x in delta.values())));cap_hit=False
    if kind=='lm_capped':
     limit=.05*float(np.sqrt(sum(np.square(x.astype(np.float64)).sum() for x in S.values())))
     if norm>limit:delta={k:x*(limit/norm) for k,x in delta.items()};cap_hit=True
    st={k:old[k]+delta[k] for k in old};put94(m,st);deW=torch.as_tensor(st['weight']-old['weight'],device='cuda').double();deb=torch.as_tensor(st['bias']-old['bias'],device='cuda').double();pred=(g*(F.linear(a,deW,deb))).sum(-1);pred-=pred.mean();predred=float((e.square().mean()-(e-pred).square().mean())/2);ee,_=error95(m,f,r0,target);after=float(ee.square().mean()/2);rho=(before-after)/max(predred,1e-30);ok=bool(predred>0 and np.isfinite(after) and after<before and rho>.1);key='i'+str(it)+'_t'+str(tr);states[key]=st;trials.append({'iteration':it+1,'trial':tr+1,'before':before,'after':after,'predicted_reduction':predred,'rho':rho,'damping':used,'accepted':ok,'cap_hit':cap_hit,'step_L2':float(np.sqrt(sum(np.square((st[k]-old[k]).astype(np.float64)).sum() for k in st))),'state_key':key})
    if ok:
     accepted=True
     if rho>.75:damping=max(1e-15,damping/3)
     elif rho<.25:damping*=2
     break
    put94(m,old);damping*=10
   records.extend(trials)
   if not accepted:reason='no_accepted_step';break
 with torch.no_grad():e,_=error95(m,f,r0,target);after=float(e.square().mean()/2)
 final=patch94(m);return final,states,records,{'initial_error':initial if initial is not None else after,'final_error':after,'relative_error':after/max(initial or 0.,1e-30),'stop_reason':reason,'accepted_steps':sum(x['accepted'] for x in records),'attempted_steps':len(records),'parameter_L2':float(np.sqrt(sum(np.square((final[k]-S[k]).astype(np.float64)).sum() for k in final)))}
