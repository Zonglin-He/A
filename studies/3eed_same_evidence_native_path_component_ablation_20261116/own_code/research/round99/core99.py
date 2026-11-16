"""Only the number of fitted views changes; source, G, head and LM rules remain fixed."""
from pathlib import Path
import sys
R99=Path(__file__).resolve().parents[2];H99=R99/'research/round99';B99=R99/'crossplatform/round99'
sys.path.insert(0,str(R99/'research/round98'));from core98 import *
FITS99=['joint3','joint3_shuffle']
REFS99=['original_source','prepared_frozen','binding_direct','R94_Adam','frozen_C','frozen_FC','frozen_four','direct_C','direct_FC','direct_four','lm_F','lm_C','lm_FC','lm_shuffle_FC']
ARMS99=REFS99+[n+'_'+v for n in FITS99 for v in ['F','C','FC']]
def errors99(m,views,r0,targets):
 ss=torch.stack([read94(m,f,r0).log() for f in views]);e=targets-ss;return e-e.mean(1,keepdim=True),ss
def lmviews99(m,views,r0,targets,S,maxiter=10):
 put94(m,S);m.requires_grad_(False);records=[];initial=damping=None;reason='max_iterations';nv=len(views);n=nv*256
 for it in range(maxiter):
  with torch.no_grad():
   aa,gg,_=zip(*(components_solve(m,f,r0) for f in views));a=torch.cat(aa);g=torch.cat(gg);e,_=errors99(m,views,r0,targets);before=float(e.square().mean()/2)
   if initial is None:initial=before
   if before<=max(1e-12,initial*1e-3):reason='relative_error';break
   G=(g@g.T)*(a@a.T+1);G=G.reshape(nv,256,nv,256);G=G-G.mean(1,keepdim=True)-G.mean(3,keepdim=True)+G.mean((1,3),keepdim=True);G=G.reshape(n,n);trace=float(G.trace())
   if damping is None:damping=max(1e-12,.01*trace/n)
   old=patch94(m);accepted=False
   for tr in range(6):
    used=damping;v=torch.linalg.solve(G+used*torch.eye(n,device='cuda',dtype=torch.float64),e.flatten());v=v.reshape(nv,256);v=v-v.mean(1,keepdim=True);v=v.flatten();dW=(v[:,None]*g).T@a;db=(v[:,None]*g).sum(0);delta={'weight':dW.float().cpu().numpy(),'bias':db.float().cpu().numpy()};st={k:old[k]+delta[k] for k in old};put94(m,st);dw=torch.as_tensor(st['weight']-old['weight'],device='cuda').double();dbb=torch.as_tensor(st['bias']-old['bias'],device='cuda').double();pred=(g*F.linear(a,dw,dbb)).sum(-1).reshape(nv,256);pred-=pred.mean(1,keepdim=True);predred=float((e.square().mean()-(e-pred).square().mean())/2);ee,_=errors99(m,views,r0,targets);after=float(ee.square().mean()/2);rho=(before-after)/max(predred,1e-30);ok=bool(predred>0 and np.isfinite(after) and after<before and rho>.1);records.append({'iteration':it+1,'trial':tr+1,'before':before,'after':after,'predicted_reduction':predred,'rho':rho,'damping':used,'accepted':ok,'step_L2':float(np.sqrt(sum(np.square((st[k]-old[k]).astype(np.float64)).sum() for k in st)))})
    if ok:
     accepted=True
     if rho>.75:damping=max(1e-15,damping/3)
     elif rho<.25:damping*=2
     break
    put94(m,old);damping*=10
   if not accepted:reason='no_accepted_step';break
 with torch.no_grad():e,_=errors99(m,views,r0,targets);after=float(e.square().mean()/2)
 final=patch94(m);return final,records,{'initial_error':initial if initial is not None else after,'final_error':after,'relative_error':after/max(initial or 0.,1e-30),'stop_reason':reason,'accepted_steps':sum(x['accepted'] for x in records),'attempted_steps':len(records),'parameter_L2':float(np.sqrt(sum(np.square((final[k]-S[k]).astype(np.float64)).sum() for k in final))),'fit_views':nv}
