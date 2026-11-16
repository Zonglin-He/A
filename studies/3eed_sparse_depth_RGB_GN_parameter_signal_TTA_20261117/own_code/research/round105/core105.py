"""Actual additional RGB banks; unchanged source-trained network and native readout."""
from pathlib import Path
import sys,contextlib
R105=Path(__file__).resolve().parents[2];H105=R105/'research/round105';B105=R105/'crossplatform/round105'
sys.path.insert(0,str(R105/'research/round99'));from core99 import *
import geometry105 as geo
@contextlib.contextmanager
def banks105(m,images):
 # Encode each observation separately. GroupNorm must never mix image banks.
 extra=[m.rgb((x-.5)*2) for x in images]
 hook=m.rgb.register_forward_hook(lambda _m,_a,o:torch.cat([o]+extra,dim=2)) if extra else None
 try:yield
 finally:
  if hook is not None:hook.remove()
def tensor_image105(g):return torch.as_tensor(g['image'].copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255
def data105(g,extra,r):return tensor_data(geo.merge_graphs([g]+list(extra)),r)
def native105(m,base,z,d,route,S,extra_images):
 setmatch91(m,S);m.route=route;m.use_interface=True;m.variant='full'
 with banks105(m,extra_images):p,q,e=m.main(base,z,d,'bidirectional')
 f={k:e[j].detach() for k,j in [('h','query_hidden'),('u','projected_text'),('late','late_RGB_delta')]};f.update(mask=z['token_mask'],r=e['residual'].detach().double(),p=p.detach());return f
def read105(m,f):return read94(m,f,f['r'])
def errors105(m,views,targets):
 ss=torch.stack([read105(m,f).log() for f in views]);e=targets-ss;return e-e.mean(1,keepdim=True),ss
def lm105(m,views,targets,S,maxiter=10):
 # R99 LM budget/acceptance, each view's head-independent frozen residual.
 put94(m,S);m.requires_grad_(False);records=[];initial=damping=None;reason='max_iterations';nv=len(views);n=nv*256
 for it in range(maxiter):
  with torch.no_grad():
   aa,gg,_=zip(*(components_solve(m,f,f['r']) for f in views));a=torch.cat(aa);g=torch.cat(gg);e,_=errors105(m,views,targets);before=float(e.square().mean()/2)
   if initial is None:initial=before
   if before<=max(1e-12,initial*1e-3):reason='relative_error';break
   G=(g@g.T)*(a@a.T+1);G=G.reshape(nv,256,nv,256);G=G-G.mean(1,keepdim=True)-G.mean(3,keepdim=True)+G.mean((1,3),keepdim=True);G=G.reshape(n,n);trace=float(G.trace())
   if damping is None:damping=max(1e-12,.01*trace/n)
   old=patch94(m);accepted=False
   for tr in range(6):
    used=damping;v=torch.linalg.solve(G+used*torch.eye(n,device='cuda',dtype=torch.float64),e.flatten());v=v.reshape(nv,256);v-=v.mean(1,keepdim=True);v=v.flatten();dW=(v[:,None]*g).T@a;db=(v[:,None]*g).sum(0);st={'weight':old['weight']+dW.float().cpu().numpy(),'bias':old['bias']+db.float().cpu().numpy()};put94(m,st);dw=torch.as_tensor(st['weight']-old['weight'],device='cuda').double();dbb=torch.as_tensor(st['bias']-old['bias'],device='cuda').double();pred=(g*F.linear(a,dw,dbb)).sum(-1).reshape(nv,256);pred-=pred.mean(1,keepdim=True);predred=float((e.square().mean()-(e-pred).square().mean())/2);ee,_=errors105(m,views,targets);after=float(ee.square().mean()/2);rho=(before-after)/max(predred,1e-30);ok=bool(predred>0 and np.isfinite(after) and after<before and rho>.1);records.append({'iteration':it+1,'trial':tr+1,'before':before,'after':after,'predicted_reduction':predred,'rho':rho,'damping':used,'accepted':ok,'step_L2':float(np.sqrt(sum(np.square((st[k]-old[k]).astype(np.float64)).sum() for k in st)))})
    if ok:
     accepted=True
     if rho>.75:damping=max(1e-15,damping/3)
     elif rho<.25:damping*=2
     break
    put94(m,old);damping*=10
   if not accepted:reason='no_accepted_step';break
 with torch.no_grad():e,_=errors105(m,views,targets);after=float(e.square().mean()/2)
 final=patch94(m);return final,records,{'initial_error':initial if initial is not None else after,'final_error':after,'relative_error':after/max(initial or 0.,1e-30),'stop_reason':reason,'accepted_steps':sum(x['accepted'] for x in records),'attempted_steps':len(records),'parameter_L2':float(np.sqrt(sum(np.square((final[k]-S[k]).astype(np.float64)).sum() for k in final))),'fit_views':nv}
def feature_save105(views):return {name+'_'+k:v.detach().cpu().numpy() for name,f in views.items() for k,v in f.items()}
