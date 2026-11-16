"""Same R99 LM; update only the existing geometry-bound late RGB linear layer."""
from pathlib import Path
import sys
R104=Path(__file__).resolve().parents[2];H104=R104/'research/round104';B104=R104/'crossplatform/round104'
sys.path.insert(0,str(R104/'research/round99'));from core99 import *
FITS104=['late','late_shuffle'];REFS104=ARMS99;ARMS104=REFS104+[x+'_'+v for x in FITS104 for v in ['F','C','FC']]
def patch104(m):return {n:p.detach().cpu().numpy().copy() for n,p in m.late.named_parameters()}
def put104(m,s):
 with torch.no_grad():
  for n,p in m.late.named_parameters():p.copy_(torch.as_tensor(s[n],device=p.device))
def features104(m,z,d,pm,im,ref):
 with torch.no_grad():
  _,I,_=m.features(z['vis0'][0],d,'bidirectional',point_mask=pm,image_mask=im);ids=z['query_points_sample_inds'].long();x=m.point_image(I,d)[ids];known=(d['p2i'][ids]>=0).any(-1);gate=torch.tanh(m.gate_late.detach());late=gate*m.late(x.float())*known[:,None];assert torch.equal(late,ref['late']),'source late input does not replay immutable native feature';qb=m.match(ref['h'][None])[0]
 return {**ref,'rgb':x.detach(),'known':known,'gate':gate,'query_raw':qb.detach()}
def raw104(m,f):return f['query_raw']+f['gate']*m.late(f['rgb'].float())*f['known'][:,None]
def scores104(m,f,r0):return native_scores(F.normalize(raw104(m,f),dim=-1),f['u'],f['mask'].bool()).double()+r0
def read104(m,f,r0):return scores104(m,f,r0).softmax(0)
def components104(m,f,r0):
 a=f['rgb'].double();t=f['gate'].double()*f['known'][:,None];raw=f['query_raw'].double()+t*F.linear(a,m.late.weight.double(),m.late.bias.double());norm=raw.norm(dim=-1,keepdim=True).clamp_min(1e-12);q=raw/norm;u=f['u'].double();mask=f['mask'].bool();v=q@u.T/.07;g=(v[:,mask].softmax(-1)@u[mask]-v[:,~mask].softmax(-1)@u[~mask])/.07;g=(g-q*(g*q).sum(-1,keepdim=True))/norm*t;s=torch.logsumexp(v[:,mask],-1)-torch.logsumexp(v[:,~mask],-1)+r0;return a,g,s
def errors104(m,views,r0,targets):
 p=torch.stack([read104(m,f,r0) for f in views]);e=p.log()-targets;e-=e.mean(1,keepdim=True);return e,p
def lm104(m,views,r0,targets,S,maxiter=10):
 put104(m,S);m.requires_grad_(False);initial=damping=None;records=[];reason='max_iterations';nv=len(views)
 for it in range(maxiter):
  with torch.no_grad():
   aa,gg,_=zip(*(components104(m,f,r0) for f in views));a=torch.cat(aa);g=torch.cat(gg);e,_=errors104(m,views,r0,targets);before=float(e.square().mean()/2)
   if initial is None:initial=before
   if before<=max(1e-12,initial*1e-3):reason='relative_error';break
   G=(g@g.T)*(a@a.T+1);G=G.reshape(nv,256,nv,256);G=G-G.mean(1,keepdim=True)-G.mean(3,keepdim=True)+G.mean((1,3),keepdim=True);G=G.reshape(nv*256,nv*256)
   if damping is None:damping=max(1e-12,.01*float(G.trace())/(nv*256))
   old=patch104(m);accepted=False
   for tr in range(6):
    used=damping;v=torch.linalg.solve(G+used*torch.eye(nv*256,device='cuda',dtype=torch.float64),(-e).flatten());v=v.reshape(nv,256);v-=v.mean(1,keepdim=True);v=v.flatten();dW=(v[:,None]*g).T@a;db=(v[:,None]*g).sum(0);st={'weight':old['weight']+dW.float().cpu().numpy(),'bias':old['bias']+db.float().cpu().numpy()};put104(m,st);dw=torch.as_tensor(st['weight']-old['weight'],device='cuda').double();dbb=torch.as_tensor(st['bias']-old['bias'],device='cuda').double();pred=(g*F.linear(a,dw,dbb)).sum(-1).reshape(nv,256);pred-=pred.mean(1,keepdim=True);predred=float((e.square().mean()-(e+pred).square().mean())/2);ee,_=errors104(m,views,r0,targets);after=float(ee.square().mean()/2);rho=(before-after)/max(predred,1e-30);ok=bool(predred>0 and np.isfinite(after) and after<before and rho>.1);records.append({'iteration':it+1,'trial':tr+1,'before':before,'after':after,'predicted_reduction':predred,'rho':rho,'damping':used,'accepted':ok,'step_L2':float(np.sqrt(sum(np.square((st[k]-old[k]).astype(float)).sum() for k in st)))})
    if ok:
     accepted=True
     if rho>.75:damping=max(1e-15,damping/3)
     elif rho<.25:damping*=2
     break
    put104(m,old);damping*=10
   if not accepted:reason='no_accepted_step';break
 with torch.no_grad():e,_=errors104(m,views,r0,targets);after=float(e.square().mean()/2)
 st=patch104(m);return st,records,{'initial_error':initial if initial is not None else after,'final_error':after,'relative_error':after/max(initial or 0.,1e-30),'stop_reason':reason,'accepted_steps':sum(x['accepted'] for x in records),'attempted_steps':len(records),'parameter_L2':float(np.sqrt(sum(np.square((st[k]-S[k]).astype(float)).sum() for k in st))),'fit_views':nv,'parameters':4160}
def check104(m,views,r0,S):
 put104(m,S);aa,gg,_=zip(*(components104(m,f,r0) for f in views));a=torch.cat(aa);g=torch.cat(gg);nv=len(views);J=torch.cat([(g[:,:,None]*a[:,None,:]).reshape(nv*256,-1),g],1).reshape(nv,256,-1);J-=J.mean(1,keepdim=True);J=J.reshape(nv*256,-1);G=(g@g.T)*(a@a.T+1);G=G.reshape(nv,256,nv,256);G=G-G.mean(1,keepdim=True)-G.mean(3,keepdim=True)+G.mean((1,3),keepdim=True);err=float(abs(G.reshape(nv*256,nv*256)-J@J.T).max());assert err<1e-7;W=m.late.weight.detach().double().requires_grad_(True);b=m.late.bias.detach().double().requires_grad_(True);maximum=0.
 for ix,f in enumerate(views):
  raw=f['query_raw'].double()+f['gate'].double()*F.linear(aa[ix],W,b)*f['known'][:,None];s=native_scores(F.normalize(raw,dim=-1),f['u'].double(),f['mask'].bool())+r0;s-=s.mean()
  for q in [0,128,255]:
   dw,db=torch.autograd.grad(s[q],(W,b),retain_graph=True);maximum=max(maximum,float(abs(torch.cat([dw.flatten(),db])-J[ix*256+q]).max()));assert maximum<1e-9
 return {'explicit_Gram_max_error':err,'autograd_Jacobian_max_error':maximum}
