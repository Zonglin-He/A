"""Ephemeral source-label controlled GN224 diagnosis; NOT target TTA/source training."""
from pathlib import Path
import sys
R121=Path(__file__).resolve().parents[2];H121=R121/'research/round121';B121=R121/'crossplatform/round121'
sys.path.insert(0,str(R121/'research/round120'));from core120 import *
ARMS121=['source','prepared','native','context_zero','P','RGB','D','wrong_P','wrong_RGB','wrong_D','random','oracle1','oracle10']

def data121(e):
 r=e['row'];dd={}
 for k,x in e['files'].items():
  f=R121/x['path'];assert sha(f)==x['sha256'];dd[k]=load(f)
 z0,g=dd['prefix'],dd['geometry'];z=tensor_prefix(z0);d=tensor_data(g,r);rt=tensor_route89(dd['route']);pm,im=maskpair79(r,d);ws=weight_sets(r,d,pm,im,np.zeros(1024),np.zeros(1536))[0]['uniform'];return z0,z,d,rt,pm,im,ws,dd['label']['ious']

def context121(m,base,z,d,rt,S):
 f=feature91(m,base,z,d,rt,S);ins=[];h=m.interface.out[0].register_forward_pre_hook(lambda _m,a:ins.append(a[0]))
 try:p,e,_=scorer91(m,f,rt,S,'R76','full_disc')
 finally:h.remove()
 return f,e,p,ins[0] if ins else torch.zeros((256,265),device='cuda'),bool(ins)

def task121(m,base,z,d,rt,S,raw,u,mask):
 f,e,p,ri,en=context121(m,base,z,d,rt,S);q=F.normalize(raw+f['late'],dim=-1);pp=(native_scores(q,u,mask).double()+e['residual'].double()).softmax(0);return pp,(f,e,p,ri,en)

def grad121(loss,m):
 gg=torch.autograd.grad(loss,[p for _,p in gn116(m)],allow_unused=True);v=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for (_,p),g in zip(gn116(m),gg)]);assert torch.isfinite(v).all();return v

def unit121(g,radius):return -radius*g/g.norm() if float(g.norm())>0 else g*0

def aux121(m,z,d,pm,im,ws,which,wrong=False):
 vals=[]
 for a,b,w in zip(pm,im,ws):
  P,I,D=m.local_losses(z,d,a,b,wrong);v={'P':P,'RGB':I,'D':D}[which];ww=w['I' if which=='RGB' else which]
  if which=='D' and wrong:
   # Same per-cell source-trained depth head; derange measured depth targets only.
   _,_,Io=m.features(z['vis0'][0],d,'bidirectional',point_mask=a,image_mask=b);pred=m.pred_depth(Io).squeeze(-1);target=d['depth_target'].clone();ids=torch.nonzero(d['depth_valid'],as_tuple=True)[0];order=ids.roll(1);target[ids]=target[order];v=F.smooth_l1_loss(pred,target,reduction='none')
  vals.append((ww*v).sum())
 return torch.stack(vals).mean()

def oracleloss121(p,best,eligible):return -p[best].sum().clamp_min(1e-300).log() if eligible else p.sum()*0

def analyze121(base,m,S,e,engineering=False):
 z0,z,d,rt,pm,im,ws,ious=data121(e);restore(m,S);freeze91(m);radius=.01*float(flat116(m).double().norm());eligible=bool(ious.max()>0);best=torch.as_tensor(np.isclose(ious,ious.max(),rtol=0,atol=1e-8),device='cuda');grads={};states={};ff={};records={'source_labels_controlled_diagnostic_only':True,'source_frame_group':e['source_frame_group'],'GT_eligible':eligible,'oracle_IoU':float(ious.max()),'best_ties':int(best.sum()),'radius':radius,'point_targets':[int((a&d['visible']).sum()) for a in pm],'RGB_targets':[int(a.sum()) for a in im],'depth_targets':int(d['depth_valid'].sum()),'GT_optimization_is_NOT_TTA':True};scores={}
 with torch.no_grad():
  with functional112(m,'neither'):native=feature91(m,base,z,d,rt,S)
  nh,nu=native['h'].detach(),native['u'].detach();raw=m.match(nh[None])[0].detach();mask=native['mask'].bool();np0=native_scores(F.normalize(raw,dim=-1),nu,mask).double().softmax(0);src=torch.as_tensor(z0['scores'],device='cuda').double();src=src/src.sum();assert np.array_equal(nh.cpu().numpy(),z0['h']) and np.array_equal(nu.cpu().numpy(),z0['tokens']);err=float(abs(native_scores(F.normalize(raw,dim=-1),nu,mask).sigmoid().cpu().numpy()-z0['scores']).max());assert err<5e-6,err
  p0,c=task121(m,base,z,d,rt,S,raw,nu,mask);scores={'source':src.cpu().numpy(),'native':np0.cpu().numpy(),'prepared':c[2].cpu().numpy(),'context_zero':p0.cpu().numpy()};ff.update(native_h=nh.cpu().numpy(),native_u=nu.cpu().numpy(),native_mask=mask.cpu().numpy());records['native_source_max_probability_error']=err
 # Every proposal is computed from original source, never another arm's state.
 for kind in ['P','RGB','D','wrong_P','wrong_RGB','wrong_D','oracle1']:
  restore(m,S);enable116(m)
  if kind=='oracle1':p,_=task121(m,base,z,d,rt,S,raw,nu,mask);L=oracleloss121(p,best,eligible)
  else:L=aux121(m,z,d,pm,im,ws,kind.removeprefix('wrong_'),kind.startswith('wrong_'))
  g=grad121(L,m);grads[kind]=g;states[kind]=unit121(g,radius);records[kind+'_loss_before']=float(L.detach())
 seed=int(hashlib.sha256(('R121_random_GN|'+e['source']+'|'+e['row']['id']).encode()).hexdigest()[:16],16);g=torch.as_tensor(np.random.default_rng(seed).normal(size=224),device='cuda',dtype=torch.float32);states['random']=radius*g/g.norm()
 # Bounded source-only capacity control, ten equal-radius task-gradient attempts.
 restore(m,S);delta=torch.zeros(224,device='cuda');steps=[]
 for t in range(10):
  put116(m,S,delta);enable116(m);p,_=task121(m,base,z,d,rt,S,raw,nu,mask);L=oracleloss121(p,best,eligible);g=grad121(L,m);step=unit121(g,radius);before=float(L.detach());trial=[];accepted=False;old=delta.clone()
  if eligible and float(g.norm())>0:
   for k in range(9):
    candidate=old+step/(2**k);put116(m,S,candidate);freeze91(m)
    with torch.no_grad():p,_=task121(m,base,z,d,rt,S,raw,nu,mask);now=float(oracleloss121(p,best,eligible))
    ok=now<=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;trial.append({'fraction':2.**(-k),'loss':now,'accepted':ok})
    if ok:delta=candidate.detach();accepted=True;break
  if not accepted:delta=old
  steps.append({'before':before,'gradient_L2':float(g.norm()),'accepted':accepted,'trials':trial,'total_patch_L2':float(delta.norm())})
 states['oracle10']=delta;records['oracle10_steps']=steps;records['endpoints']={};records['gradient_L2']={k:float(v.norm()) for k,v in grads.items()};go=grads['oracle1'];records['gradient_oracle_cosine']={k:(float((v.double()@go.double())/(v.double().norm()*go.double().norm())) if v.norm()>0 and go.norm()>0 else None) for k,v in grads.items()}
 for setting,delta in [('context_zero',None)]+list(states.items()):
  restore(m,S)
  if delta is not None:put116(m,S,delta)
  freeze91(m)
  with torch.no_grad():
   p,c=task121(m,base,z,d,rt,S,raw,nu,mask);f,extra,_,ri,en=c;scores[setting]=p.cpu().numpy();ff[setting+'_late']=f['late'].cpu().numpy();ff[setting+'_r']=extra['residual'].cpu().numpy();ff[setting+'_r_input']=ri.cpu().numpy();ff[setting+'_r_enabled']=np.asarray(en);actual=state(m);changed=[k for k in S if not np.array_equal(actual[k],S[k])];assert set(changed)<={n for n,_ in gn116(m)};actualdelta=torch.cat([(p-torch.as_tensor(S[n],device='cuda')).flatten() for n,p in gn116(m)]);losses={kind:float(aux121(m,z,d,pm,im,ws,kind)) for kind in ['P','RGB','D']};records['endpoints'][setting]={'oracle_loss':float(oracleloss121(p,best,eligible)),'aux_losses':losses,'changed_tensors':len(changed),'actual_writeback_L2':float(actualdelta.double().norm()),'proposal_L2':float(delta.norm()) if delta is not None else 0.,'native_h_u_FIXED':True}
   if engineering:
    with functional112(m,'neither'):nn=feature91(m,base,z,d,rt,S)
    assert torch.equal(nn['h'],nh) and torch.equal(nn['u'],nu);records['endpoints'][setting]['actual_native_h_u_bitexact']=True
 restore(m,S);freeze91(m);assert set(scores)==set(ARMS121) and all(np.array_equal(state(m)[k],v) for k,v in S.items());return scores,ff,{k:v.cpu().numpy() for k,v in states.items()}|{k+'_gradient':v.cpu().numpy() for k,v in grads.items()},records
