"""Source-label capacity intervention at the exact R123 pointUp interface; NOT TTA."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2];H=R/'research/round124';B=R/'crossplatform/round124'
sys.path.insert(0,str(R/'research/round123'));from core123 import *
sys.path.insert(0,str(R/'research/round121'));from core121 import data121,oracleloss121
# Imported historical helpers export generic path constants; restore this round.
R=Path(__file__).resolve().parents[2];H=R/'research/round124';B=R/'crossplatform/round124'
ARMS=['original_source','prepared','zero','oracle1','oracle3','random1','random3']
def pointread(m,base,z,d,late,res):
 m.route=None;m.use_interface=False;m.variant='full';_,_,e=m.main(base,z,d,'bidirectional');h,u=e['query_hidden'],e['projected_text'];q=F.normalize(m.match(h[None])[0]+late,dim=-1);p=(native_scores(q,u,z['token_mask'].bool()).double()+res.double()).softmax(0);return p,h,u
def process(base,m,S,e,engineering=False):
 restore(m,S);S0,radius=start123(m,S);z0,z,d,rt,pm,im,ws,ious=data121(e);sid=e['row']['id'];old=np.load(R/f"crossplatform/round121/formal/{e['source']}/samples_PRIVATE/{sid}.npz");prior=dict(zip(old['arm_names'],old['scores']));fc=np.load(R/f"crossplatform/round121/formal/{e['source']}/features_PRIVATE/{sid}.npz");late=torch.as_tensor(fc['context_zero_late'],device='cuda');res=torch.as_tensor(fc['context_zero_r'],device='cuda');best=torch.as_tensor(np.isclose(ious,ious.max(),rtol=0,atol=1e-8),device='cuda');eligible=bool(ious.max()>0);params=[dict(m.named_parameters())[n] for n in NAMES];states={};grads={};traces=[];ff={'fixed_late':late.cpu().numpy(),'fixed_r':res.cpu().numpy()};scores={k:prior[{'original_source':'source','prepared':'prepared'}[k]].copy() for k in ['original_source','prepared']};fd=None
 with torch.no_grad():
  p,h,u=pointread(m,base,z,d,late,res);assert np.array_equal(h.cpu().numpy(),z0['h']) and np.array_equal(u.cpu().numpy(),z0['tokens']);assert np.max(np.abs(p.cpu().numpy()-prior['context_zero']))<5e-6 and p.argmax().item()==prior['context_zero'].argmax()
 delta=torch.zeros(4896,device='cuda')
 for t in range(3):
  put123(m,S0,delta)
  for param in params:param.requires_grad_(True)
  p,h,u=pointread(m,base,z,d,late,res);loss=oracleloss121(p,best,eligible);gg=torch.autograd.grad(loss,params,allow_unused=True);g=torch.cat([(torch.zeros_like(param) if v is None else v).detach().flatten() for param,v in zip(params,gg)]);assert torch.isfinite(g).all();grads['step'+str(t+1)]=g.clone();step=-radius*g/g.norm() if eligible and g.norm()>0 else g*0
  for param in params:param.requires_grad_(False)
  if engineering and t==0 and eligible and g.norm()>0:
   direction=g/g.norm();eps=radius*.001;lv=[]
   for sign in [-1,1]:
    put123(m,S0,sign*eps*direction)
    with torch.no_grad():pp,_,_=pointread(m,base,z,d,late,res);lv.append(float(oracleloss121(pp,best,eligible)))
   finite=(lv[1]-lv[0])/(2*eps);analytic=float(g.norm());relative=abs(finite-analytic)/max(abs(analytic),1e-12);assert relative<.1,(finite,analytic,relative);fd={'central_difference':finite,'autograd_directional':analytic,'relative_error':relative,'epsilon':eps}
  delta=delta+step;put123(m,S0,delta)
  with torch.no_grad():pp,_,_=pointread(m,base,z,d,late,res);after=float(oracleloss121(pp,best,eligible))
  traces.append({'step':t+1,'loss_before':float(loss.detach()),'loss_after':after,'gradient_L2':float(g.norm()),'step_L2':float(step.norm()),'cumulative_L2':float(delta.norm())})
  if t in [0,2]:states['oracle'+str(t+1)]=delta.clone()
 for t in [1,3]:
  rng=np.random.default_rng(int(hashlib.sha256(('R124|'+e['source']+'|'+sid+'|'+str(t)).encode()).hexdigest()[:16],16));v=torch.as_tensor(rng.normal(size=4896),dtype=torch.float32,device='cuda');states['random'+str(t)]=v/v.norm()*states['oracle'+str(t)].norm()
 rec={'source_GT_diagnostic_NOT_TTA':True,'new_source_preparation':False,'source_frame_group':e['source_frame_group'],'eligible':eligible,'oracle_IoU':float(ious.max()),'radius':radius,'best_ties':int(best.sum()),'traces':traces,'finite_difference':fd,'endpoints':{}}
 for setting,ds in [('zero',torch.zeros(4896,device='cuda'))]+list(states.items()):
  put123(m,S0,ds);freeze91(m)
  with torch.no_grad():
   p,h,u=pointread(m,base,z,d,late,res);scores[setting]=p.cpu().numpy();ff[setting+'_h']=h.cpu().numpy();ff[setting+'_u']=u.cpu().numpy();ff[setting+'_mask']=z['token_mask'].cpu().numpy();now=state(m);changed=[k for k in S0 if not np.array_equal(now[k],S0[k])];assert set(changed)<=set(NAMES);actual=flat123(m);assert torch.equal(actual,ds);rec['endpoints'][setting]={'changed_tensors':len(changed),'parameter_L2':float(actual.double().norm()),'GT_loss':float(oracleloss121(p,best,eligible)),'h_change_L2':float((h-torch.as_tensor(z0['h'],device='cuda')).norm())}
 restore(m,S);freeze91(m);assert all(np.array_equal(state(m)[k],v) for k,v in S.items());return scores,ff,{**{k:v.cpu().numpy() for k,v in states.items()},**{k+'_gradient':v.cpu().numpy() for k,v in grads.items()}},rec
