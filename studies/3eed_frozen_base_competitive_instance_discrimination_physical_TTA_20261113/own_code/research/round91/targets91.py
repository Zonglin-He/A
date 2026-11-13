"""Identical R80 views/scope/direction, prepared frozen heads and matched direct readouts."""
from core91 import *
GRID91=[.5/2**k for k in range(9)]
LEGACY91=['source','R76_frozen','ABC','R59','R80_raw','R82_bt','R89_continue_frozen','R89_continue_oldTTA','R90_joint_frozen','R90_joint_tta','R90_feature3']
NEW91=[b+'_'+h+'_'+a for b in BASES91 for h in HEADS91 for a in ['frozen','mean3','feature3','physical_raw','physical_bt']]+[b+'_full_disc_'+a for b in BASES91 for a in ['conditional_raw','conditional_bt','wrong_physical_raw','wrong_physical_bt']]
def setmatch91(m,s):
 with torch.no_grad():
  for n,p in m.match.named_parameters():p.copy_(torch.as_tensor(s['match.'+n],device=p.device))
def theta_patch91(m):return {n:p.detach().cpu().numpy().copy() for n,p in m.tta_parameters()}
def puttheta91(m,s):
 assert set(s)=={n for n,p in m.tta_parameters()}
 with torch.no_grad():
  for n,p in m.tta_parameters():p.copy_(torch.as_tensor(s[n],device=p.device))
def theta_step91(m,S,d):
 off=0
 with torch.no_grad():
  for n,p in m.tta_parameters():k=p.numel();p.copy_(torch.as_tensor(S[n],device=p.device)+d[off:off+k].reshape_as(p));off+=k
 assert off==len(d)
def feature91(m,base,z,d,route,S,a=None,b=None):
 setmatch91(m,S);m.route=route;return native_feature91(m,base,z,d,a,b)
def scorer91(m,a,route,source_state,b,h):
 setmatch91(m,source_state);copy_head91(m,source_state);return cached91(m,a,route,h)
def mean_reads91(m,views,route,source_state,b,h):
 ps=[];qq=[];uu=[];ins=[];ee=[]
 for a in views:
  hook=m.interface.out[0].register_forward_pre_hook(lambda _m,args:ins.append(args[0]))
  try:p,e,_=scorer91(m,a,route,source_state,b,h)
  finally:hook.remove()
  ps.append(p);qq.append(e['projected_query']);uu.append(a['u']);ee.append(e)
 q=F.normalize(torch.stack(qq).mean(0),dim=-1);u=F.normalize(torch.stack(uu).mean(0),dim=-1);s=native_scores(q,u,views[0]['mask'].bool()).double()
 if ins:assert len(ins)==3;s=s+m.interface.out(torch.stack(ins).mean(0)).squeeze(-1).double()
 return ps[0],torch.stack(ps).mean(0),s.softmax(0),ee[0]
def keep91(p,q,rp,rq):return (rp*(rp.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum()+.1*(q-rq).square().sum(-1).mean()/2
def enable_theta91(m):
 freeze91(m)
 for n,p in m.tta_parameters():p.requires_grad_(True)
def physical_direction91(m,base,z,d,S,pm,im,ws,wrong=False):
 restore(m,S);m.route=None;m.use_interface=False;m.variant='full';enable_theta91(m)
 with torch.no_grad():rp,rq,_=m.main(base,z,d,'bidirectional')
 # Exact existing original objective and parameter order; preservation starts at itself.
 g,raw,rec=gradient79(m,base,z,d,rp,rq,pm,im,ws,wrong);return g,raw,rec
def conditional_objective91(m,base,z,d,route,source_state,b,refs,pm,im,ws):
 # Frozen eta, original theta only; all three live suffix paths remain differentiable.
 setmatch91(m,source_state);copy_head91(m,source_state);m.route=route;m.variant='full';m.use_interface=True
 p,q,e=m.main(base,z,d,'bidirectional');rp,rq,rc,rv=refs;keep=keep91(p,q,rp,rq);corr=p.sum()*0
 for a,i in zip(pm,im):
  _,_,ee=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=i);corr+=conditional_loss90(ee['conditional'],rc,rv&ee['conditional_valid'])/2
 aux=uniform_aux91(m,z,d,pm,im,ws);return aux.double()+keep+.1*corr,p,{'physical':aux,'keep':keep,'conditional':.1*corr}
def make_refs91(m,views,route,source_state,b,h):
 p,e,_=scorer91(m,views[0],route,source_state,b,h);return p.detach(),e['projected_query'].detach(),e['conditional'].detach(),e['conditional_valid'].detach()
def legacy91(r):
 a=load(B90/'prediction336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));p=dict(zip(a['arm_names'],a['scores']));x=load(ROOT/'crossplatform/round79/prediction464_attempt001/samples_PRIVATE'/(r['id']+'.npz'));y=dict(zip(x['arm_names'],x['scores']));t=load(ROOT/'crossplatform/round82/dev336_attempt001/samples_PRIVATE'/(r['id']+'.npz')) if (ROOT/'crossplatform/round82/dev336_attempt001/samples_PRIVATE'/(r['id']+'.npz')).exists() else None
 if t is None:t=load(ROOT/'crossplatform/round82/prediction2640_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
 v=dict(zip(t['arm_names'],t['scores']));old=load(B89/'prediction336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));oo=dict(zip(old['arm_names'],old['scores']));return {'source':p['source'],'R76_frozen':p['frozen_bi'],'ABC':p['frozen_ABC'],'R59':p['R59'],'R80_raw':y['uniform_sgd1'],'R82_bt':v['bt_correct'],'R89_continue_frozen':oo['continue_frozen'],'R89_continue_oldTTA':p['R89_continue_tta1'],'R90_joint_frozen':p['candidate_joint_frozen'],'R90_joint_tta':p['candidate_joint_tta1'],'R90_feature3':p['candidate_joint_feature3']}

_LEGACY_STATUS91={}
def legacy_status91(r):
 if not _LEGACY_STATUS91:
  for n,f in [('R90',B90/'prediction336_attempt001/READOUT_STATUS_PRIVATE.json'),('R89',B89/'prediction336_attempt001/READOUT_STATUS_PRIVATE.json'),('R79',ROOT/'crossplatform/round80/prediction2640_attempt001/READOUT_STATUS_PRIVATE.json'),('R82',ROOT/'crossplatform/round82/prediction2640_attempt001/READOUT_STATUS_PRIVATE.json')]:_LEGACY_STATUS91[n]={x['id']:x['arm_statuses'] for x in read(f)}
 mapping={'source':('R90','source'),'R76_frozen':('R90','frozen_bi'),'ABC':('R90','frozen_ABC'),'R59':('R90','R59'),'R80_raw':('R79','uniform_sgd1'),'R82_bt':('R82','bt_correct'),'R89_continue_frozen':('R89','continue_frozen'),'R89_continue_oldTTA':('R90','R89_continue_tta1'),'R90_joint_frozen':('R90','candidate_joint_frozen'),'R90_joint_tta':('R90','candidate_joint_tta1'),'R90_feature3':('R90','candidate_joint_feature3')}
 return {n:_LEGACY_STATUS91[x][r['id']][a].copy() for n,(x,a) in mapping.items()}
