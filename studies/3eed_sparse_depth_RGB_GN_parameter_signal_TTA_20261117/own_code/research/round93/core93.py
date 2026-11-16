"""Fixed R91 architecture; cross-mask competitive evidence directly changes base-score updates."""
from pathlib import Path
import sys,itertools
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round92'))
from common92 import *
H93=ROOT/'research/round93';B93=ROOT/'crossplatform/round93';P92=B92/'prediction336_attempt001'
METHODS93=['competition_projected','competition_sum','competition_only','current_top1','reverse_preference','wrong_binding']
REFS93=['source','R76_frozen','ABC','R59','R80_raw','R82_bt','R91_frozen','R91_mean3','R91_feature3','R92_bt','R92_raw_static']
ARMS93=REFS93+['competition_direct','reverse_preference_direct']+[m+'_'+k for m in METHODS93 for k in ['raw','bt']]
def boxiou93(a,b):
 lo=np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2);hi=np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2);x=np.maximum(hi-lo,0).prod();return float(x/max(1e-30,a[3:].prod()+b[3:].prod()-x))
def evidence93(boxes,own,visible,parts):
 scores=np.stack([(b+r).cpu().numpy() for b,r,q,p in parts]);res=np.stack([r.cpu().numpy() for b,r,q,p in parts]);support=own.astype(bool);v=support&visible[None];orders=[np.argsort(-scores[0],kind='stable'),np.argsort(-scores[1:].mean(0),kind='stable')];pool=[]
 for i in range(256):
  for order in orders:
   q=int(order[i]);duplicate=False
   for p in pool:
    jac=(support[q]&support[p]).sum()/max(1,int((support[q]|support[p]).sum()));iou=boxiou93(boxes[q],boxes[p])
    if q==p or iou>=.9 or (iou>=.5 and jac>=.5):duplicate=True;break
   if not duplicate:pool.append(q)
   if len(pool)==4:break
  if len(pool)==4:break
 pairs=[];targets=[];elig=[]
 for i,j in itertools.combinations(pool,2):
  jac=(support[i]&support[j]).sum()/max(1,int((support[i]|support[j]).sum()));iou=boxiou93(boxes[i],boxes[j]);spatial=bool(v[i].sum()>=2 and v[j].sum()>=2 and iou<=.1 and jac<=.1);a,b=scores[1:,i]-scores[1:,j];c,d=res[1:,i]-res[1:,j];supported=bool(spatial and min(abs(a),abs(b),abs(c),abs(d))>1e-8 and np.sign(a)==np.sign(b)==np.sign(c)==np.sign(d));elig.append({'i_PRIVATE':i,'j_PRIVATE':j,'box_IoU':iou,'support_Jaccard':float(jac),'visible_support_i':int(v[i].sum()),'visible_support_j':int(v[j].sum()),'spatial_distinct_hypotheses':spatial,'masked_margin_A':float(a),'masked_margin_B':float(b),'residual_margin_A':float(c),'residual_margin_B':float(d),'accepted_weak_preference':supported})
  if supported:pairs.append((i,j));targets.append((a+b)/2)
 return np.array(pool,np.int64),np.array(pairs,np.int64).reshape(-1,2),np.array(targets,np.float64),elig
def direct93(s0,pairs,target):
 if not len(pairs):return s0.copy()
 E=np.zeros((len(pairs),256));E[np.arange(len(pairs)),pairs[:,0]]=1;E[np.arange(len(pairs)),pairs[:,1]]=-1;out=s0+E.T@np.linalg.pinv(E@E.T,rcond=1e-12)@(target-E@s0);assert np.max(abs(E@out-target))<1e-8;return out
def live93(m,base,z,d,route,r0):
 m.route=route;m.variant='full';m.use_interface=False;p,q,e=m.main(base,z,d,'bidirectional');b=native_scores(q,e['projected_text'],z['token_mask'].bool()).double();return (b+r0).softmax(0),q,b+r0
def gradients93(m,base,z,d,route,S,pm,im,ws,pairs,target,r0):
 gp,raw,info=physical_direction91(m,base,z,d,S,pm,im,ws,False);gw,_,wronginfo=physical_direction91(m,base,z,d,S,pm,im,ws,True);restore(m,S);enable_theta91(m);p,q,s=live93(m,base,z,d,route,r0);ps=[p for n,p in m.tta_parameters()];q0=int(p.argmax());cur=-p[q0].clamp_min(1e-300).log();marg=(s[torch.as_tensor(pairs[:,0],device='cuda')]-s[torch.as_tensor(pairs[:,1],device='cuda')]) if len(pairs) else s.new_zeros(0);t=torch.as_tensor(target,device='cuda');err=marg-t;rev=marg+t
 if len(pairs):
  lc=err.square().mean()/2;lr=rev.square().mean()/2;gc=flat90(torch.autograd.grad(lc,ps,retain_graph=True,allow_unused=True),ps);gr=flat90(torch.autograd.grad(lr,ps,retain_graph=True,allow_unused=True),ps)
 else:gc=gp*0;gr=gp*0
 gcur=flat90(torch.autograd.grad(cur,ps,allow_unused=True),ps)
 return {'physical':gp.detach(),'wrong_physical':gw.detach(),'pair':gc,'reverse_pair':gr,'current':gcur,'margins':marg.detach(),'initial_pair_loss':float(err.detach().square().mean()/2) if len(pairs) else 0.,'initial_reverse_loss':float(rev.detach().square().mean()/2) if len(pairs) else 0.,'initial_current_loss':float(cur.detach()),'current_query_PRIVATE':q0,'physical_before':info['auxiliary_before'],'wrong_physical_before':wronginfo['auxiliary_before'],'original_raw':raw,'initial_p':p.detach(),'initial_q':q.detach()}
def proposals93(g,npairs):
 out={};radius=float(g['original_raw'].norm())
 for name in METHODS93:
  gp=g['wrong_physical'] if name=='wrong_binding' else g['physical'];gc=g['reverse_pair'] if name=='reverse_preference' else g['current'] if name=='current_top1' else g['pair'];active=npairs>0 and float(gc.norm())>1e-12;coef=float(gp.norm()/gc.norm()) if active else 0.;wphys=0. if name=='competition_only' and active else 1.;dot=float(gp.double()@gc.double());conflict=active and dot<0;projected=gp-min(0.,dot)/max(float(gc.double()@gc.double()),1e-30)*gc if active else gp;task=coef*gc
  if not active:direction=gp;initial_gradient=gp
  else:
   direction=task if name=='competition_only' else gp+task if name=='competition_sum' else projected+task;initial_gradient=wphys*gp+task
  norm=float(direction.norm());delta=-radius*direction/norm if norm>1e-12 else torch.zeros_like(direction);out[name]={'delta':delta.detach(),'initial_gradient':initial_gradient.detach(),'coefficient':coef,'physical_weight':wphys,'task_active':bool(active),'conflict':bool(conflict),'gradient_cosine':dot/max(float(gp.norm()*gc.norm()),1e-30),'proposal_norm':float(delta.norm()),'task_gradient_norm':float(gc.norm()),'physical_gradient_norm':float(gp.norm()),'wrong_binding':name=='wrong_binding'}
 return out
def task93(s,p,name,pairs,target,current):
 if name=='current_top1':return -p[current].clamp_min(1e-300).log()
 if not len(pairs):return s.sum()*0
 ii=torch.as_tensor(pairs[:,0],device='cuda');jj=torch.as_tensor(pairs[:,1],device='cuda');t=torch.as_tensor(-target if name=='reverse_preference' else target,device='cuda');return (s[ii]-s[jj]-t).square().mean()/2
def objective93(m,base,z,d,route,r0,refs,pm,im,ws,name,pr,pairs,target,current):
 p,q,s=live93(m,base,z,d,route,r0);physical=uniform_aux91(m,z,d,pm,im,ws,pr['wrong_binding']);task=task93(s,p,name,pairs,target,current);keep=keep91(p,q,refs[0],refs[1]);total=pr['physical_weight']*physical.double()+pr['coefficient']*task+keep;return total,p,{'physical':physical,'task':task,'keep':keep}
