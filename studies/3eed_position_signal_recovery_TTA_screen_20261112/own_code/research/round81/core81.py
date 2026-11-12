"""Bounded position x measured-recovery-signal screen with label-free backtracking."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round80'))
from core80 import *
H81=ROOT/'research/round81';B81=ROOT/'crossplatform/round81'
SCOPES=['image_adapter','point_adapter','point_to_image','image_to_point','all']
CONFIGS=[(s,'P') for s in SCOPES]+[(s,'RGB') for s in ['image_adapter','point_adapter','point_to_image','all']]+[('image_adapter','D'),('all','ALL')]
CONFIG_NAMES=[s+'__'+t for s,t in CONFIGS]
REFS81=['source','frozen_bi','frozen_direct3','R80_uniform_raw','frozen_ABC','R59']
def paramscope(n,s):return s=='all' or n.startswith(s+'.') or (s=='point_to_image' and n=='gate_I') or (s=='image_to_point' and n=='gate_P')
def signal81(parts,sig):return sum(parts.values()) if sig=='ALL' else parts[sig]
def target_permutations(r,d,pm,im):
 out=[]
 for view,(p,i) in enumerate(zip(pm,im)):
  one={};valid={'P':p&d['visible'],'RGB':i,'D':d['depth_valid']}
  for kind,n in [('P',1024),('RGB',1536),('D',1536)]:
   seed=int(hashlib.sha256(('R81targetshuffle|'+r['id']+'|'+str(view)+'|'+kind).encode()).hexdigest()[:16],16)
   ids=np.flatnonzero(valid[kind].cpu().numpy());v=np.arange(n);v[ids]=np.random.default_rng(seed).permutation(ids);one[kind]=torch.as_tensor(v,device='cuda')
  out.append(one)
 return out
def parts81(m,z,d,pm,im,ws,permutation=None):
 pp=[];ii=[];dd=[]
 for view,(p,i,w) in enumerate(zip(pm,im,ws)):
  P=z['vis0'][0];PP,II,Ionly=m.features(P,d,'bidirectional',point_mask=p,image_mask=i);rgb=m.point_image(II,d);estimate=F.normalize(m.pred_point(PP)+m.pred_point_image(rgb),dim=-1);target=F.normalize(P[:,:64].detach(),dim=-1);trgb=F.avg_pool2d(d['image'],8,8).flatten(2).transpose(1,2)[0].detach();td=d['depth_target'].detach()
  if permutation is not None:target=target[permutation[view]['P']];trgb=trgb[permutation[view]['RGB']];td=td[permutation[view]['D']]
  lp=(estimate-target).square().sum(-1)/2;li=F.smooth_l1_loss(m.pred_color(II).sigmoid(),trgb,reduction='none').mean(-1);ld=F.smooth_l1_loss(m.pred_depth(Ionly).squeeze(-1),td,reduction='none');pp.append(.1*(w['P']*lp).sum());ii.append(.1*(w['I']*li).sum());dd.append(.01*(w['D']*ld).sum())
 return {'P':torch.stack(pp).mean(),'RGB':torch.stack(ii).mean(),'D':torch.stack(dd).mean()}
def gradients81(m,z,d,pm,im,ws,permutation=None):
 enable79(m);parts=parts81(m,z,d,pm,im,ws,permutation);ps=m.tta_parameters();out={}
 for k,t in enumerate(['P','RGB','D']):
  gg=torch.autograd.grad(parts[t],[v for _,v in ps],retain_graph=k<2,allow_unused=True);out[t]=torch.cat([(torch.zeros_like(v) if a is None else a).detach().flatten() for (_,v),a in zip(ps,gg)])
 return out,{k:float(v.detach()) for k,v in parts.items()}
def scope_gradient(m,gg,scope,sig):
 g=sum(gg.values()) if sig=='ALL' else gg[sig];mask=torch.cat([torch.full((v.numel(),),paramscope(n,scope),device='cuda',dtype=torch.bool) for n,v in m.tta_parameters()]);return g*mask
def holdout81(r,d):
 pm,im=masks(r,d,76003);return [pm],[im],uniform_weights80(d,[pm],[im])
def step81(m,base,z,d,S,refp,refq,r,scope,sig,gg,initial,pm,im,ws,permutation=None):
 g=scope_gradient(m,gg,scope,sig);gn=float(g.norm());before=signal81(initial,sig);attempts=[];p0=refp.cpu().numpy();accepted_p=p0;accepted_delta=torch.zeros_like(g);alpha=0.;keep0=0.;after=before;postaux=before
 if gn>1e-12:
  direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g
  for a in [.5,.25,.125,.0625,.03125,.015625,.0078125]:
   delta=a*direction;set_step(m,S,delta)
   with torch.no_grad():
    pp,q,_=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;parts=parts81(m,z,d,pm,im,ws,permutation);aux=signal81(parts,sig);L=float(aux.double()+keep);bound=before+1e-4*float(g.double()@delta.double());ok=np.isfinite(L) and L<=bound+1e-10;attempts.append({'alpha':a,'objective':L,'bound':bound,'accepted':ok})
   if ok:alpha=a;after=L;postaux=float(aux);keep0=float(keep);accepted_p=pp.cpu().numpy();accepted_delta=delta;break
  if alpha==0:restore(m,S)
 with torch.no_grad():
  actual=torch.cat([(v-torch.tensor(S[n],device='cuda')).flatten() for n,v in m.tta_parameters()]);hpm,him,hws=holdout81(r,d);heldparts=parts81(m,z,d,hpm,him,hws);held=float(signal81(heldparts,sig))
 record={'scope':scope,'signal':sig,'scope_parameters':sum(v.numel() for n,v in m.tta_parameters() if paramscope(n,scope)),'alpha':alpha,'gradient_norm':gn,'objective_before':before,'objective_after':after,'auxiliary_after':postaux,'preservation_after':keep0,'parameter_step_L2':float(actual.norm()),'query_changed':int(accepted_p.argmax())!=int(p0.argmax()),'attempt_count':len(attempts),'accepted':alpha>0,'zero_auxiliary_gradient':gn<=1e-12,'heldout_true_auxiliary_after':held,'shuffled_targets':permutation is not None,'attempts':attempts}
 patch={n:v.detach().cpu().numpy() for n,v in m.tta_parameters()};restore(m,S);return accepted_p,patch,record
def reference80(r):
 old=load(ROOT/'crossplatform/round80/prediction2640_attempt001/samples_PRIVATE'/(r['id']+'.npz'));v=dict(zip(old['arm_names'],old['scores']));return old,{a:v['uniform_sgd1' if a=='R80_uniform_raw' else a] for a in REFS81}
