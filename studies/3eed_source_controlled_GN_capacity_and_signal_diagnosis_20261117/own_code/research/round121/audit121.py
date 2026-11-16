"""Separate CPU head replay and component recovery/gradient checks for source diagnostic."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round121';B=R/'crossplatform/round121';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round116'));from audit116 import Fusion91,tensor_data,masks
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
sys.path.insert(0,str(R/'research/round120'));from audit120 import residue

def recovery(m,z,d,pm,im,which,wrong):
 vals=[]
 for a,b in zip(pm,im):
  # Reconstruct the three existing components without importing local_losses/aux121.
  PP,II,I=m.features(z['vis0'][0],d,'bidirectional',point_mask=a,image_mask=b,wrong=wrong)
  if which=='P':
   rgb=m.point_image(II,d);rgb=rgb[d['wrong_permutation']] if wrong else rgb;pred=F.normalize(m.pred_point(PP)+m.pred_point_image(rgb),dim=-1);target=F.normalize(z['vis0'][0,:,:64].detach(),dim=-1);loss=(pred-target).square().sum(-1)/2;valid=a&d['visible']
  elif which=='RGB':loss=F.smooth_l1_loss(m.pred_color(II).sigmoid(),F.avg_pool2d(d['image'],8,8).flatten(2).transpose(1,2)[0],reduction='none').mean(-1);valid=b
  else:
   pred=m.pred_depth(I).squeeze(-1);target=d['depth_target'].clone();valid=d['depth_valid']
   if wrong:ids=torch.nonzero(valid,as_tuple=True)[0];target[ids]=target[ids.roll(1)]
   loss=F.smooth_l1_loss(pred,target,reduction='none')
  vals.append(loss[valid].mean() if valid.any() else loss.sum()*0)
 return torch.stack(vals).mean()

def main():
 torch.set_num_threads(4);reg=rd(H/'SOURCE_CACHE_REGISTRY_PRIVATE.json');mat=rd(B/'MATRIX_SEALED.json');lock=rd(H/'experiment/DESIGN_LOCK.json');assert mat['n']==384 and rd(H/'ENGINEERING.json')['n']==6
 for p,h in lock['files'].items():assert sha(R/p)==h,p
 col=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];n=heads=gnstates=auxchecks=gradchecks=0;maxp=maxr=maxloss=maxgrad=maxdelta=0.;ties=0
 for src in ['waymo','drone','quad']:
  assert sha(R/col[src]['state_path'])==col[src]['state_sha256'];S=dict(np.load(R/col[src]['state_path']));m=Fusion91(nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64))).cpu().eval().requires_grad_(False);m.load_state_dict({k:torch.from_numpy(np.asarray(v).copy()) for k,v in S.items()});names=[('rgb.'+p+'.'+k,v) for p,mod in m.rgb.named_modules() if isinstance(mod,nn.GroupNorm) for k,v in mod.named_parameters(recurse=False)];assert sum(p.numel() for _,p in names)==224;head={k[6:]:v for k,v in S.items() if k.startswith('match.')};radius=.01*float(torch.cat([p.detach().flatten() for _,p in names]).double().norm());D=B/'formal'/src;assert sha(D/'SEALED.json')==mat['source_seals'][src]
  for f,h in rd(D/'SEALED.json')['files'].items():assert sha(D/f)==h
  for ix,e in enumerate(x for x in reg if x['source']==src):
   sid=e['row']['id'];rec=rd(D/'records_PRIVATE'/(sid+'.json'));a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));st=np.load(D/'states_PRIVATE'/(sid+'.npz'));assert np.array_equal(a['queries'],a['scores'].argmax(1));po=dict(zip(a['arm_names'],a['scores']));z0=np.load(R/e['files']['prefix']['path']);geom=dict(np.load(R/e['files']['geometry']['path']));z={'vis0':torch.from_numpy(z0['vis0'])};d=tensor_data(geom,e['row'],device='cpu');mi=[masks(e['row'],d,k) for k in [76001,76002]];pm=[x[0] for x in mi];im=[x[1] for x in mi];assert np.array_equal(ff['native_h'],z0['h']) and np.array_equal(ff['native_u'],z0['tokens'])
   for k in ['context_zero','P','RGB','D','wrong_P','wrong_RGB','wrong_D','random','oracle1','oracle10']:
    delta=torch.zeros(224) if k=='context_zero' else torch.from_numpy(st[k]);off=0
    with torch.no_grad():
     for name,p in names:cnt=p.numel();p.copy_(torch.from_numpy(S[name])+delta[off:off+cnt].reshape_as(p));off+=cnt
    want=rec['endpoints'][k];actualnorm=float(torch.sqrt(sum((p-torch.from_numpy(S[name])).double().square().sum() for name,p in names)));assert abs(actualnorm-want['actual_writeback_L2'])<1e-9
    if k not in ['context_zero','oracle10']:assert abs(float(delta.norm())-(radius if st[k].any() else 0))<1e-6
    if k=='oracle10':assert float(delta.norm())<=10*radius+1e-5
    res=residue(ff[k+'_r_input'],bool(ff[k+'_r_enabled']),S);er=float(abs(res-ff[k+'_r']).max());maxr=max(maxr,er);assert er<5e-6;p=cpu112(ff['native_h'],ff['native_u'],ff['native_mask'],ff[k+'_late'],res,head);er=float(abs(p-po[k]).max());maxp=max(maxp,er);assert er<5e-6
    if p.argmax()!=po[k].argmax():assert po[k].max()-po[k][p.argmax()]<2*er+1e-12;ties+=1
    heads+=1
    with torch.no_grad():
     for kind in ['P','RGB','D']:L=float(recovery(m,z,d,pm,im,kind,False));er=abs(L-want['aux_losses'][kind]);maxloss=max(maxloss,er);assert er<5e-6;auxchecks+=1
    gnstates+=1
   # CPU component gradients at original state, including wrong correspondences.
   with torch.no_grad():
    for name,p in names:p.copy_(torch.from_numpy(S[name]))
   for name,p in names:p.requires_grad_(True)
   for kind in ['P','RGB','D','wrong_P','wrong_RGB','wrong_D']:
    L=recovery(m,z,d,pm,im,kind.removeprefix('wrong_'),kind.startswith('wrong_'));gg=torch.autograd.grad(L,[p for _,p in names],allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for (_,p),v in zip(names,gg)]);want=torch.from_numpy(st[kind+'_gradient']);er=float(abs(g-want).max());maxgrad=max(maxgrad,er);assert er<1e-5;got=-radius*want/want.norm() if want.norm()>0 else want*0;er=float(abs(got-torch.from_numpy(st[kind])).max());maxdelta=max(maxdelta,er);assert er<1e-6;gradchecks+=1
   for name,p in names:p.requires_grad_(False)
   go=torch.from_numpy(st['oracle1_gradient']);expect=-radius*go/go.norm() if go.norm()>0 else go*0;assert float(abs(expect-torch.from_numpy(st['oracle1'])).max())<1e-6
   for v in rec['oracle10_steps']:
    if v['accepted']:last=v['trials'][-1];assert last['accepted'] and last['loss']<=v['before']+1e-10
   assert len(rec['oracle10_steps'])==10;n+=1
   if (ix+1)%16==0:print('AUDIT121',src,ix+1,flush=True)
 assert n==384 and heads==3840 and gradchecks==2304 and gnstates==3840;out={'state':'PASS_ALL_SOURCE_CPU_HEAD_GN_PHYSICAL_COMPONENT_GRADIENTS_AND_PATCHES','n':n,'heads':heads,'GN_states':gnstates,'aux_components':auxchecks,'physical_gradients':gradchecks,'max_probability_error':maxp,'max_residual_error':maxr,'max_physical_loss_error':maxloss,'max_gradient_error':maxgrad,'max_patch_error':maxdelta,'near_ties':ties,'source_GT_used_for_diagnostic_only':True,'target_data_read':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
