"""Two source training inputs verify the full meta path before any source continuation."""
from core78 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
plan=read(H78/'experiment/DESIGN_LOCK.json');pin(plan);pin(read(H78/'experiment/CODE_LOCK.json'));base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/plan['init_state']);m=Fusion78(base.contrastive_align_projection_image).cuda().eval();restore(m,S);rows=[];start=time.perf_counter()
for k,r in enumerate(source_rows('TRAIN512')[:2]):
 _,z,d=prepare_item(r);y=source_y(r);pm,im=maskpair(r,d);m.requires_grad_(True);fast,a,gn=inner78(m,z,d,pm,im,create_graph=True);p1,_,_=main78(m,base,z,d,fast);L=ground78(p1,y);heads=[m.pred_color.weight,m.pred_depth.weight,m.pred_point.weight];gg=torch.autograd.grad(L,heads,allow_unused=True);norms=[0. if g is None else float(g.norm()) for g in gg];assert all(np.isfinite(norms)) and max(norms)>1e-9,('no_auxhead_meta_path',norms)
 fast_stop,a2,_=inner78(m,z,d,pm,im,create_graph=True,detach_gradient=True);pstop,_,_=main78(m,base,z,d,fast_stop);gstop=torch.autograd.grad(ground78(pstop,y),heads,allow_unused=True);sn=[0. if g is None else float(g.norm()) for g in gstop];assert max(sn)==0.,sn
 with torch.no_grad():
  for n,p in m.tta_parameters():p.copy_(fast[n])
  preal,_,_=main78(m,base,z,d);err=float((preal-p1.detach()).abs().max());assert err<1e-10;restore(m,S)
 # Actual differentiable training objective must send gradients through frozen suffix.
 m.zero_grad(set_to_none=True);fast,a,_=inner78(m,z,d,pm,im,create_graph=True);p0,_,_=main78(m,base,z,d);p1,_,_=main78(m,base,z,d,fast);loss=.5*ground78(p0,y)+.5*ground78(p1,y)+a;loss.backward();grad=float(m.image_adapter.up.weight.grad.norm());assert grad>0 and np.isfinite(grad);assert all(p.grad is None and not p.requires_grad for p in base.parameters());rows.append({'row':k+1,'auxhead_ground1_full_meta_gradient_norms':norms,'auxhead_ground1_stopgrad_norms':sn,'functional_vs_real_SGD_max_error':err,'source_TTA_scope':sum(p.numel() for _,p in m.tta_parameters()),'source_outer_scope':sum(p.numel() for p in m.parameters()),'upstream_gradient_through_frozen_suffix':grad})
assert fingerprint(base)==fp;out=H78/'engineering_attempt001';out.mkdir();dump(out/'RESULT.json',{'state':'PASS_R78_TRUE_HIGHER_ORDER_META_PATH_AND_TRAIN_TEST_ONE_STEP_PARITY','rows':rows,'seconds':time.perf_counter()-start,'source_labels_only':True,'target_or_FINAL_read':False});print(read(out/'RESULT.json'))
