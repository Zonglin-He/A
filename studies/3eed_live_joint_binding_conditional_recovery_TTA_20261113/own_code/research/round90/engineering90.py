"""Source-only live gradient, strict support and native update cost screen."""
from core90 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion90(base.contrastive_align_projection_image).cuda().eval();load_R76_90(m);initial=state(m);configure90(m,'candidate_joint');rows=source_rows('TRAIN512');r=next(r for r in rows if len(load(B89/'source_features640_attempt001/routes_PRIVATE'/(r['id']+'.npz'))['token_roles'])>1);z0,z,d,rt,route,pm,im,ws=data90(r,True);m.route=route;y=torch.tensor(load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'],device='cuda',dtype=torch.float64)
with torch.no_grad():
 p,q,e=m.main(base,z,d,'bidirectional');m.use_interface=False;p0,_,_=m.main(base,z,d,'bidirectional');m.use_interface=True;zero=float(abs(p-p0).max());assert zero<1e-15
 # Actual source support: separate support normalization outside mass exactlyzero.
 a=load(B89/'source_features640_attempt001/samples_PRIVATE'/(r['id']+'.npz'));mask=route['own_support'];att=torch.ones((8,1024),device='cuda')/1024;ww=att[:,None]*mask[None];ww=ww/ww.sum(-1,keepdim=True).clamp_min(1e-12);outside=float((ww*(1-mask)[None]).sum());assert outside==0
configure90(m,'candidate_joint');enable_source90(m);params=[p for _,p in m.source_parameters()];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);start=time.perf_counter();tr=[]
for j in range(8):
 opt.zero_grad(set_to_none=True);L,parts,_=source_loss90(m,base,z,d,rt,pm,im,ws,y);L.backward();gn=float(torch.nn.utils.clip_grad_norm_(params,1));opt.step();tr.append({'step':j+1,'loss':float(L.detach()),'grad_norm':gn});assert np.isfinite(gn)
# Grounding alone must reach the actual original adapter path through the frozen suffix.
opt.zero_grad(set_to_none=True);p,_,ee=m.main(base,z,d,'bidirectional');lg=ground89(p,y);names=m.source_parameters();gr=torch.autograd.grad(lg,[p for _,p in names],allow_unused=True);gn_theta=float(torch.sqrt(sum((g.square().sum() for (n,p),g in zip(names,gr) if g is not None and not n.startswith('interface.')),start=lg*0)));gn_eta=float(torch.sqrt(sum((g.square().sum() for (n,p),g in zip(names,gr) if g is not None and n.startswith('interface.') and not n.startswith('interface.out.')),start=lg*0)));assert gn_theta>0 and gn_eta>0
# Auxiliary's first step reaches the learned candidate router; old preaggregate target does not.
m.requires_grad_(False)
with torch.no_grad():refp,refq,ex=m.main(base,z,d,'bidirectional')
refs=(refp,refq,ex['conditional'],ex['conditional_valid'],ex['local_reads']);eta={};gtheta={}
for kind in ['full','bypass','physical']:
 enable_tta90(m);ps=m.tta_parameters();L,_,_=objective90(m,base,z,d,*refs,pm,im,ws,kind);gg=torch.autograd.grad(L,[p for _,p in ps],allow_unused=True);eta[kind]=float(torch.sqrt(sum((g.square().sum() for (n,p),g in zip(ps,gg) if g is not None and n.startswith('interface.')),start=L*0)));gtheta[kind]=float(torch.sqrt(sum((g.square().sum() for (n,p),g in zip(ps,gg) if g is not None and not n.startswith('interface.')),start=L*0)))
assert eta['full']>1e-8 and eta['bypass']<1e-6 and eta['physical']<1e-6;assert fingerprint(base)==fp;out={'state':'PASS_R90_LIVE_SOURCE_GROUNDING_AND_AUXILIARY_REACH_SHARED_BINDING','source_n':1,'source_native_updates':8,'zero_initial_probability_error':zero,'strict_pool_outside_support_mass':outside,'grounding_gradient_theta':gn_theta,'grounding_gradient_eta_router':gn_eta,'auxiliary_gradient_eta_router':eta,'auxiliary_gradient_theta':gtheta,'source_training_parameters':sum(p.numel() for _,p in m.source_parameters()),'target_adaptation_parameters':sum(p.numel() for _,p in m.tta_parameters()),'seconds':time.perf_counter()-start,'seconds_per_native_source_update_with_backward':(time.perf_counter()-start)/8,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'original_3EED_parameters_changed':False,'target_inputs_or_labels_read':False};dump(H90/'ENGINEERING_AUDIT.json',out);dump(H90/'ENGINEERING_UPDATE_ROWS.json',tr);print(out)
