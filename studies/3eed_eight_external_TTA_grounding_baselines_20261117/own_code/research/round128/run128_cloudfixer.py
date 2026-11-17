"""Author diffusion-guided input optimization, registered to fixed scene anchors.

Object-classification diffusion prior is an additional resource, not 3EED source
knowledge. Whole-scene normalization is a disclosed task adaptation. Rigid pose
is undone before grounding, so image-left/world-frame text does not rotate.
"""
from core128 import *
from types import SimpleNamespace
def diffusion():
 sys.path.insert(0,str(B/'official/cloudfixer'))
 from diffusion.build_model import get_model
 cfg=SimpleNamespace(model='transformer',probabilistic_model='diffusion',diffusion_steps=500,diffusion_noise_schedule='polynomial_2',diffusion_noise_precision=1e-5,diffusion_loss_type='l2');m=get_model(cfg,torch.device('cuda'));p=B/'checkpoints/cloudfixer_modelnet40.pt';assert sha(p)==jread(H/'CLOUDFIXER_WEIGHTS_AUDIT.json')['sha256'];m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);return m.cuda().eval().requires_grad_(False)
def rotation6(v):
 x=F.normalize(v[:,:3],dim=1);z=F.normalize(torch.cross(x,v[:,3:],dim=1),dim=1);y=torch.cross(z,x,dim=1);return torch.stack([x,y,z],dim=-1)
def fix(dm,pc,salt):
 seed=int(hashlib.sha256((salt+'|CloudFixer').encode()).hexdigest()[:12],16);torch.manual_seed(seed);ix=torch.linspace(0,pc.shape[1]-1,1024,device='cuda').long();small=pc[:,ix].detach();center=small.mean(1,keepdim=True);scale=small.std(dim=(1,2),keepdim=True).clamp_min(1e-3);x=(small-center)/scale;dist=torch.cdist(x,x).square();kd=dist.topk(5,largest=False).values.mean(-1)+1e-8;weight=kd.reciprocal();weight/=weight.sum(1,keepdim=True);delta=nn.Parameter(torch.zeros_like(x));rot=nn.Parameter(torch.zeros((1,6),device='cuda'));rbase=torch.tensor([[1.,0,0,0,1,0]],device='cuda');opt=torch.optim.Adamax([{'params':[delta],'lr':.2},{'params':[rot],'lr':.02}],betas=(.9,.999),weight_decay=0.);node=torch.ones((1,1024,1),device='cuda');trace=[];direct=[]
 def schedule(step):return step/6 if step<6 else .05+max(0.,(30-step)/24)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
 for step in range(30):
  opt.zero_grad(set_to_none=True);t=(.02+.1*torch.rand((1,1),device='cuda'));t=(t*500).long().float()/500;gam=dm.inflate_batch_array(dm.gamma(t),x);alpha=dm.alpha(gam,x);sigma=dm.sigma(gam,x);trans=(x+delta)@rotation6(rot+rbase)
  eps=torch.randn_like(x)
  with torch.no_grad():
   _,target=dm(trans*alpha+eps*sigma,phi=True,return_x0_est=True,t=t,node_mask=node);_,unoptimized=dm(x*alpha+eps*sigma,phi=True,return_x0_est=True,t=t,node_mask=node);direct.append(unoptimized.detach())
  distances=torch.cdist(trans,target).square();match=distances.min(2).values.mean()+distances.min(1).values.mean();reg=(delta.square()*weight[:,:,None]).sum(1).mean()*(10*(1-step/30)+step/30);L=match+reg;assert torch.isfinite(L);L.backward();opt.step();sched.step();trace.append({'step':step+1,'matching':float(match.detach()),'regularization':float(reg.detach()),'objective':float(L.detach())})
 # Optimized rigid alignment belongs only to diffusion's canonical frame.
 # Undo it before calibrated grounding; retain the learned local deformation.
 corr=delta.detach()*scale;nearest=torch.cdist(pc,small).argmin(-1);lift=corr[:,nearest[0]];out=pc+lift;perm=torch.as_tensor(np.random.default_rng(seed).permutation(1024),device='cuda');bad=pc+corr[:,perm][:,nearest[0]]
 denoise=torch.stack(direct).mean(0);dcorr=(denoise-x)*scale;direct_pc=pc+dcorr[:,nearest[0]]
 return out,bad,direct_pc,{'trace':trace,'deformation_mean_m':float(lift.norm(dim=-1).mean()),'deformation_max_m':float(lift.norm(dim=-1).max()),'diffusion_forwards':60,'same_noise_frozen_denoising_forwards':30,'backprop_through_diffusion':False,'global_rotation_undone_for_language_frame':True,'normalized_scene_1024':True}
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
 strict97();dm=diffusion();reg=[e for e in jread(R/'research/round122/CACHE_REGISTRY_PRIVATE.json') if e['cohort']=='DEV'];seals={}
 for src in ['waymo','drone','quad']:
  base=base97(src)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B/('engineering_cloudfixer' if args.engineering else 'formal_cloudfixer')/name
   for sub in ['samples_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   rr=[e for e in reg if e['source']==src and e['target']==tgt]
   if args.engineering:rr=[next(e for e in rr if not e['protected_alias_fallback'])]
   for j,e in enumerate(rr):
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();sid=e['row']['id'];fp=D/'samples_PRIVATE'/(sid+'.npz');rp=D/'records_PRIVATE'/(sid+'.json')
    if fp.exists() and rp.exists():continue
    prior=load(B/f'formal_fast/{name}/samples_PRIVATE/{sid}.npz');original=prior['scores'][list(prior['arm_names']).index('source')];out={'source':original}
    if e['protected_alias_fallback']:out.update(CloudFixer=original,CloudFixer_shuffled_deformation=original,CloudFixer_frozen_direct_denoise=original);rec={'protected_alias_fallback':True}
    else:
     z=load(R/e['prefix']);pc=points(e['row'],z);adapted,bad,direct_pc,rec=fix(dm,pc,sid)
     with torch.no_grad():s,rrr=native(base,adapted,e['row'],z);out['CloudFixer']=cpu(s);s,br=native(base,bad,e['row'],z);out['CloudFixer_shuffled_deformation']=cpu(s);s,dr=native(base,direct_pc,e['row'],z);out['CloudFixer_frozen_direct_denoise']=cpu(s)
     rec.update(protected_alias_fallback=False,registration={k:v for k,v in rrr.items() if k.startswith('seed_')})
    keys=sorted(out);arr=np.stack([out[k] for k in keys]);assert np.isfinite(arr).all();np.savez_compressed(fp,arm_names=np.array(keys),scores=arr,queries=arr.argmax(1),source_boxes=prior['source_boxes']);rec['seconds']=time.perf_counter()-tick;rec['peak_GPU_GiB']=torch.cuda.max_memory_allocated()/2**30;jdump(rp,rec)
    if j%8==0:print('R128_CLOUDFIXER',name,j+1,round(rec['seconds'],3),flush=True)
   jdump(D/'SEALED.json',{'n':len(rr),'target_GT_used':False,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file() and f.name!='SEALED.json'}});seals[name]=sha(D/'SEALED.json')
  del base;torch.cuda.empty_cache()
 jdump(H/('ENGINEERING_CLOUDFIXER.json' if args.engineering else 'CLOUDFIXER_MATRIX_SEALED.json'),{'n':6 if args.engineering else 768,'direction_seals':seals,'FINAL_access':False})
if __name__=='__main__':main()
