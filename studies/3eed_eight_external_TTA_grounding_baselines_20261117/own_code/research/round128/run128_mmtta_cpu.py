"""MM-TTA norm-state timing repair, before target-GT scoring.

One fast training forward per input advances running statistics. Diagnostic
student readouts restore their BN buffers. Original stats3 attempt is retained.
This CPU implementation does not import the first producer's update function.
"""
from pathlib import Path
import json,hashlib,time,copy,sys,numpy as np,torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round128';B=R/'crossplatform/round128';sys.path.insert(0,str(H));from models128 import PairHead,RGBHead
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:torch.as_tensor(z[k].copy()) for k in z.files}
def forward(ms,a):return ms['pair'](F.normalize(a['joint'].float(),dim=1)),ms['rgb'](a['images'].permute(0,3,1,2).float()/255,a['text'])
def read(p,i,vis):
 l=torch.where(vis[:,None],(p+i)/2,p);return (l[:,1]-l[:,0]).detach().double().softmax(0).numpy()
def bnconfig(m):
 m.eval().requires_grad_(False);ps=[]
 for mod in m.modules():
  if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
   mod.train()
   for p in mod.parameters():p.requires_grad_(True);ps.append(p)
 return ps
def ema(slow,fast):
 with torch.no_grad():
  for (n,a),(nnn,b) in zip(slow.named_modules(),fast.named_modules()):
   assert n==nnn
   if isinstance(a,(nn.BatchNorm1d,nn.BatchNorm2d)):
    for k in ['weight','bias','running_mean','running_var']:getattr(a,k).lerp_(getattr(b,k),.01)
    a.num_batches_tracked.copy_(b.num_batches_tracked)
def diagnostic(ms,a):
 saved={k:{n:b.clone() for n,b in m.named_buffers()} for k,m in ms.items()}
 with torch.no_grad():p,i=forward(ms,a);out=read(p,i,a['visible'])
 for k,m in ms.items():
  for n,b in m.named_buffers():b.copy_(saved[k][n])
 return out
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--independent',action='store_true');args=ap.parse_args()
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);reg=[e for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json') if e['cohort']=='DEV'];seals={}
 for src in ['waymo','drone','quad']:
  sel=rd(B/f'prepared_PRIVATE/{src}/SELECTED.json');models={'pair':PairHead(),'rgb':RGBHead()}
  if args.independent:sel['models']['rgb']=rd(B/f'prepared_PRIVATE/{src}/MMTTA_INDEPENDENT_RGB_SELECTED.json')
  for k,m in models.items():assert sha(R/sel['models'][k]['path'])==sel['models'][k]['sha256'];m.load_state_dict(torch.load(R/sel['models'][k]['path'],weights_only=True,map_location='cpu'));m.eval().requires_grad_(False)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B/('formal_mmtta_independent' if args.independent else 'formal_mmtta_corrected')/name
   for sub in ['samples_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   fast=copy.deepcopy(models);slow=copy.deepcopy(models);bf=copy.deepcopy(models);bs=copy.deepcopy(models);opts=[torch.optim.SGD(bnconfig(m),lr=.001,momentum=.9) for m in fast.values()]
   for m in bf.values():bnconfig(m);m.requires_grad_(False)
   rr=[e for e in reg if e['source']==src and e['target']==tgt]
   for j,e in enumerate(rr):
    tick=time.perf_counter();sid=e['row']['id'];out={};meta={'protected_alias_fallback':e['protected_alias_fallback'],'fast_stat_updates_per_input':1,'diagnostic_buffer_advancement':False,'CPU':True}
    if e['protected_alias_fallback']:
     old=np.load(B/f'formal_fast/{name}/samples_PRIVATE/{sid}.npz');p=old['scores'][list(old['arm_names']).index('source')];boxes=old['source_boxes'];keys=['MMTTA_correct_teacher_pre','MMTTA_correct_teacher_post','MMTTA_correct_student_post','MMTTA_correct_BN_pre','MMTTA_correct_BN_post'];out={k:p.copy() for k in keys}
    else:
     a=load(B/f'regions_PRIVATE/target/{name}/{sid}.npz')
     if args.independent:a['text']=torch.from_numpy(np.load(B/f'independent_text_PRIVATE/{name}/{sid}.npy',allow_pickle=False))
     boxes=a['boxes'].numpy();vis=a['visible'];before={k:{n:p.detach().clone() for n,p in m.named_parameters()} for k,m in fast.items()}
     with torch.no_grad():
      if args.independent:
       zp,zi=forward(models,a);out['MMTTA_independent_frozen_ensemble']=read(zp,zi,vis)
      sp,si=forward(slow,a);out['MMTTA_correct_teacher_pre']=read(sp,si,vis);bp,bi=forward(bs,a);out['MMTTA_correct_BN_pre']=read(bp,bi,vis);forward(bf,a)
      for k in bs:ema(bs[k],bf[k])
      bp,bi=forward(bs,a);out['MMTTA_correct_BN_post']=read(bp,bi,vis)
     p,i=forward(fast,a)
     with torch.no_grad():
      def trust(x,y):
       a=F.kl_div(x.log_softmax(1),y.softmax(1),reduction='none').sum(1).clamp_min(1e-6);b=F.kl_div(y.log_softmax(1),x.softmax(1),reduction='none').sum(1).clamp_min(1e-6);return .5*(a.reciprocal()+b.reciprocal())
      wp=trust(p,sp);wi=trust(i,si)*vis;den=(wp+wi).clamp_min(1e-20);pseudo=((wp/den)[:,None]*(p.detach()+sp)/2+(wi/den)[:,None]*(i.detach()+si)/2).argmax(1);valid=torch.maximum(wp,wi)>=.3
     L=F.cross_entropy(p[valid],pseudo[valid]);iv=valid&vis
     if iv.any():L=L+F.cross_entropy(i[iv],pseudo[iv])
     for opt in opts:opt.zero_grad(set_to_none=True)
     L.backward();gn=float(torch.sqrt(sum((v.grad.detach().double().square().sum() for m in fast.values() for v in m.parameters() if v.grad is not None),torch.zeros((),dtype=torch.float64))))
     for opt in opts:opt.step()
     delta=float(torch.sqrt(sum(((p.detach()-before[k][n]).double().square().sum() for k,m in fast.items() for n,p in m.named_parameters()),torch.zeros((),dtype=torch.float64))))
     for k in slow:ema(slow[k],fast[k])
     out['MMTTA_correct_student_post']=diagnostic(fast,a)
     with torch.no_grad():sp,si=forward(slow,a);out['MMTTA_correct_teacher_post']=read(sp,si,vis)
     assert np.isfinite(delta) and np.isfinite(gn);meta['actual_parameter_step_nonzero']=bool(delta>0);meta.update(loss=float(L.detach()),gradient_L2=gn,parameter_step_L2=delta,pseudo_pairs=int(valid.sum()),RGB_pairs=int(iv.sum()))
    if args.independent and e['protected_alias_fallback']:out['MMTTA_independent_frozen_ensemble']=next(iter(out.values())).copy()
    if args.independent:out={k.replace('MMTTA_correct_','MMTTA_independent_'):v for k,v in out.items()}
    keys=sorted(out);arr=np.stack([out[k] for k in keys]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-8);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(keys),scores=arr,queries=arr.argmax(1),source_boxes=boxes);meta['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),meta)
    if j%32==0:print('R128_MMTTA_CORRECT',name,j+1,flush=True)
   dump(D/'SEALED.json',{'n':len(rr),'target_GT_used':False,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file() and f.name!='SEALED.json'}});seals[name]=sha(D/'SEALED.json')
 dump(H/('MMTTA_INDEPENDENT_MATRIX_SEALED.json' if args.independent else 'MMTTA_CORRECTED_MATRIX_SEALED.json'),{'n':768,'direction_seals':seals,'one_BN_training_forward_per_input':True,'FINAL_access':False})
if __name__=='__main__':main()
