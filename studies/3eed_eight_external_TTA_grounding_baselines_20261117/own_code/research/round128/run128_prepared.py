from core128 import *
from models128 import PairHead,RGBHead,MaskedRegion,tensor,image_tensor
from memory128 import UniCache
import copy,importlib.util

def prepared(src):
 root=B/'prepared_PRIVATE'/src;sel=jread(root/'SELECTED.json');mods={'pair':PairHead().cuda(),'rgb':RGBHead().cuda(),'mate':MaskedRegion().cuda()}
 for k,m in mods.items():p=R/sel['models'][k]['path'];assert sha(p)==sel['models'][k]['sha256'];m.load_state_dict(torch.load(p,weights_only=True,map_location='cuda'));m.eval().requires_grad_(False)
 return mods
def pairscores(log):return cpu(log[:,1]-log[:,0])
def nnmodel():
 # The author implementation imports unused Open3D and another FPS package.
 # Only those imports are adapted; Point-NN's parameter-free math is unchanged.
 import types
 pkg=types.ModuleType('r128_pointnn');pkg.__path__=[str(B/'official/bftt3d/models')];sys.modules[pkg.__name__]=pkg
 path=B/'official/bftt3d/models/point_nn.py';src=path.read_text().replace('from pointnet2_ops import pointnet2_utils','from ops.teed_pointnet.pointnet2_batch import pointnet2_utils').replace('import open3d as o3d','o3d=None  # Unused in Point_NN.forward; no Open3D dependency')
 module=types.ModuleType('r128_pointnn.point_nn');module.__package__='r128_pointnn';exec(compile(src,str(path),'exec'),module.__dict__);return module.Point_NN(input_points=128,num_stages=3,embed_dim=24,k_neighbors=16,alpha=100,beta=1000).cuda().eval().requires_grad_(False)
def nnfeat(nn,a):
 with torch.no_grad():f=nn(a['crops'].transpose(1,2));return F.normalize(torch.cat([F.normalize(f,dim=1),F.normalize(a['joint'].float(),dim=1)],1),dim=1)
def herding(x,cap):
 mu=x.mean(0);acc=torch.zeros_like(mu);take=[];available=torch.ones(len(x),device='cuda',dtype=torch.bool)
 for i in range(min(cap,len(x))):
  err=((acc[None]+x)/(i+1)-mu).square().sum(1);err[~available]=torch.inf;j=int(err.argmin());take.append(j);available[j]=False;acc+=x[j]
 return x[take]
def prototypes(src,nn):
 mem=B/'prepared_PRIVATE'/src/'pointnn_prototypes.pt'
 if mem.exists():q=torch.load(mem,weights_only=True,map_location='cuda');return q['features'],q['labels']
 xx=[];yy=[];reg=[e for e in jread(H/'REGION_REGISTRY_PRIVATE.json') if e['role']=='source' and e['source']==src and e['split']=='TRAIN512']
 for e in reg:
  a=tensor(load(R/e['path']));y=a['source_IoU'];ids=[]
  for mask in [y>=.25,y<.05]:
   ix=torch.where(mask)[0];ids.extend(ix[torch.linspace(0,len(ix)-1,min(4,len(ix)),device='cuda').long()].tolist() if len(ix) else [])
  if not ids:continue
  mini={k:(v[ids] if k in ['crops','joint'] else v) for k,v in a.items()};xx.append(nnfeat(nn,mini));yy.append((y[ids]>=.25).long())
 x=torch.cat(xx);y=torch.cat(yy);feats=[];labels=[]
 for k in [0,1]:
  subset=x[y==k];assert len(subset)>0;v=herding(subset,min(64,max(1,len(subset)//4)));feats.append(v);labels.append(torch.full((len(v),),k,device='cuda',dtype=torch.long))
 feats=torch.cat(feats);labels=torch.cat(labels);torch.save({'features':feats.cpu(),'labels':labels.cpu()},mem);return feats,labels
def bftt(nn,a,px,py,source_logits):
 from scipy.linalg import eigh
 x=nnfeat(nn,a);s=px.detach().cpu().double().numpy();t=x.detach().cpu().double().numpy();X=np.concatenate([s,t]).T;X/=np.maximum(np.linalg.norm(X,axis=0),1e-20);ns=len(s);nt=len(t);vec=np.r_[np.ones(ns)/ns,-np.ones(nt)/nt];M=np.outer(vec,vec);M/=np.linalg.norm(M);n=X.shape[1];J=np.eye(n)-np.ones((n,n))/n;aa=X@M@X.T+np.eye(X.shape[0]);bb=X@J@X.T;w,V=eigh(aa,bb+np.eye(len(bb))*1e-6,subset_by_index=[0,149]);Z=V.T@X;Z/=np.maximum(np.linalg.norm(Z,axis=0),1e-20);sim=torch.as_tensor(Z[:,ns:].T@Z[:,:ns],device='cuda',dtype=torch.float32);nlog=torch.exp(-205*(1-sim.clamp(-1,1)))@F.one_hot(py,2).float();Hs=entropy(source_logits);Hn=entropy(nlog);ws=Hn/(Hs+Hn).clamp_min(1e-20);fused=(1-ws[:,None])*nlog+ws[:,None]*source_logits
 rawlog=torch.exp(-205*(1-(x@px.T).clamp(-1,1)))@F.one_hot(py,2).float();hr=entropy(rawlog);wr=hr/(Hs+hr).clamp_min(1e-20);rawfused=(1-wr[:,None])*rawlog+wr[:,None]*source_logits
 return pairscores(fused),pairscores(nlog),pairscores(rawfused),{'source_prototypes':len(px),'TCA_dimensions':150,'finite':bool(torch.isfinite(fused).all()),'backprop_steps':0}
def setnorm(m):
 m.eval().requires_grad_(False);ps=[]
 for mod in m.modules():
  if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
   mod.train()
   for p in mod.parameters():p.requires_grad_(True);ps.append(p)
 return ps
def ema_norm(slow,fast):
 with torch.no_grad():
  sf=dict(fast.named_modules())
  for n,m in slow.named_modules():
   if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
    f=sf[n]
    for k in ['weight','bias','running_mean','running_var']:getattr(m,k).lerp_(getattr(f,k),.01)
    m.num_batches_tracked.copy_(f.num_batches_tracked)
def two(mods,a):
 joint=F.normalize(a['joint'].float(),dim=1);p=mods['pair'](joint);i=mods['rgb'](image_tensor(a),a['text']);return p,i
def multimodal(p,i,vis):return torch.where(vis[:,None],(p+i)/2,p)
def mmstep(fast,slow,opts,a):
 vis=a['visible'];p,i=two(fast,a)
 with torch.no_grad():
  sp,si=two(slow,a)
  def reliability(f,s):
   kl1=F.kl_div(f.log_softmax(1),s.softmax(1),reduction='none').sum(1).clamp_min(1e-6);kl2=F.kl_div(s.log_softmax(1),f.softmax(1),reduction='none').sum(1).clamp_min(1e-6);return .5*(kl1.reciprocal()+kl2.reciprocal())
  wp=reliability(p,sp);wi=reliability(i,si)*vis;den=(wp+wi).clamp_min(1e-20);target=((wp/den)[:,None]*(p.detach()+sp)/2+(wi/den)[:,None]*(i.detach()+si)/2).argmax(1);valid=torch.maximum(wp,wi)>=.3
 L=F.cross_entropy(p[valid],target[valid]);iv=valid&vis
 if iv.any():L=L+F.cross_entropy(i[iv],target[iv])
 for op in opts:op.zero_grad(set_to_none=True)
 L.backward();gn=float(torch.sqrt(sum((q.grad.detach().double().square().sum() for m in fast.values() for q in m.parameters() if q.grad is not None),torch.zeros((),device='cuda',dtype=torch.float64))))
 for op in opts:op.step()
 for k in slow:ema_norm(slow[k],fast[k])
 return {'pseudo_pairs':int(valid.sum()),'RGB_pairs':int(iv.sum()),'loss':float(L.detach()),'gradient_L2':gn,'teacher_momentum':.99,'reference_weight_normalization_bug_corrected':True}
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
 strict97();reg=[e for e in jread(R/'research/round122/CACHE_REGISTRY_PRIVATE.json') if e['cohort']=='DEV'];allseals={}
 # Configure existing local CUDA FPS implementation, without loading a source model.
 from core97 import factory96
 factory96.configure({'seed':0,'torch_num_threads':4});nn=nnmodel()
 for src in ['waymo','drone','quad']:
  models=prepared(src);px,py=prototypes(src,nn)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B/('engineering_prepared' if args.engineering else 'formal_prepared')/name
   for s in ['samples_PRIVATE','records_PRIVATE']:(D/s).mkdir(parents=True,exist_ok=True)
   rr=[e for e in reg if e['source']==src and e['target']==tgt]
   if args.engineering:rr=[next(e for e in rr if not e['protected_alias_fallback'])]
   cache=UniCache();badcache=UniCache();fast={k:copy.deepcopy(models[k]) for k in ['pair','rgb']};slow={k:copy.deepcopy(models[k]) for k in ['pair','rgb']};bnfast={k:copy.deepcopy(models[k]) for k in ['pair','rgb']};bnslow={k:copy.deepcopy(models[k]) for k in ['pair','rgb']}
   for mm in bnfast.values():setnorm(mm);mm.requires_grad_(False)
   opts=[torch.optim.SGD(setnorm(m),lr=.001,momentum=.9) for m in fast.values()];mate=models['mate'];mateS=copy.deepcopy(mate.state_dict())
   for j,e in enumerate(rr):
    tick=time.perf_counter();sid=e['row']['id'];rec={'protected_alias_fallback':e['protected_alias_fallback']}
    if e['protected_alias_fallback']:
     old=np.load(R/f'crossplatform/round128/formal_fast/{name}/samples_PRIVATE/{sid}.npz');srcp=old['scores'][list(old['arm_names']).index('source')];keys=['pair_prepared_zero','RGB_prepared_zero','MMTTA_frozen_ensemble','MMTTA_teacher_pre','MMTTA_BN_only','MMTTA_BN_only_post','MMTTA_student_pre','MMTTA_student_post','MMTTA_teacher_post','MATE_prepared_zero','MATE_step1','MATE_step3','UniAdapter_online','UniAdapter_episodic','UniAdapter_shuffled','BFTT3D','BFTT3D_no_TCA','BFTT3D_nonparametric_only'];out={k:srcp.copy() for k in keys};boxes=old['source_boxes']
    else:
     a=tensor(load(B/f'regions_PRIVATE/target/{name}/{sid}.npz'));boxes=a['boxes'].cpu().numpy();out={}
     with torch.no_grad():
      phase=time.perf_counter();p,i=two(models,a);joint=F.normalize(a['joint'].float(),dim=1);out['pair_prepared_zero']=pairscores(p);out['RGB_prepared_zero']=pairscores(torch.where(a['visible'][:,None],i,p));out['MMTTA_frozen_ensemble']=pairscores(multimodal(p,i,a['visible']));sp,si=two(slow,a);out['MMTTA_teacher_pre']=pairscores(multimodal(sp,si,a['visible']));fp,fi=two(fast,a);out['MMTTA_student_pre']=pairscores(multimodal(fp,fi,a['visible']));bp,bi=two(bnslow,a);out['MMTTA_BN_only']=pairscores(multimodal(bp,bi,a['visible']));two(bnfast,a);two(bnfast,a)
      for kk in bnslow:ema_norm(bnslow[kk],bnfast[kk])
      bp,bi=two(bnslow,a);out['MMTTA_BN_only_post']=pairscores(multimodal(bp,bi,a['visible']));feat=joint;prob=p.softmax(1);up,rec['UniAdapter']=cache.forward(feat,prob,models['pair']);out['UniAdapter_online']=cpu(up[:,1].clamp_min(1e-20).log()-up[:,0].clamp_min(1e-20).log());ep,_=UniCache().forward(feat,prob,models['pair']);out['UniAdapter_episodic']=cpu(ep[:,1].clamp_min(1e-20).log()-ep[:,0].clamp_min(1e-20).log());perm=torch.as_tensor(np.random.default_rng(128).permutation(256),device='cuda');up,_=badcache.forward(feat[perm],prob,models['pair'],query_features=feat);out['UniAdapter_shuffled']=cpu(up[:,1].clamp_min(1e-20).log()-up[:,0].clamp_min(1e-20).log());torch.cuda.synchronize();rec['pair_RGB_Uni_seconds']=time.perf_counter()-phase;phase=time.perf_counter();out['BFTT3D'],out['BFTT3D_nonparametric_only'],out['BFTT3D_no_TCA'],rec['BFTT3D']=bftt(nn,a,px,py,p);torch.cuda.synchronize();rec['BFTT3D']['seconds']=time.perf_counter()-phase
     phase=time.perf_counter();rec['MMTTA']=mmstep(fast,slow,opts,a)
     with torch.no_grad():
      fp,fi=two(fast,a);sp,si=two(slow,a);out['MMTTA_student_post']=pairscores(multimodal(fp,fi,a['visible']));out['MMTTA_teacher_post']=pairscores(multimodal(sp,si,a['visible']))
     torch.cuda.synchronize();rec['MMTTA']['seconds']=time.perf_counter()-phase;phase=time.perf_counter();mate.load_state_dict(mateS);mate.eval().requires_grad_(True);op=torch.optim.AdamW(mate.parameters(),lr=5e-5,weight_decay=.05);trace=[]
     with torch.no_grad():p0,_=mate(a['crops'],joint);out['MATE_prepared_zero']=pairscores(p0)
     for step in range(3):
      op.zero_grad(set_to_none=True);torch.manual_seed(128000+j*3+step);_,L=mate(a['crops'],joint,True);L.backward();gn=float(torch.nn.utils.clip_grad_norm_(mate.parameters(),1));op.step();trace.append({'reconstruction':float(L.detach()),'gradient_norm':gn,'selected_parameters':sum(p.numel() for p in mate.parameters()),'gradient_parameters':sum(p.numel() for p in mate.parameters() if p.grad is not None)})
      if step in [0,2]:
       with torch.no_grad():po,_=mate(a['crops'],joint);out['MATE_step'+str(step+1)]=pairscores(po)
     mate.load_state_dict(mateS);mate.eval().requires_grad_(False);torch.cuda.synchronize();rec['MATE']={'seconds':time.perf_counter()-phase,'steps':trace,'mask_ratio':.875,'source_prepared':True}
    keys=sorted(out);arr=np.stack([out[k] for k in keys]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-8);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(keys),scores=arr,queries=arr.argmax(1),source_boxes=boxes);rec['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),rec)
    if j%16==0:print('R128_PREPARED',name,j+1,round(rec['seconds'],3),flush=True)
   jdump(D/'SEALED.json',{'n':len(rr),'target_GT_used':False,'source_preparation':True,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file() and f.name!='SEALED.json'}});allseals[name]=sha(D/'SEALED.json')
 jdump(H/('ENGINEERING_PREPARED.json' if args.engineering else 'PREPARED_MATRIX_SEALED.json'),{'n':6 if args.engineering else 768,'direction_seals':allseals,'FINAL_access':False})
if __name__=='__main__':main()
