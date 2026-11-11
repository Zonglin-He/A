"""C source and all three fixed random-SA3 repeats only after all AB fits."""
import sys,time,torch
from common69 import *
from method69 import F,suffix,random_indices
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
H=ROOT/'research/round69';p=read(H/'experiment/C_RANDOM_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);assert read(ROOT/p['fit_root']/'SEALED.json')['n']==len(rows);out=ROOT/p['C_random_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();m=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(m);start=time.perf_counter();src=rep=0
for i,r in enumerate(rows):
 rel='samples_PRIVATE/'+r['id']+'.npz';ab=load(ROOT/p['AB_capture_root']/rel);points,ix,_=sample_points(dict(r,seed=cseed(r)));got={}
 def pre(_m,args):got['SA2_xyz']=args[0].detach().clone().cpu().numpy();got['SA2_feat']=args[1].detach().clone().cpu().numpy()
 hk=m.backbone_net.SA_modules[2].register_forward_pre_hook(pre)
 try:c,_=F.capture(m,points,ix,r['caption']);src+=1
 finally:hk.remove()
 c.update(got);payload={k:v for k,v in c.items()}
 for v in 'ABC':
  z=c if v=='C' else {k[2:]:value for k,value in ab.items() if k.startswith(v+'_')};xyz=torch.tensor(z['SA2_xyz'],device='cuda');feat=torch.tensor(z['SA2_feat'],device='cuda');from ops.teed_pointnet.pointnet2_batch import pointnet2_utils as U
  native=U.furthest_point_sample(xyz.contiguous(),1024)[0].cpu().numpy().astype(np.int64);D=list(dict.fromkeys(map(int,native)))
  for j in range(1024):
   if len(D)>=256:break
   if j not in D:D.append(j)
  native=np.r_[np.array(D[:256],np.int64),np.setdiff1d(np.arange(1024),D[:256])]
  payload[v+'_D_order']=native
  for seedno in range(3):
   ids=random_indices(native,r['id']+'|'+v,seedno);vis,_=suffix(m,xyz,feat,ids);y=F.replay(m,z,alternate_vis0=vis.cpu().numpy());rep+=1
   for k in ['h','query','tokens']:payload['s'+str(seedno)+'_'+v+'_'+k]=y[k]
   payload['s'+str(seedno)+'_'+v+'_SA3_indices']=ids
 np.savez_compressed(out/rel,**payload)
 if(i+1)%32==0:print('C_RANDOM69',i+1,len(rows),src,rep,flush=True)
assert fingerprint(m)==fp;dump(out/'COSTS.json',{'n':len(rows),'C_source_forwards':src,'random_suffix_replays':rep,'source_unchanged':True,'all_fits_sealed_before_C':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});seal(out,'R69_C_AND_THREE_RANDOM_REPEATS_NO_NEW_LABELS',n=len(rows));print({'n':len(rows),'source_C':src,'suffixes':rep})
