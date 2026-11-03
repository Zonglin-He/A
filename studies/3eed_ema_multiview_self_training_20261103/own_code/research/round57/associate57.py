"""Geometric and raw-point overlap transport, explicit source-neutral unknown mass."""
import argparse,time
from scipy.special import softmax
from common57 import *
sys=None

def affinity(a,b,xyz):
 aa=((abs(xyz[None]-a[:,None,:3])<=a[:,None,3:]/2).all(-1)&(a[:,3:]>0).all(1)[:,None]).astype(np.float32);bb=((abs(xyz[None]-b[:,None,:3])<=b[:,None,3:]/2).all(-1)&(b[:,3:]>0).all(1)[:,None]).astype(np.float32)
 shared=aa@bb.T;union=aa.sum(1)[:,None]+bb.sum(1)[None]-shared;j=np.divide(shared,union,out=np.zeros_like(shared),where=union>0)
 lo=np.maximum(a[:,None,:3]-a[:,None,3:]/2,b[None,:,:3]-b[None,:,3:]/2);hi=np.minimum(a[:,None,:3]+a[:,None,3:]/2,b[None,:,:3]+b[None,:,3:]/2);inter=np.maximum(hi-lo,0).prod(-1);u=np.maximum(a[:,3:],0).prod(-1)[:,None]+np.maximum(b[:,3:],0).prod(-1)[None]-inter;iou=np.divide(inter,u,out=np.zeros_like(inter),where=u>0);w=np.sqrt(np.maximum(j*iou,0));w[shared<4]=0;return w.astype(np.float32)
def z_native(z,prefix=''):
 l=np.einsum('ik,jk->ij',z[prefix+'query'].astype(float),z[prefix+'tokens'].astype(float))/.07;m=z[prefix+'token_mask'];from scipy.special import logsumexp
 return logsumexp(l[:,m],axis=1)-logsumexp(l[:,~m],axis=1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['association_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();rows=read(ROOT/p['manifest']);ab=ROOT/p['AB_root'];cc=ROOT/p['capture_root'];abseal=read(ab/'SEALED.json');cs=read(cc/'SEALED.json');start=time.perf_counter();stats=[]
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(ab/rel)==abseal['files'][rel] and sha(cc/rel)==cs['files'][rel];v=load(ab/rel);c=load(cc/rel);ids=np.unique(np.concatenate([v['A_point_indices'],v['B_point_indices'],c['point_indices']]));path=Path(row['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,dtype=np.float32).reshape(-1,4);xyz=raw[ids,:3].astype(np.float32).copy()
  if row['platform']=='drone':xyz[:,2]+=1.8
  assert not np.array_equal(c['point_indices'],v['A_point_indices']) and not np.array_equal(c['point_indices'],v['B_point_indices']);assert np.array_equal(v['A_token_mask'],c['token_mask']) and np.array_equal(v['A_token_mask'],v['B_token_mask']);base=softmax(z_native(v,'A_'));result={'base':base,'union_raw_indices':ids};rec={}
  for name,b,zz in [('B',v['B_boxes'],z_native(v,'B_')),('C',c['boxes'],z_native(c))]:
   af=affinity(v['A_boxes'].astype(float),b.astype(float),xyz.astype(float));w=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));mass=float((w@softmax(zz)).sum());result[name+'_affinity']=af;result[name+'_W']=w;result[name+'_fixed_mass']=np.array(mass);rec[name+'_known_mass']=mass;rec[name+'_mapped_A_candidates']=int((w.sum(1)>0).sum());rec[name+'_pairs']=int((w>0).sum())
  np.savez_compressed(out/rel,**result);stats.append({'platform':row['platform'],**rec});
  if len(stats)%32==0:print('ASSOC57',len(stats),336,flush=True)
 dump(out/'COVERAGE_PUBLIC_SAFE.json',stats);dump(out/'COSTS.json',{'n':336,'seconds':time.perf_counter()-start});seal(out,'R57_CROSS_SAMPLING_SUPPORT_TRANSPORT_FIXED_BEFORE_ADAPTATION',n=336)
if __name__=='__main__':main()
