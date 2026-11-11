"""Explicit SA3 256 D + 256 joint XYZ/feature FPS, fixed source scales."""
import numpy as np,torch,hashlib

def choices(module,xyz,features,key):
 from ops.teed_pointnet.pointnet2_batch import pointnet2_utils as u
 d=u.furthest_point_sample(xyz.contiguous(),1024)[0].cpu().numpy().astype(np.int64)
 base=list(dict.fromkeys(map(int,d)))[:256]
 for j in range(1024):
  if len(base)==256:break
  if j not in base:base.append(j)
 joint=torch.cat([xyz,features.transpose(1,2)],-1);dist=module.calc_square_dist(joint,joint).contiguous();f=u.furthest_point_sample_with_dist(dist,1024)[0].cpu().numpy().astype(np.int64)
 ss=int(hashlib.sha256(('R67random|'+key).encode()).hexdigest()[:16],16);rand=np.random.default_rng(ss).permutation(1024)
 result={};stats={}
 for mode,ranking in [('random',rand),('feature',f)]:
  ids=base.copy();seen=set(ids);added=0
  for j in ranking:
   if int(j) not in seen:ids.append(int(j));seen.add(int(j));added+=1
   if len(ids)==512:break
  extra=0
  for j in list(d)+list(range(1024)):
   if len(ids)==512:break
   if int(j) not in seen:ids.append(int(j));seen.add(int(j));extra+=1
  result[mode]=np.array(ids,np.int64);assert len(set(ids))==512
  stats[mode]={'sample_count':512,'D_count':256,'secondary_count':added,'fallback_count':extra,'unique_indices':512,'unique_coordinates':len(np.unique(xyz[0,result[mode]].cpu().numpy(),axis=0))}
 assert np.array_equal(result['random'][:256],result['feature'][:256]);return result,stats

def suffix(model,xyz,feat,idx=None):
 b=model.backbone_net
 if idx is None:x3,f3,_,ix=b.SA_modules[2](xyz,feat)
 else:
  ix=torch.as_tensor(idx,device=xyz.device,dtype=torch.int32)[None];centers=xyz.gather(1,ix.long()[...,None].expand(-1,-1,3));x3,f3,_,_=b.SA_modules[2](xyz,feat,ctr_xyz=centers)
 x4,f4,_,_=b.SA_modules[3](x3,f3);f=b.fp1(x3,x4,f3,f4);f=b.fp2(xyz,x3,feat,f);return f.transpose(1,2).contiguous(),ix
