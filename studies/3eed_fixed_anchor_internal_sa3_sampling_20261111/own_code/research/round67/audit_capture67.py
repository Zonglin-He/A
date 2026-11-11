"""Independent full source provenance, explicit sampler and native head replay.
No sampling/capture producer imports, labels, or task scores.
"""
import sys,time,torch
from common67 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points

def pick(module,xyz,feat,key):
 from ops.teed_pointnet.pointnet2_batch import pointnet2_utils as u
 ds=u.furthest_point_sample(xyz.contiguous(),1024)[0].cpu().tolist();first=[]
 for value in ds+list(range(1024)):
  if value not in first:first.append(value)
  if len(first)==256:break
 v=torch.cat((xyz,feat.permute(0,2,1)),dim=2);n=(v*v).sum(2);distance=(n[:,:,None].repeat(1,1,1024)+n[:,None,:].repeat(1,1024,1)-2*torch.matmul(v,v.transpose(1,2))).contiguous();ff=u.furthest_point_sample_with_dist(distance,1024)[0].cpu().tolist();rng=np.random.default_rng(int(hashlib.sha256(('R67random|'+key).encode()).hexdigest()[:16],16));result={}
 for name,order in [('random',rng.permutation(1024).tolist()),('feature',ff)]:
  seq=first.copy()
  for value in order+ds+list(range(1024)):
   if value not in seq:seq.append(value)
   if len(seq)==512:break
  result[name]=np.asarray(seq,np.int64)
 return result

def suffix(model,xyz,feat,ids):
 b=model.backbone_net;centers=xyz[:,ids,:];m=b.SA_modules[2];parts=[]
 for group,mlp in zip(m.groupers,m.mlps):parts.append(mlp(group(xyz,centers,feat)).max(dim=3).values)
 f3=torch.cat(parts,dim=1);f3=m.aggregation_layer(f3);x4,f4,_,_=b.SA_modules[3](centers,f3);out=b.fp1(centers,x4,f3,f4);return b.fp2(xyz,centers,feat,out).transpose(1,2).contiguous()

def readout(model,z,vis):
 t=lambda k:torch.as_tensor(z[k],device='cuda');V,L=model.cross_encoder(vis_feats=vis,pos_feats=t('pos0'),padding_mask=torch.zeros((1,1024),dtype=torch.bool,device='cuda'),text_feats=t('lang0'),text_padding_mask=t('textpad'),end_points={},detected_feats=None,detected_mask=None)
 _,features,_=model.gsample_module(t('xyz'),V.transpose(1,2).contiguous(),t('query_points_sample_inds').int()[None]);q=model.decoder_query_proj(features).transpose(1,2).contiguous()
 for j in range(6):q=model.decoder[j](q,V,L,t('position'+str(j)),None,t('textpad'),detected_feats=None,detected_mask=None)
 return {'h':q[0].cpu().numpy(),'query':torch.nn.functional.normalize(model.contrastive_align_projection_image(q),dim=-1)[0].cpu().numpy(),'tokens':torch.nn.functional.normalize(model.contrastive_align_projection_text(L),dim=-1)[0].cpu().numpy()}

H=ROOT/'research/round67';p=read(H/'experiment/CAPTURE_AUDIT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);cap=ROOT/p['capture_root'];cs=read(cap/'SEALED.json');model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();nc=0;ns=0
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel];allz=load(cap/rel);abcase=read(ROOT/p['AB_root']/'cases_PRIVATE'/(sid+'.json'));seeds={'A':r['seed'],'B':abcase['B_seed'],'C':seed(r)}
 for v in 'ABC':
  z={k[len(v)+1:]:val for k,val in allz.items() if k.startswith(v+'_')};pts,ix,_=sample_points(dict(r,seed=seeds[v]));got={}
  def pre(_m,args):got['xyz']=args[0].detach().clone();got['feat']=args[1].detach().clone()
  hk=model.backbone_net.SA_modules[2].register_forward_pre_hook(pre)
  try:out,_=source_capture(model,pts,ix,r['caption']);nc+=1
  finally:hk.remove()
  for k in ['h','query','tokens','boxes','point_indices','query_points_sample_inds']:assert np.array_equal(out[k],z[k]),('source',v,k)
  assert np.array_equal(got['xyz'].cpu().numpy(),z['SA2_xyz']) and np.array_equal(got['feat'].cpu().numpy(),z['SA2_feat']) and np.array_equal(z['SA2_xyz'],z['xyz'])
  inds=pick(model.backbone_net.SA_modules[2],got['xyz'],got['feat'],sid+'|'+v)
  for name in ['native','random','feature']:
   ids=allz[name+'_'+v+'_SA3_indices'].astype(np.int64)
   if name!='native':assert np.array_equal(ids,inds[name]) and len(np.unique(ids))==512
   vis=suffix(model,got['xyz'],got['feat'],ids);rr=readout(model,z,vis);ns+=1
   if name=='native':assert np.array_equal(vis.cpu().numpy(),z['vis0'])
   for k in ['h','query','tokens']:assert np.array_equal(rr[k],allz[name+'_'+v+'_'+k]),('independent_suffix',name,v,k,float(abs(rr[k]-allz[name+'_'+v+'_'+k]).max()))
 if (i+1)%32==0:print('AUDIT67',i+1,len(rows),nc,ns,flush=True)
assert fingerprint(model)==fp;out=H/'independent_capture_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R67_ALL_FULL_SOURCE_SA2_SAMPLER_AND_FIXED_ANCHOR_SUFFIXES_BITEXACT','n':336,'independent_full_source_forwards':nc,'independent_suffix_replays':ns,'max_feature_error':0.,'all_SA1_SA2_seed_and_anchor_positions_preserved':True,'all_explicit_mixed_indices_exact':True,'source_unchanged':True,'capture_seal_sha256':sha(cap/'SEALED.json'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'AUDIT_RESULT.json',result);print(result)
