"""Exact existing R94 evidence + R95 update, transferred to the three source models."""
from pathlib import Path
import sys
R98=Path(__file__).resolve().parents[2];H98=R98/'research/round98';B98=R98/'crossplatform/round98'
sys.path.insert(0,str(R98/'research/round97'))
from core97 import *
sys.path.insert(0,str(R98/'research/round95'))
from core95 import *
REFERENCES98=['original_source','prepared_frozen','R97_direct_mean3','R97_direct_feature3','R97_physical_bt','binding_direct','binding_mixture','R94_Adam','R94_GN']
DIRECT98=['frozen_C','frozen_FC','frozen_four','direct_C','direct_FC','direct_four','mixture_FC','shuffle_FC']
FITS98=['lm','lm_mixture','lm_shuffle','lm_capped']
ARMS98=REFERENCES98+DIRECT98+[n+'_'+v for n in FITS98 for v in ['F','C','FC']]
def capture98(base,r,src):
 pts,ix,_=sample_points(r)
 if r['platform']=='waymo':
  from utils.transform_waymo import transform_to_front_view
  pts,_=transform_to_front_view(pts,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pts=np.ascontiguousarray(pts,np.float32)
 z,receipt=capture(base,pts,ix,r['caption']);receipt.update(source_platform=src,source_checkpoint_sha256=source97(src)['checkpoint_sha256']);key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();cf=B97/'shared_physical_geometry_PRIVATE'/(r['id']+'_'+key[:20]+'.npz');assert cf.is_file(),'R97 read-only exact physical cache missing';proof=jread(cf.with_suffix('.json'));assert proof['geometry_seed_and_sample_sha256']==key and proof['image_sha256']==sha(r['imagepath']) and proof['meta_sha256']==sha(r['metapath']);g=load(cf);g['query_anchor_indices']=np.asarray(z['query_points_sample_inds'],np.int64);return z,g,receipt
def gn_reference98(m,f,r0,target,Shead):
 put94(m,Shead);enable94(m);m.zero_grad(set_to_none=True);delta,meta=proposal_solve(m,f,r0,target,'gn');L,p=score_loss_solve(m,f,r0,target);L.backward();gg={n:p.grad.detach().cpu().numpy().copy() for n,p in m.match[-1].named_parameters()};before=float(L.detach());records=[];accepted=False
 for k in range(9):
  scale=2.**(-k);st={n:Shead[n]+scale*delta[n] for n in Shead};put94(m,st)
  with torch.no_grad():loss,p=score_loss_solve(m,f,r0,target)
  bound=before+1e-4*sum(float((gg[n].astype(np.float64)*(st[n]-Shead[n])).sum()) for n in gg);ok=float(loss)<=bound+1e-10;records.append({'scale':scale,'loss':float(loss),'bound':bound,'accepted':bool(ok)})
  if ok:accepted=True;break
 if not accepted:put94(m,Shead)
 with torch.no_grad():p=read94(m,f,r0)
 m.requires_grad_(False);return p.cpu().numpy(),{'before':before,'accepted':accepted,'trials':records,**meta}
def complete_case98(base,m,S,r,z0,g,rt,old):
 restore(m,S);m.eval().requires_grad_(False);z=tensor_prefix(z0);d=tensor_data(g,r);route=tensor_route89(rt);pm,im=maskpair79(r,d);Shead=patch94(m);begin=time.perf_counter();fv=[];parts=[]
 with torch.no_grad():
  for a,b in [(None,None)]+list(zip(pm,im)):
   f=feature91(m,base,z,d,route,S,a,b);fv.append(f);parts.append(parts92(m,f,route,S))
  correct=torch.stack([x[0] for x in parts]);r0=parts[0][1];wrong=[];perms=[]
  for c in range(2):
   perm=permutation94(r,d,c);perms.append(perm);dd=bound94(d,perm);assert torch.equal(dd['p2i'].ge(0).any(1),d['p2i'].ge(0).any(1));ss=[]
   for a,b in [(None,None)]+list(zip(pm,im)):
    f=feature91(m,base,z,dd,route,S,a,b);ss.append(parts92(m,f,route,S)[0])
   wrong.append(torch.stack(ss))
  wrong=torch.stack(wrong);targets,p0,delta=targets94(correct,wrong,r0,r);qp=targets['binding'];mix=(p0+qp)/2;shuf=targets['shuffle'];oldrows=dict(zip(old['arm_names'],old['scores']));assert float(abs(p0-torch.as_tensor(oldrows['prepared_frozen'],device='cuda')).max())<1e-7
  results={k:oldrows[{'R97_direct_mean3':'direct_mean3','R97_direct_feature3':'direct_feature3','R97_physical_bt':'physical_bt'}.get(k,k)] for k in REFERENCES98[:5]};results.update(binding_direct=qp.cpu().numpy(),binding_mixture=mix.cpu().numpy());records={}
 # Legacy finite numerical references, same evidence, never promoted to primary.
 probs,recs,states=optimize94(m,fv[0],r0,qp.detach(),p0.detach(),1,Shead);results['R94_Adam']=probs['bt1'];records['R94_Adam']={'steps':recs};del states
 results['R94_GN'],records['R94_GN']=gn_reference98(m,fv[0],r0,qp.log(),Shead)
 final={}
 for name in FITS98:
  st,states,trials,rec=lm95(m,fv[0],r0,(mix if name=='lm_mixture' else shuf if name=='lm_shuffle' else qp).log(),Shead,name);final[name]=st;records[name]={**rec,'trials':trials};del states
 # C is generated/read only after every fit and acceptance decision is finished.
 restore(m,S);hp,hi=masks(r,d,76003)
 with torch.no_grad():
  fc=feature91(m,base,z,d,route,S,hp,hi);put94(m,Shead);pC=read94(m,fc,r0);qC=(pC.log()+delta).softmax(0);q4=torch.stack([(b+r0+delta).softmax(0) for b in correct]+[qC]).mean(0);p4=torch.stack([(b+r0).softmax(0) for b in correct]+[pC]).mean(0);mixC=(pC+qC)/2;shC=(pC.log()+delta[torch.as_tensor(qperm94(r),device='cuda')]).softmax(0);results.update(frozen_C=pC.cpu().numpy(),frozen_FC=((p0+pC)/2).cpu().numpy(),frozen_four=p4.cpu().numpy(),direct_C=qC.cpu().numpy(),direct_FC=((qp+qC)/2).cpu().numpy(),direct_four=q4.cpu().numpy(),mixture_FC=((mix+mixC)/2).cpu().numpy(),shuffle_FC=((shuf+shC)/2).cpu().numpy())
  for name,st in final.items():
   put94(m,st);pF=read94(m,fv[0],r0);pC1=read94(m,fc,r0);results[name+'_F']=pF.cpu().numpy();results[name+'_C']=pC1.cpu().numpy();results[name+'_FC']=((pF+pC1)/2).cpu().numpy()
 keep=['h','u','late','mask'];features={v+'_'+k:f[k].detach().cpu().numpy() for v,f in [('F',fv[0]),('C',fc)] for k in keep};evidence={'correct':correct.cpu().numpy(),'wrong':wrong.cpu().numpy(),'r0':r0.cpu().numpy(),'delta':delta.cpu().numpy(),'target':qp.cpu().numpy(),'mixture':mix.cpu().numpy(),'shuffle':shuf.cpu().numpy(),'permutations':np.stack(perms)};restore(m,S);m.requires_grad_(False);scores=np.stack([results[a] for a in ARMS98]);assert scores.shape==(29,256) and np.isfinite(scores).all() and np.allclose(scores.sum(1),1,atol=1e-10)
 return scores,records,final,features,evidence,{'seconds':time.perf_counter()-begin,'unknown_roles':len(rt['token_roles'])==0,'visible_seeds':int(d['visible'].sum()),'C_only_after_fits':True,'TTA_parameters':sum(p.numel() for p in head94(m))}
