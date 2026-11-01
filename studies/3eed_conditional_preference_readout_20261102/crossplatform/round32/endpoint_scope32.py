"""Label-free support-localized correction and same-information coordinate memory."""
import sys,time,numpy as np,collections
from common32 import *
from coordinate_projection32 import project_group_preferences
EPS=1e-6
KEYS=['noun_interaction','full_literal_caption_interaction'];KS=[1,4]
def normalized(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def stable_logodds(p):
 p=np.asarray(p,float);assert np.isfinite(p).all() and p.min()>-EPS and p.max()<1+EPS
 return np.log(p+EPS)-np.log(1+EPS-p)
def make_key(z,r,kind,tokenizer):
 if kind=='noun_interaction':t=z['tokens'][z['token_mask']].mean(0)
 else:
  off=np.asarray(tokenizer([r['caption']+' . not mentioned'],return_offsets_mapping=True)['offset_mapping'][0]);assert len(off)==len(z['tokens']);mask=(off[:,1]>off[:,0])&(off[:,0]<len(r['caption']));assert mask.any();t=z['tokens'][mask].mean(0)
 return normalized(z['query']*normalized(t)).astype(np.float32)
def group_readout(r,s):
 if r['stage'] not in ['Q','R','S','T','U','F']:
  groups=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=set();options=[];source=feature(r)['scores']
  for inds in groups.values():
   used.update(inds);rep=max(inds,key=lambda i:(float(source[i]),-i));options.append((float(np.max(s[inds])),-rep,rep))
  for i in range(256):
   if i not in used:options.append((float(s[i]),-i,i))
  return int(max(options)[2])
 return int(np.argmax(s))
def parent_scores(r):
 if not hasattr(parent_scores,'cache'):
  parent_scores.cache={}
  for pl in ['drone','quad']:
   p=load_npz(ROOT/'crossplatform/round31/loo19_attempt002/predictions_PRIVATE'/(pl+'_all.npz'))
   parent_scores.cache.update({rid:p['scores'][i] for i,rid in enumerate(p['row_ids_PRIVATE'].tolist())})
 return parent_scores.cache[r['id']]
def build_library(A,key_kind,tokenizer,feedback):
 import torch
 keys=np.stack([make_key(feature(r),r,key_kind,tokenizer) for r in A]);raw=np.zeros((len(A),256),float);flag=np.zeros((len(A),256),np.float32);spec=[]
 for j,r in enumerate(A):
  ff=[x for x in feedback if x['id_PRIVATE']==r['id']];
  for f in ff:flag[j,f['positive_group_PRIVATE']+f['negative_group_PRIVATE']]=1.
  z=feature(r);p=z['scores'];q0=int(p.argmax());ell=stable_logodds(p)
  if ff:
   pr=project_group_preferences(ell,[(x['positive_group_PRIVATE'],x['negative_group_PRIVATE']) for x in ff],margin=.1,q0=q0,representative_scores=p);raw[j]=pr['raw_query_correction'];spec.append({'alias':r['alias'],'constraints':len(ff),'objective':pr['objective_half_group_l2'],'max_KKT_residual':pr['kkt_max_abs_error']})
 return {'K':torch.tensor(keys,device='cuda'),'raw':torch.tensor(raw,device='cuda',dtype=torch.float64),'flag':torch.tensor(flag,device='cuda'),'coordinate_specs':spec}
def neighbors(q,lib):
 import torch
 sim=torch.einsum('qc,anc->qan',torch.tensor(q,device='cuda'),lib['K']);near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);return best,obs,pos

def score_row(r,p0,pT,lib,knn):
 import torch
 q0=int(p0.argmax());z0=stable_logodds(p0);zt=stable_logodds(pT);d0=z0-z0[q0];dt=zt-zt[q0];delta=dt-d0;best,obs,pos=knn;out={};stats={}
 for k in KS:
  g=(lib['flag'][obs[:,:k],pos[:,:k]].max(-1).values>0).cpu().numpy().astype(float);w=((best[:,:k]-1)/.1).exp().double();raw=((lib['raw'][obs[:,:k],pos[:,:k]]*w).sum(-1)/w.sum(-1).clamp_min(1e-12)).cpu().numpy();md=raw-raw[q0];alpha_mean=float(g.mean());alpha_norm=float(np.linalg.norm(g*delta)/max(np.linalg.norm(delta),1e-12));local=d0+g*delta
  # Exact binary endpoints retain official float32 saturation ties.
  local[g==0]=d0[g==0];local[g==1]=dt[g==1]
  out[k]={'scope_TTA':local,'scope_memory':d0+g*md,'global_mean_TTA':d0+alpha_mean*delta,'global_norm_TTA':d0+alpha_norm*delta,'ungated_memory':d0+md}
  stats[k]={'gate_fraction':alpha_mean,'delta_norm':float(np.linalg.norm(delta)),'gated_delta_norm':float(np.linalg.norm(g*delta)),'global_norm_alpha':alpha_norm,'q0_gate':float(g[q0]),'raw_memory_norm':float(np.linalg.norm(md))}
 return out,stats

def main():
 import torch
 from transformers import RobertaTokenizerFast
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;out=BASE/'scope400_endpoint_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();tokenizer=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True);feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');start=time.perf_counter();endpointchecks=0;stats=[];specs=[];rr=[dict(r,split=split) for split in ['A208','DEV64','R64','S64'] for r in rows(split)]
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl]
  for key in KEYS:
   lib=build_library(A,key,tokenizer,feedback);specs.append({'platform':pl,'key':key,'coordinate_specs':lib['coordinate_specs']})
   for i,r in enumerate([r for r in rr if r['platform']==pl]):
    z=feature(r);p0=z['scores'];pT=parent_scores(r);q0=int(p0.argmax());z0=stable_logodds(p0);zt=stable_logodds(pT);assert group_readout(r,z0-z0[q0])==q0;assert group_readout(r,zt-zt[q0])==group_readout(r,pT);endpointchecks+=1;knn=neighbors(make_key(z,r,key,tokenizer),lib);v,st=score_row(r,p0,pT,lib,knn);names=['source','R30_ungated'];qs=[q0,group_readout(r,pT)]
    for k in KS:
     for arm,s in v[k].items():names.append('k'+str(k)+'_'+arm);qs.append(group_readout(r,s))
     stats.append({'alias':r['alias'],'platform':pl,'split':r['split'],'key':key,'k':k,**st[k]})
    np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'_'+key+'.npz'),arm_names=np.array(names),queries=np.array(qs),boxes=z['boxes'][qs]);
   print('SCOPE32',pl,key,'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'PROCESS_STATS.json',stats);dump(out/'COORDINATE_LIBRARY_SPECS.json',specs);seal(out,'R32_endpoint_scope_families_all400_before_current_scores',n=400,families=4,endpoint_checks=endpointchecks,source_and_R30_endpoints_exact=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),new_optimizer_steps=0,new_Qwen_calls=0);print('SCOPE32_SEALED')
if __name__=='__main__':main()
