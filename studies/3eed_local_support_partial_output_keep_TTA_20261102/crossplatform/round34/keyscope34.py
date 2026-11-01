"""R34 single-factor source h288 vs original noun64 scope. Inference only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,argparse,numpy as np
from common34 import *
from coordinate_projection32 import project_group_preferences

def build_libraries(A,feedback):
 libs={};spec=[];flag=np.zeros((len(A),256),np.float32);raw=np.zeros((len(A),256),np.float64)
 for j,r in enumerate(A):
  ff=[f for f in feedback if f['id_PRIVATE']==r['id']];z=feature(r);p=z['scores'];q0=int(p.argmax());ell=relative_logodds(p,q0)
  for f in ff:flag[j,f['positive_group_PRIVATE']+f['negative_group_PRIVATE']]=1
  if ff:
   pr=project_group_preferences(ell,[(f['positive_group_PRIVATE'],f['negative_group_PRIVATE']) for f in ff],margin=MARGIN,q0=q0,representative_scores=p);raw[j]=pr['raw_query_correction'];spec.append({'alias':r['alias'],'constraints':len(ff),'QP_objective':pr['objective_half_group_l2'],'KKT_error':pr['kkt_max_abs_error']})
 for key in KEYS:libs[key]={'keys':np.stack([make_key(feature(r),key) for r in A]),'endpoint':flag.copy(),'raw':raw.copy(),'A_ids':np.array([r['id'] for r in A])}
 return libs,spec

def gpu_neighbors(q,lib):
 import torch
 # Original R32 max + topk(4), then use only first neighbor. Exact ties keep
 # original Torch behavior; do not substitute stable argsort after seeing U.
 Q=torch.tensor(q,device='cuda');K=lib['keys_GPU'];sim=torch.einsum('qc,anc->qan',Q,K);near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);weights=((best-1)/TEMPERATURE).exp().double()
 selected=sim[torch.arange(len(Q),device='cuda'),obs[:,0]];v=selected.topk(2,dim=-1).values
 result={'obs':obs.cpu().numpy(),'pos':pos.cpu().numpy(),'similarity':best.cpu().numpy(),'weights':weights.cpu().numpy(),'nearest_A_row_margin':(best[:,0]-best[:,1]).cpu().numpy(),'chosen_row_query_margin':(v[:,0]-v[:,1]).cpu().numpy(),'nearest_A_row_exact_tie_count':(near==best[:,0,None]).sum(-1).cpu().numpy(),'chosen_row_query_exact_tie_count':(selected==best[:,0,None]).sum(-1).cpu().numpy()}
 return result

def adapted_probabilities(B):
 import torch
 F=torch.nn.functional;m=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64)).cuda().eval().requires_grad_(False)
 pl=B[0]['platform'];ss=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in ss.items()});zz=[feature(r) for r in B];T=max(len(z['tokens']) for z in zz);tokens=np.zeros((len(B),T,64),np.float32);padding=np.ones((len(B),T),bool);mask=np.zeros((len(B),T),bool)
 for j,z in enumerate(zz):tokens[j,:len(z['tokens'])]=z['tokens'];padding[j,:len(z['tokens'])]=False;mask[j,:len(z['tokens'])]=z['token_mask']
 with torch.inference_mode():
  q=F.normalize(m(torch.tensor(np.stack([z['h'] for z in zz]),device='cuda')),dim=-1);v=q@torch.tensor(tokens,device='cuda').transpose(-1,-2)/.07;v=v.masked_fill(torch.tensor(padding,device='cuda')[:,None],-torch.inf);p=(v.softmax(-1)*torch.tensor(mask,device='cuda')[:,None]).sum(-1).cpu().numpy()
 return p

def main():
 import torch
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['diagnostic','U128'],required=True);args=p.parse_args();check_plan();out=output(args.split);out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();(out/'libraries_PRIVATE').mkdir()
 torch.set_num_threads(4);torch.manual_seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if args.split=='U128':verify(ROOT/'crossplatform/round33/sourceU128_attempt001')
 feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(feedback)==22;B=rows(args.split);assert len(B)==(528 if args.split=='diagnostic' else 128);start=time.perf_counter();records=[];libspec=[];baseline=[]
 for pl in ['drone','quad']:
  A=[r for r in rows('A208') if r['platform']==pl];bb=[r for r in B if r['platform']==pl];libs,spec=build_libraries(A,feedback);libspec.extend([dict(platform=pl,**x) for x in spec])
  for key,lib in libs.items():
   np.savez_compressed(out/'libraries_PRIVATE'/(pl+'_'+key+'.npz'),**lib);lib['keys_GPU']=torch.tensor(lib['keys'],device='cuda')
  pp=adapted_probabilities(bb) if args.split=='U128' else np.stack([parent_probability(r) for r in bb])
  for r,pT in zip(bb,pp):
   z=feature(r);p0=z['scores'];knn={key:gpu_neighbors(make_key(z,key),lib) for key,lib in libs.items()};ss,proc=all_scores(p0,pT,libs,knn);q=np.array([group_readout(r,ss[a]) for a in ARMS]);assert q[0]==int(p0.argmax());assert q[1]==group_readout(r,pT)
   payload={'arm_names':np.array(ARMS),'queries':q,'boxes':z['boxes'][q],'complete_scores':np.stack([ss[a] for a in ARMS]),'source_probability':p0,'R30_probability':pT,'source_q0':np.array(int(p0.argmax()))}
   for key in KEYS:
    for k,v in knn[key].items():payload[key+'_'+k]=v
    for k in ['gate','raw_retrieved','relative_memory']:payload[key+'_'+k]=proc[key][k]
   np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),**payload)
   records.append({'alias':r['alias'],'platform':pl,'split':r.get('split',args.split),'gate_stats':{key:{k:v for k,v in proc[key].items() if np.isscalar(v)} for key in KEYS},'precision':{key:{'A_exact_tie_queries':int((knn[key]['nearest_A_row_exact_tie_count']>1).sum()),'candidate_exact_tie_queries':int((knn[key]['chosen_row_query_exact_tie_count']>1).sum()),'A_row_gap_min':float(knn[key]['nearest_A_row_margin'].min()),'candidate_gap_min':float(knn[key]['chosen_row_query_margin'].min()),'A_row_gap_lt5e7':int((knn[key]['nearest_A_row_margin']<5e-7).sum())} for key in KEYS}})
   if args.split=='diagnostic':
    oldpath=ROOT/'crossplatform/round32/scope400_endpoint_attempt001/predictions_PRIVATE'/(r['id']+'_noun_interaction.npz') if r['stage']!='T' else ROOT/'crossplatform/round32/selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz');old=load_npz(oldpath);names=old['arm_names'].tolist();oldname='k1_scope_TTA' if r['stage']!='T' else 'scoped_TTA';oldq=int(old['queries'][names.index(oldname)]);baseline.append({'alias':r['alias'],'platform':pl,'old_R32_noun_query':oldq,'R34_noun_query':int(q[ARMS.index('noun_scoped_R30')]),'exact_match':oldq==q[ARMS.index('noun_scoped_R30')]})
  print('KEYSCOPE34',args.split,pl,len(bb),'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'PROCESS_STATS.json',records);dump(out/'QP_LIBRARY_SPECS.json',libspec);dump(out/'OLD_NOUN_ENDPOINT_COMPARISON.json',baseline)
 seal(out,'R34_fixed_h_key_single_factor_complete_predictions_before_current_labels',split=args.split,n=len(B),all256_fullscore_vectors_saved=True,GPU_nearest_indices_saved=True,GPU_topk4_then_k1=True,**plan_receipt(),optimizer_steps=0,new_G_calls=0,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('KEYSCOPE34_SEALED',args.split,flush=True)
if __name__=='__main__':main()
