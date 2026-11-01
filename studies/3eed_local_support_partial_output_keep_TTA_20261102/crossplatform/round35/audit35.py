"""Independent NumPy64 replay of sealed six heads and original mask reductions.
No optimizer/autograd/GPU; only A208+DEV64. Never imports fitting implementation.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import argparse,collections,json,numpy as np
from common35 import *
CONFIGS=['R30_U_rows','U_rows_matched_denominator','partial_output_keep']

def normalized(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def project(h,state):
 x=np.asarray(h,np.float64)
 for i in [0,2,4]:
  x=x@state[str(i)+'.weight'].astype(float).T+state[str(i)+'.bias'].astype(float)
  if i!=4:x=np.maximum(x,0)
 return normalized(x)
def lse(x):
 m=x.max(-1);return m+np.log(np.exp(x-m[...,None]).sum(-1))
def replay(z,state):
 q=project(z['h'],state);v=q@z['tokens'].astype(float).T/.07;mask=z['token_mask'];p=np.exp(v-lse(v)[:,None]);noun=p[:,mask].sum(-1);logodds=lse(v[:,mask])-lse(v[:,~mask]);return noun,logodds

def group_readout_local(r,s,p0):
 if r['stage']=='Q':return int(s.argmax())
 groups=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=set();opts=[]
 for ix in groups.values():
  used.update(ix);rep=max(ix,key=lambda i:(float(p0[i]),-i));opts.append((float(s[ix].max()),-rep,rep))
 for i in range(256):
  if i not in used:opts.append((float(s[i]),-i,i))
 return int(max(opts)[2])
def independent_masks(A,fb):
 lookup={r['id']:j for j,r in enumerate(A)};ids={f['id_PRIVATE'] for f in fb};old=np.array([[r['id'] not in ids]*256 for r in A],bool);partial=np.ones((len(A),256),bool)
 for f in fb:partial[lookup[f['id_PRIVATE']],set_to_list(f['positive_group_PRIVATE']+f['negative_group_PRIVATE'])]=False
 return old,partial
def set_to_list(x):return sorted(set(x))
def main():
 plan=read(EXP/'PLAN_LOCK.json');assert plan['configurations']==CONFIGS
 for n,h in plan['implementation_sha256'].items():assert sha(BASE/n)==h
 out=BASE/'partial_keep_attempt001';assert (out/'SEALED.json').exists(),'Wait for complete training seal before audit';verify(out);receipts=read(out/'FIT_RECEIPTS.json');assert len(receipts)==6
 fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');source=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');source={n.removeprefix('image.'):v for n,v in source.items() if n.startswith('image.')};dest=EXP/'CPU_REPLAY_PRIVATE';dest.mkdir(exist_ok=False);headrows=[];predrows=[];pairrows=[]
 for pl in ['drone','quad']:
  A=[r for r in rows('A208') if r['platform']==pl];D=[r for r in rows('DEV64') if r['platform']==pl];Aindex={r['id']:i for i,r in enumerate(A)};ff=[f for f in fb if f['platform']==pl];old,partial=independent_masks(A,ff);savedmask=load_npz(out/'masks_PRIVATE'/(pl+'.npz'));assert np.array_equal(savedmask['old_keep'],old) and np.array_equal(savedmask['partial_keep'],partial);assert len(ff)==plan['preference_denominators'][pl];zs={r['id']:feature(r) for r in A+D};source_z=np.stack([replay(zs[r['id']],source)[1] for r in A]);q0=np.array([int(zs[r['id']]['scores'].argmax()) for r in A]);d0=source_z-source_z[np.arange(len(A)),q0][:,None]
  for name in CONFIGS:
   state=load_npz(out/'states_PRIVATE'/(pl+'_'+name+'.npz'));assert set(state)=={'0.weight','0.bias','2.weight','2.bias','4.weight','4.bias'} and sum(v.size for v in state.values())==184960;rec=next(r for r in receipts if r['platform']==pl and r['name']==name);maxerror=0.;numeric_tie_mismatches=[];ZA=None
   for split,rr in [('A208',A),('DEV64',D)]:
    saved=load_npz(out/'predictions_PRIVATE'/(pl+'_'+name+'_'+split+'.npz'));assert saved['row_ids_PRIVATE'].tolist()==[r['id'] for r in rr];cp=[];cz=[]
    for j,r in enumerate(rr):
     p,z=replay(zs[r['id']],state);cp.append(p);cz.append(z);err=float(abs(p-saved['scores'][j]).max());maxerror=max(maxerror,err);assert err<2e-5,(pl,name,split,r['alias'],err);official=group_readout_local(r,saved['scores'][j],zs[r['id']]['scores']);assert official==int(saved['queries'][j]);cpu=group_readout_local(r,p,zs[r['id']]['scores'])
     if cpu!=official:numeric_tie_mismatches.append({'alias':r['alias'],'official_GPU_query':official,'CPU64_query':cpu,'GPU_score_at_GPU':float(saved['scores'][j,official]),'GPU_score_at_CPU':float(saved['scores'][j,cpu]),'CPU64_score_gap_GPU_minus_CPU':float(p[official]-p[cpu])})
     predrows.append({'platform':pl,'configuration':name,'split':split,'alias':r['alias'],'probability_max_abs_error':err,'official_saved_query_recomputed':True,'CPU64_argmax_same':cpu==official})
    cp=np.stack(cp);cz=np.stack(cz);np.savez_compressed(dest/(pl+'_'+name+'_'+split+'.npz'),row_ids_PRIVATE=saved['row_ids_PRIVATE'],probability_CPU64=cp,exact_logodds_CPU64=cz)
    if split=='A208':ZA=cz
   gaps=np.array([ZA[Aindex[f['id_PRIVATE']],f['positive_group_PRIVATE']].max()-ZA[Aindex[f['id_PRIVATE']],f['negative_group_PRIVATE']].max() for f in ff]);gaperr=float(abs(gaps-np.array(rec['all_original_pair_gaps'])).max());assert gaperr<5e-4,(pl,name,gaperr);pref=float(np.logaddexp(0,-gaps/.1).mean());relative=ZA-ZA[np.arange(len(A)),q0][:,None];square=(relative-d0)**2;den=int(old.sum() if name=='R30_U_rows' else partial.sum());use=partial if name=='partial_output_keep' else old;keep=float(square[use].sum()/den);assert den==rec['keep_normalization_denominator'];assert int(use.sum())==rec['keep_mask_coordinate_count'];assert abs(pref-rec['final_preference_loss'])<5e-4 and abs(keep-rec['final_keep_MSE'])<2e-3,(pl,name,pref,keep,rec['final_preference_loss'],rec['final_keep_MSE'])
   repro=None
   if name=='R30_U_rows':
    ref=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));repro=max(float(abs(state[k]-ref[k]).max()) for k in state);assert repro<1e-6
   for j,f in enumerate(ff):pairrows.append({'platform':pl,'configuration':name,'pair_alias':f['pair_alias'],'feedback_observation_alias':next(r['alias'] for r in A if r['id']==f['id_PRIVATE']),'label':f['label'],'GPU_original_gap':rec['all_original_pair_gaps'][j],'independent_CPU64_gap':float(gaps[j]),'positive_over_negative':bool(rec['all_original_pair_gaps'][j]>0),'no_feedback_relabel_or_delete':True})
   headrows.append({'platform':pl,'configuration':name,'score_vectors_replayed':len(A)+len(D),'complete_256_probability_maxerror':maxerror,'exact_logodds_pair_gap_maxerror':gaperr,'final_pref_GPU':rec['final_preference_loss'],'final_pref_CPU64':pref,'final_keep_GPU':rec['final_keep_MSE'],'final_keep_CPU64':keep,'mask_denominator':den,'mask_coordinates':int(use.sum()),'original_U_coordinate_weight':1/den,'R30_state_replay_maxerror':repro,'CPU64_numeric_argmax_mismatch_count':len(numeric_tie_mismatches),'CPU64_numeric_argmax_mismatches':numeric_tie_mismatches,'official_GPU_predictions_never_replaced':True})
 result={'state':'PASSED','probability_forward_replay_valid':True,'heads':6,'score_vectors_replayed':len(predrows),'pair_gaps_checked':len(pairrows),'R30_control_vectors_checked':sum(h['score_vectors_replayed'] for h in headrows if h['configuration']=='R30_U_rows'),'mask_independently_reconstructed':True,'matched_denominator_reduction_independently_checked':True,'CPU64_argmax_difference_total':sum(h['CPU64_numeric_argmax_mismatch_count'] for h in headrows),'argmax_difference_scope':'Float64 replay is numerical diagnostic only; task scoring uses sealed GPU probability argmax, including saturation ties.','model_training_GPU_autograd_used':False,'no_U_FINAL_inputs_or_GT_access_in_audit':True,'training_seal_sha256':sha(out/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json')}
 dump(EXP/'CPU_HEAD_AUDIT_ROWS.json',headrows);dump(EXP/'CPU_FORWARD_ROWS_PRIVATE.json',predrows);dump(EXP/'CPU_PAIR_GAPS.json',pairrows);dump(EXP/'CPU_AUDIT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
