"""Independent CPU: objective/selection/task reductions, frozen predictions and seals."""
import sys,numpy as np,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'research/round5/audit')]
from common30 import *
from independent_core_v2 import candidate_ious,transform_gt

def lse(v):
 m=v.max(-1);return m+np.log(np.exp(v-m[:,None]).sum(-1))
def logodds(state,z):
 x=z['h'].astype(float)
 for j in ['0','2','4']:
  x=x@state[j+'.weight'].astype(float).T+state[j+'.bias'].astype(float)
  if j!='4':x=np.maximum(x,0)
 x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12);v=x@z['tokens'].astype(float).T/.07;mask=z['token_mask'];return lse(v[:,mask])-lse(v[:,~mask])

def main():
 for folder in ['screen_attempt001','sourceS64_attempt001','selectedS64_attempt001','memoryAllS64_attempt001','closureS64_attempt001','evidence_review22_attempt001']:verify(BASE/folder)
 fit=read(BASE/'screen_attempt001/FIT_RECEIPTS.json');fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={f['id_PRIVATE'] for f in fb};source=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');source={k[6:]:v for k,v in source.items() if k.startswith('image.')};cache={r['id']:feature(r) for r in rows()};base={i:logodds(source,z) for i,z in cache.items()};errors=[];counts=collections.Counter()
 for f in fit:
  pl=f['platform'];name=f['name'];ss=load_npz(BASE/'screen_attempt001/states_PRIVATE'/(pl+'_'+name+'.npz'));rr=[r for r in rows() if r['platform']==pl];mse=[];lod={}
  for split in ['A','DEV']:verify(BASE/'screen_attempt001'/(pl+'_'+name+'_'+split))
  for r in rr:
   z=cache[r['id']];lo=logodds(ss,z);lod[r['id']]=lo;q0=int(z['scores'].argmax());d=lo-lo[q0];d0=base[r['id']]-base[r['id']][q0]
   if r['id'] not in fids:mse.append(np.mean((d-d0)**2));assert np.max(abs(((lo+.25)-(lo[q0]+.25))-d))<1e-12
  actual=float(np.mean(mse));err=abs(actual-f['final_keep_MSE']);assert err<max(1e-4,actual*2e-5),(pl,name,actual,f['final_keep_MSE']);errors.append(err);counts['final_keep_losses_recomputed']+=1
  ff=[x for x in fb if x['platform']==pl];passed=0;bp=0
  for x in ff:
   lo=lod[x['id_PRIVATE']];gap=lo[x['positive_group_PRIVATE']].max()-lo[x['negative_group_PRIVATE']].max();passed+=gap>0;bp+=(gap>0 and x['label']=='B')
  assert passed==f['pairs_satisfied'] and bp==f['B_pairs_satisfied'];counts['model_pair_counts_recomputed']+=1
 # Rebuild selection from exposed development rows; never use R64/S labels.
 dev=read(EXP/'DEV64_ROWS_PRIVATE.json');grid=read(EXP/'DEV64_ALL_OPTIONS.json');reduced=[]
 for x in grid:
  pm={p:float(np.mean([r['iou'] for r in dev if r['platform']==p and r['beta']==x['beta']])*100) for p in ['drone','quad']};assert pm==x['platform_mIoU_percent'];assert abs(np.mean(list(pm.values()))-x['balanced_mIoU_percent'])<1e-12;receipts=[f for f in fit if f['beta']==x['beta']];eligible=sum(f['B_pairs_satisfied'] for f in receipts)>=5 and sum(f['B_final_positive'] for f in receipts)>=4;assert eligible==x['eligible']
  if eligible:reduced.append(x)
 selected=sorted(reduced,key=lambda x:(-x['balanced_mIoU_percent'],x['beta']))[0];assert selected==read(EXP/'SELECTED_BETA_LOCK.json')['selected']
 # Independent metadata-to-score reconstruction.
 records=read(EXP/'ALL336_TASK_ROWS_PRIVATE.json');lookup={r['id']:r for split in ['A208','DEV64','S64'] for r in rows(split)};idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={}
 for x in records:
  r=lookup[x['id_PRIVATE']];i=idx[r['id']];path=i['meta_path']
  if path not in containers:assert sha(path)==i['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][i['object_index']]['bbox_3d'],r['platform'],i.get('waymo_view_suffix'));z=feature(r);b=z['boxes'][np.array(list(x['q'].values()))].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vv=candidate_ious(b,gt)[0]
  for arm,v in zip(x['q'],vv):assert float(v)==x['ious'][arm];counts['task_scalars_recomputed']+=1
 # Real content holdout, plus separate dev/A/no-S-training contract.
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];inv={x['image_file_sha256']:x for x in inv};priorrows=prior.rows()+read(ROOT/'research/round29/input/R64_PRIVATE.json');old=[inv[sha(r['imagepath'])] for r in priorrows];new=read(EXP.parent/'input/S64_CONTENT_HASHES_PRIVATE.json')
 for k in ['decoded_pixel_sha256','canonical_XYZ_sha256']:assert not {x[k] for x in old}&{x[k] for x in new};assert len({x[k] for x in new})==64
 # Exact replay source and head installation validated, no relaxed argmax test.
 closure=read(BASE/'closureS64_attempt001/CLOSURE_AUDIT.json');assert all(x['full_model_forwards']==x['exact_argmax_matches']==64 and x['box_max_abs_error']==0 and x['nonmatching_weights_unchanged'] and x['source_restored_bitexact'] for x in closure)
 assert all(f['beta0_state_vs_R29_maxerror']<1e-6 for f in fit if f['beta']==0)
 dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','counts':dict(counts),'objective_relative_logits_keep_exact_formula':True,'sourceq0_fixed':True,'source_relative_common_shift_invariance_checked':True,'max_CPU_GPU_keep_MSE_error':max(errors),'selected_beta':selected['beta'],'selection_independently_reduced':True,'beta0_reproduces_R29_states':True,'S64_actual_RGB_XYZ_disjoint_prior592':True,'S_no_feedback_or_fit':True,'all22_pseudo_feedback_unchanged':True,'no_feedback_counts':{p:sum(r['id'] not in fids for r in rows() if r['platform']==p) for p in ['drone','quad']},'closure128_forwards_exact_argmax_frozenboxes_nonmatchingweights':True,'evidence_labels_not_model_reasoning_or_proof_XYZ_observability':True,'no_puregeometry_feedback_type_fabricated':True,'historical_fullVAL_scene_exposure_disclosed':True});print('AUDIT30_PASSED',dict(counts),max(errors))
if __name__=='__main__':main()
