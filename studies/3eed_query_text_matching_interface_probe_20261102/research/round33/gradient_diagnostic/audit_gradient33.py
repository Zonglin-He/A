"""CPU recomputation of sealed gradient matrices; no model/autograd/labels."""
import sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'crossplatform/round30'))
from common30 import read,dump,verify,sha
BASE=Path(__file__).resolve().parent
out=BASE/'attempt001';verify(out);ss=read(out/'STATE_SUMMARY.json');obs=read(out/'OBSERVATION_ROWS.json');pairs=read(out/'PAIRWISE_ROWS.json');count=0
for s in ss:
 z=np.load(out/'gradients_PRIVATE'/(s['platform']+'_'+s['state']+'.npz'));G=z['gradient_by_observation'].astype(float);gp=z['gradient_pref'].astype(float);gk=z['gradient_keep'].astype(float);norm=np.sqrt((G*G).sum(1));dot=G@G.T;alias=z['aliases'].tolist();ix=np.triu_indices(len(G),1);nonzero=norm[ix[0]]*norm[ix[1]]>0;ds=dot[ix][nonzero];products=(norm[ix[0]]*norm[ix[1]])[nonzero];energy=np.abs(ds).sum();negative=float(np.maximum(-ds,0).sum()/energy) if energy else None
 assert abs(np.linalg.norm(gp)-s['pref_gradient_norm'])<1e-10;assert abs(np.linalg.norm(gk)-s['keep_gradient_norm'])<1e-10
 if len(ds):assert abs(negative-s['all_exact_nonzero']['negative_dot_energy_fraction'])<1e-12;assert abs(ds.sum()/products.sum()-s['all_exact_nonzero']['norm_weighted_cosine'])<1e-12
 rel=np.linalg.norm(G.sum(0)-gp)/max(np.linalg.norm(gp),1e-30);assert abs(rel-s['perobs_sum_vs_original_pref_relative_error'])<1e-12
 if s['state']=='source':assert np.count_nonzero(gk)==0 and s['pref_keep_cosine'] is None
 for row in [r for r in obs if r['platform']==s['platform'] and r['state']==s['state']]:assert abs(row['gradient_norm']-norm[alias.index(row['feedback_alias'])])<1e-10
 for row in [r for r in pairs if r['platform']==s['platform'] and r['state']==s['state']]:
  i=alias.index(row['a']);j=alias.index(row['b']);assert abs(row['dot']-dot[i,j])<1e-9
  if norm[i] and norm[j]:assert abs(row['cosine']-dot[i,j]/(norm[i]*norm[j]))<1e-12
  else:assert row['cosine'] is None
  count+=1
 assert s['parameters_unchanged'] and s['model_state_sha256_before']==s['model_state_sha256_after'] and s['optimizer_calls']==0
 assert s['parameter_grad_buffers_still_None']
dump(BASE/'CPU_AUDIT.json',{'state':'PASSED','sealed_source':sha(out/'SEALED.json'),'state_summaries_recomputed':len(ss),'observation_gradients_checked':len(obs),'pairwise_dots_and_cosines_checked':count,'source_keep_zero_not_compatibility':True,'original_denominator_preserved':True,'no_model_or_GPU_or_training_in_this_audit':True,'no_R_S_T_labels_accessed':True,'tiny_norm_tasks_do_not_justify_PCGrad_from_raw_angle_counts':True});print('CPU_GRADIENT_AUDIT_PASSED',len(ss),len(obs),count)
