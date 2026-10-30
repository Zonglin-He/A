"""R9 independent normal11 prediction/cache/trace audit. No target GT API."""
from pathlib import Path
import argparse,collections,json
import numpy as np
from score_normal import sha,read,dump,bounded,prediction,validate_cache

def main(a):
 sp=a.predictions/'SEALED.json';assert sha(sp)==a.expected_seal_sha256;s=read(sp)
 assert s['state']=='sealed_R9_normal_fixedq0_predictions_and_sourcecaches_before_GT' and s['phase']=='normal' and s['n_arms']==11
 assert s['manifest_sha256']==sha(a.manifest) and s['source_checkpoint_sha256']==sha(a.source_checkpoint)
 for key in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert s['strict_FP32_flags'][key] is False
 assert s['strict_FP32_flags']['cudnn_deterministic'] is True
 for name,h in s['files'].items():assert sha(bounded(a.predictions,name))==h
 for name,h in s['program_sha256'].items():assert sha(bounded(a.program_root,name))==h
 assert s['program_sha256_base']=='workspace_root' and s['all_normal_arm_queries_fixed_q0'] and s['all_11arms_and_sourcecache_sealed_before_GT'] and s['grounder_restored_and_CLIP_frozen']
 assert s['source_cache_count']==s['n_samples'] and s['source_cache_files']=={k:v for k,v in s['files'].items() if k.startswith('source_cache/')}
 samples=read(a.manifest);plan=read(a.predictions/'registered_plan.json');assert len(samples)==s['n_samples'] and len({r['id'] for r in samples})==len(samples)
 trace={t['id']:t for t in read(a.predictions/'trace.json')};assert set(trace)=={r['id'] for r in samples}
 counts=collections.defaultdict(collections.Counter);reasons=collections.defaultdict(collections.Counter);maxkernel=0.;maxselected=0.;maxinterval=0.
 for sample in samples:
  sid=sample['id'];d=prediction(a.predictions/(sid+'.npz'));validate_cache(a.predictions,s,sample,d,plan['sentinel']);t=trace[sid];assert t['q_source0']==int(d['q_source'])
  source=d['arm_boxes_raw6'][0];names=d['arm_names'].tolist()
  for mi,mode in enumerate(('rgb_fulltext','pc_only')):
   key=sample['platform']+'|'+mode;mt=t['G_modes'][mode];assert mt['original_q0']==mt['associated_q1']==int(d['q_source']);w=d['pseudo_reliability6'][mi];target=d['pseudo_target6'][mi];rawdelta=d['raw_consensus_center'][mi]-source[:3]
   counts[key]['n']+=1;counts[key]['accepted_any_axis']+=int(w.any());counts[key]['raw_consensus_nonzero']+=int(np.any(rawdelta));counts[key]['actual_float32_pseudo_nonzero']+=int(np.any((target-source)[w>0]!=0));reasons[key][mt['G_diagnostics']['reason']]+=1
   with np.load(a.predictions/('source_cache/'+sid+'.npz'),allow_pickle=False) as cache:
    q=int(d['q_source']);cn=float(np.square(cache['center_features'][0,:,q],dtype=np.float32).sum(dtype=np.float32))+1.;sn=float(np.square(cache['size_features'][0,:,q],dtype=np.float32).sum(dtype=np.float32))+1.
   # CUDA reduction may have small float32 summation differences; verify formula
   # against recorded actual norms and separately validate cached norm agreement.
   tc=mt['feature_norm2_plus1']['center'];ts=mt['feature_norm2_plus1']['size'];assert np.isclose(tc,cn,rtol=2e-6,atol=1e-6) and np.isclose(ts,sn,rtol=2e-6,atol=1e-6)
   assert mt['actual_lr']==.25*.1/max(tc,ts)
   for k in ('1','3'):
    pa=d['arm_boxes_raw6'][names.index(mode+'__param'+k)];ke=d['arm_boxes_raw6'][names.index(mode+'__outputspace'+k)];err=float(abs(pa-ke).max());maxselected=max(maxselected,err);assert err<=5e-5
    tr=mt['steps'].get(k,{}).get('trace',[]);counts[key]['executed_param'+k+'_steps']+=len(tr);changed=bool(tr and any(v>0 for v in tr[-1]['parameter_delta_max_by_name'].values()));counts[key]['nonzero_parameter_update_param'+k]+=changed;counts[key]['selected_change_param'+k]+=int(not np.array_equal(pa,source))
    for z in tr:
     counts[key]['nonzero_gradient_steps_param'+k]+=int(z['parameter_gradient_norm']>0 and z['nonzero_gradient_parameter_tensors']>0);assert np.isfinite(z['loss']) and np.isfinite(z['parameter_gradient_norm']);maxkernel=max(maxkernel,z['output_space_replay_max_abs_error']);assert z['output_space_replay_max_abs_error']<=5e-5
     assert all(v==0 for n,v in z['parameter_delta_max_by_name'].items() if '.size_pred_head.' in n)
    lower=np.minimum(source[:3],target[:3]);upper=np.maximum(source[:3],target[:3]);excess=float(np.maximum(lower-pa[:3],pa[:3]-upper).clip(min=0).max());maxinterval=max(maxinterval,excess);assert excess<=5e-5
 repeat=read(a.predictions/'repeat_restore_control.json');assert all(repeat[k] for k in ['source_repeat_bitexact','source_final_Conv_bridge_bitexact','post_all_modes_grounder_fullstate_restored','frozen_CLIP_fullstate_unchanged'])
 out={'state':'independent_R9_normal11_unlabelled_audit_pass','auditor_sha256':sha(Path(__file__)),'normal_scorer_contract_sha256':sha(Path(__file__).with_name('score_normal.py')),'prediction_seal_sha256':sha(sp),'n':s['n_samples'],'source_cache_count':s['source_cache_count'],'counts':{k:dict(v) for k,v in counts.items()},'reasons':{k:dict(v) for k,v in reasons.items()},'max_allquery_kernel_error_m':maxkernel,'max_selected_param_kernel_error_m':maxselected,'max_target_interval_excess_m':maxinterval,'registered_equivalence_atol_m':5e-5,'source_fullstate_and_CLIP_restore_receipts_checked':True,'cache_arrays_point_index_text_manifestrow_contract_checked':True,'GT_metadata_index_or_target_annotations_accessed':False,'actual_updates_are_engineering_evidence_not_pseudo_correctness':True};assert not a.output.exists();dump(a.output,out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','manifest','program-root','source-checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--expected-seal-sha256',required=True);main(p.parse_args())
