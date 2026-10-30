"""Read only SEALED prediction arrays/trace for cost and repeat audit. NO GT reads."""
from pathlib import Path
import argparse,json,hashlib,ast,re,time
import numpy as np
R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def stats(x):
 a=np.asarray(x,float);assert len(a) and np.isfinite(a).all();return {'n':len(a),'sum':float(a.sum()),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'min':float(a.min()),'max':float(a.max())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--smoke',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();began=time.perf_counter();sealpath=a.cache/'sealed.json';assert sealpath.is_file(),'Pending fullseal: no audit of partial source directory.';seal=read(sealpath);ss=read(a.smoke/'sealed.json');manifest=read(a.manifest);assert seal['manifest_sha256']==sha(a.manifest);assert seal['n_samples']==len(manifest)==9706;assert len(seal['files'])==19412
 plan=read(a.cache/'registered_plan.json');assert seal['plan_sha256']==sha(a.cache/'registered_plan.json');codehashes={f:sha(R/f) for f in plan['code_sha256']};assert codehashes==plan['code_sha256'];assert seal['all_parameters_frozen'] and seal['parameter_versions_unchanged'] and seal['buffers_bitexact'] and seal['no_actual_target_class_or_numeric_GT_read']
 expected={s['id']+'_'+c+'.npz' for s in manifest for c in ['full','noun']};assert set(seal['files'])==expected;assert {p.name for p in a.cache.glob('*.npz')}==expected
 for name,h in seal['files'].items():assert re.fullmatch(r'[a-f0-9]{16}_(full|noun)\.npz',name) and sha(a.cache/name)==h,name
 for name,h in ss['files'].items():assert sha(a.smoke/name)==h,name
 for f,k in [('trace.json','trace_sha256'),('point_provenance.json','point_provenance_sha256'),('repeat_control.json','repeat_control_sha256'),('native_readout_bridge.json','native_readout_bridge_sha256')]:assert sha(a.cache/f)==seal[k]
 pairs=[]
 for row in manifest:
  sid=row['id']
  with np.load(a.cache/(sid+'_full.npz'),allow_pickle=False) as full,np.load(a.cache/(sid+'_noun.npz'),allow_pickle=False) as noun:
   for key in ['point_sample_sha256','raw_indices_sha256']:assert full[key].shape==(32,) and np.array_equal(full[key],noun[key]),(sid,key)
   assert full['boxes'].shape==(7,256,6) and noun['boxes'].shape==(7,256,6)
   assert full['bbf_scores_by_class'].shape==noun['bbf_scores_by_class'].shape==(7,6,256)
 # Smoke-to-full overlap is exact at file level if NPZ serialization also matches;
 # always compare every array to distinguish content from serialization changes.
 repeat=[]
 for name,h in sorted(ss['files'].items()):
  assert name in seal['files'];differ=[];maxdiff=0.
  with np.load(a.smoke/name,allow_pickle=False) as x,np.load(a.cache/name,allow_pickle=False) as y:
   assert set(x.files)==set(y.files)
   for k in x.files:
    if not np.array_equal(x[k],y[k]):
     differ.append(k)
     if x[k].dtype.kind not in 'US':maxdiff=max(maxdiff,float(np.max(np.abs(x[k].astype(float)-y[k].astype(float)))))
  repeat.append({'file':name,'file_SHA_equal':h==seal['files'][name],'all_arrays_bitexact':not differ,'differing_fields':differ,'max_numeric_abs_delta':maxdiff})
 traces=read(a.cache/'trace.json');assert len(traces)==19412;index={(x['id'],x['condition']):x for x in traces};assert len(index)==19412
 for row in manifest:
  for c in ['full','noun']:assert index[row['id'],c]['platform']==row['platform']
 costs=[]
 for platform in ['waymo','drone','quad']:
  for cond in ['full','noun']:
   sub=[x for x in traces if x['platform']==platform and x['condition']==cond];costs.append({'platform':platform,'condition':cond,'n_samples':len(sub),'model_only_seconds':stats([x['model_only_seconds'] for x in sub]),'model_plus_readout_CPU_prepare_seconds':stats([x['forward_and_fixed_scores_seconds'] for x in sub]),'peak_allocated_cuda_bytes':max(x['peak_allocated_bytes'] for x in sub),'tokens_minmax':[min(x['tokens'] for x in sub),max(x['tokens'] for x in sub)],'raw_nonpositive_candidates_by_head':np.sum([x['nonpositive_raw_size_count_by_head'] for x in sub],axis=0).astype(int).tolist()})
 e2e=[]
 for platform in ['waymo','drone','quad']:
  sub=[x for x in traces if x['platform']==platform and 'sample_end_to_end_seconds_including_sampling_saving' in x];assert len(sub)==sum(x['platform']==platform for x in manifest);e2e.append({'platform':platform,'two_condition_sample_end_to_end_seconds':stats([x['sample_end_to_end_seconds_including_sampling_saving'] for x in sub])})
 tree=ast.parse((R/'run_frozen.py').read_text());bad=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Attribute) and node.attr in ['backward','optim','optimizer']:bad.append(node.attr)
 assert not bad
 bridge=read(a.cache/'native_readout_bridge.json');assert bridge['all_bitexact'] and len(bridge['first_sample_both_conditions_all7heads_all6class'])==14
 rcontrol=read(a.cache/'repeat_control.json');out={'state':'sealed_full9706_CPU_engineering_receipt','full_source_seal_sha256':sha(sealpath),'smoke_source_seal_sha256':sha(a.smoke/'sealed.json'),'manifest_sha256':sha(a.manifest),'code_sha256':codehashes,'auditor_sha256':sha(Path(__file__)),'counts':{'samples':9706,'main_model_forwards':19412,'additional_repeat_forward':1,'prediction_NPZ':19412,'smoke_overlap_samples':48,'smoke_overlap_NPZ':96},'all_prediction_file_SHA_verified':True,'all9706_full_noun_same_point_and_indices_hash':True,'all_parameters_and_buffers_unchanged':True,'frozen_runner_AST_no_optimizer_or_backward':True,'actual_class_or_bbox_not_read_by_this_auditor':True,'native_first_sample_14_all6class_bridge_records':True,'first_full_sample_repeat':rcontrol,'costs_by_platform_condition':costs,'two_condition_sample_end_to_end_by_platform':e2e,'total_run_startup_through_cache_repeat_hash_seconds':seal['elapsed_through_cache_and_repeat_seconds'],'total_prediction_NPZ_bytes':sum((a.cache/n).stat().st_size for n in seal['files']),'smoke_full_all_arrays_bitexact':all(x['all_arrays_bitexact'] for x in repeat),'smoke_full_all_file_SHA_equal':all(x['file_SHA_equal'] for x in repeat),'smoke_full_comparison':repeat,'cost_scope_notes':['Model timer synchronizes immediately around model(inp); excludes point upload, CPU class-map construction and fixed-score reduction.','Broader fixed-readout timer includes CPU token/span-map preparation, upload and originalallclassscore reductions.','End-to-end is once per two-condition sample and includes point load/hash/sampling and NPZ saves; startup/repeat/finalhash remain only in total seal duration.','Peak is allocated CUDA memory, not reserved memory or total nvidia-smi memory.','Per-call model timer sum omits the one extra repeat call because runner records repeat separately.'],'audit_seconds':time.perf_counter()-began}
 assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'receipt':str(a.output),'sha256':sha(a.output),'smoke_full_all_arrays_bitexact':out['smoke_full_all_arrays_bitexact'],'main_files':19412,'GPU_called':False,'GT_read':False}))
if __name__=='__main__':main()
