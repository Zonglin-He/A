"""Independent R10 sourcecache/observation binding audit. No GT/G/IoU/model API."""
from pathlib import Path
import argparse,json,hashlib,collections
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main(a):
 sp=a.source/'SEALED.json';assert sha(sp)==a.expected_seal_sha256;s=read(sp);assert s['state']=='sealed_R10_source_forward_and_input_observations_no_G_no_GT' and s['n_samples']==s['n_source_cache_files']==s['n_observation_files']==160
 for rel,h in s['files'].items():
  p=(a.source/rel).resolve();assert p.is_relative_to(a.source.resolve()) and sha(p)==h
 assert len(s['files'])==323 and s['source_repeat_bitexact'] and s['source_unchanged']
 assert sha(a.manifest)==s['manifest_sha256'];samples=read(a.manifest);assert len(samples)==160 and len({r['id'] for r in samples})==160 and len({r['pointpath'] for r in samples})==160
 plan=read(a.source/'registered_plan.json');assert sha(a.source/'registered_plan.json')==s['plan_sha256'] and plan['seed']==0 and plan['strict_FP32_flags']==s['strict_FP32_flags']
 for key in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert s['strict_FP32_flags'][key] is False
 assert s['strict_FP32_flags']['cudnn_deterministic']
 for rel,h in plan['capture_program_sha256'].items():assert sha(a.program_root/rel)==h
 assert plan['source_model_state_fingerprint']==s['source_model_state_fingerprint'] and plan['source_checkpoint_sha256']==s['source_checkpoint_sha256']
 trace={r['id']:r for r in read(a.source/'trace.json')};assert set(trace)=={r['id'] for r in samples};heads=set();counts=collections.defaultdict(collections.Counter);scenes=collections.defaultdict(collections.Counter)
 for row in samples:
  assert set(row)=={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
  sid=row['id'];cache_rel='source_cache/'+sid+'.npz';obs_rel='observations/'+sid+'.npz';t=trace[sid]
  with np.load(a.source/cache_rel,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
  with np.load(a.source/obs_rel,allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
  assert str(c['sample_id'])==sid and str(c['source_checkpoint_sha256'])==s['source_checkpoint_sha256'];heads.add(str(c['selected_parameter_sha256']))
  q=int(c['query_index']);assert q==int(c['text_scores'][0].argmax())==t['q0'] and c['boxes_raw6'].shape==(1,256,6) and c['text_scores'].shape==(1,256) and np.isfinite(c['boxes_raw6']).all() and np.isfinite(c['text_scores']).all()
  for k,shape in [('base_xyz',(1,256,3)),('center_features',(1,288,256)),('size_features',(1,288,256))]:assert c[k].shape==shape and np.isfinite(c[k]).all()
  assert o['sampled_points_model'].shape==(16384,3) and o['sampled_points_model'].dtype==np.float32 and np.isfinite(o['sampled_points_model']).all() and o['raw_indices'].shape==(16384,)
  assert (o['raw_indices']>=0).all() and (o['raw_indices']<t['raw_points']).all() and len(np.unique(o['raw_indices']))==t['unique_sample_indices']
  for datafield,hashfield in [('sampled_points_model','point_sample_sha256'),('raw_indices','raw_indices_sha256')]:
   digest=array_sha(o[datafield]);assert o[hashfield].tobytes().hex()==digest and c[hashfield].tobytes().hex()==digest
  assert c['actual_model_text_sha256'].tobytes()==hashlib.sha256((row['caption']+plan['sentinel']).encode()).digest()
  assert c['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).digest()
  assert t['cache_sha256']==s['files'][cache_rel] and t['observation_sha256']==s['files'][obs_rel]
  counts[row['platform']]['n']+=1;counts[row['platform']]['source_selected_nonpositive_size']+=int((c['boxes_raw6'][0,q,3:]<=0).any());counts[row['platform']]['sampling_with_repeated_raw_index']+=int(len(np.unique(o['raw_indices']))<16384);scenes[row['platform']][row['scene']]+=1
 assert len(heads)==1
 old=read(a.prior_manifest);assert not ({r['id'] for r in old}&{r['id'] for r in samples}) and not ({r['pointpath'] for r in old}&{r['pointpath'] for r in samples})
 out={'state':'independent_actual_R10_source160_observation_binding_audit_pass','auditor_sha256':sha(Path(__file__)),'source_seal_sha256':sha(sp),'manifest_sha256':sha(a.manifest),'capture_program_sha256':sha(a.program_root/'crossplatform/round10/capture_source.py'),'n':160,'bound_files_checked':len(s['files']),'counts':{k:dict(v) for k,v in counts.items()},'scene_counts':{k:len(v) for k,v in scenes.items()},'per_scene_samples_histogram':{k:dict(collections.Counter(v.values())) for k,v in scenes.items()},'all_same_source_head_sha256':next(iter(heads)),'source_cache_observation_point_and_index_arrays_bound':True,'all_originalq0_argmax_checked':True,'same_manifestrow_actualmodeltext_and_checkpoint_binding':True,'priorR9_ID_and_pointpath_intersection0':True,'source_frozen_scope':'Sourcefullstate fingerprint and dynamic assertions/repeat in sealed producer; no new model instantiation.','targetGT_annotations_scoringindex_camera_metadata_content_or_G_read':False,'no_GPU_model_IoU':True};assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('source','manifest','program-root','prior-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--expected-seal-sha256',required=True);main(p.parse_args())
