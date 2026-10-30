"""Independent R12 source146 sealed NPZ/hash/observation audit, CPU only.
No model, GPU, target annotation, GT/index/scored/media/point-file or G access.
"""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np
B=Path(__file__).resolve().parents[3]
KEYS={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bounded(root,name):
 p=(Path(root)/name).resolve();assert p.is_relative_to(Path(root).resolve());return p

def main(a):
 sp=a.source/'SEALED.json';assert sha(sp)==a.expected_seal_sha256;s=read(sp);assert s['state']=='sealed_R12_source_forward_and_input_observations_no_G_no_GT'
 assert s['n_samples']==s['n_source_cache_files']==s['n_observation_files']==146 and len(s['files'])==295
 for rel,digest in s['files'].items():assert sha(bounded(a.source,rel))==digest,rel
 assert s['source_repeat_bitexact'] and s['source_unchanged'] and s['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 assert sha(a.manifest)==s['manifest_sha256'];rows=read(a.manifest);assert len(rows)==146 and len({r['id'] for r in rows})==len({r['pointpath'] for r in rows})==len({r['imagepath'] for r in rows})==146
 assert all(set(r)==KEYS for r in rows);assert read(a.source/'manifest_input.json')==rows
 plan=read(a.source/'registered_plan.json');assert sha(a.source/'registered_plan.json')==s['plan_sha256'] and plan['manifest_sha256']==s['manifest_sha256'] and plan['expected_samples']==146 and plan['stage']=='source_capture' and plan['source_no_G_or_GT'] and plan['no_source_retraining_or_teacher_EMA']
 assert plan['seed']==0 and plan['sentinel']==' . not mentioned' and plan['noun_fallback']=='all_content_tokens' and plan['strict_FP32_flags']==s['strict_FP32_flags']=={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
 assert plan['source_model_state_fingerprint']==s['source_model_state_fingerprint']=='0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9' and plan['source_checkpoint_sha256']==s['source_checkpoint_sha256']
 for rel,digest in plan['capture_program_sha256'].items():assert sha(bounded(B,rel))==digest
 trace=read(a.source/'trace.json');assert len(trace)==146 and len({r['id'] for r in trace})==146;trace={r['id']:r for r in trace};assert set(trace)=={r['id'] for r in rows}
 frameindex=read(Path(__file__).with_name('FRAME_INDEX_PRIVATE.json'));pair_rows=read(Path(__file__).with_name('PAIRS_PRIVATE.json'))['rows_PRIVATE'];anchor_alias={r['anchor_id_PRIVATE']:r['anchor_frame_alias'] for r in pair_rows};assert set(anchor_alias)==set(trace)
 assert {p.name for p in (a.source/'source_cache').glob('*.npz')}=={r['id']+'.npz' for r in rows} and {p.name for p in (a.source/'observations').glob('*.npz')}=={r['id']+'.npz' for r in rows}
 counts=collections.defaultdict(collections.Counter);scenes=collections.defaultdict(collections.Counter);heads=set();ranges=collections.defaultdict(list)
 for r in rows:
  sid=r['id'];t=trace[sid];platform=r['platform'];assert t['platform']==platform and t['scene']==r['scene'];scenes[platform][r['scene']]+=1;counts[platform]['n']+=1
  cp='source_cache/'+sid+'.npz';op='observations/'+sid+'.npz'
  with np.load(a.source/cp,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
  with np.load(a.source/op,allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
  assert str(c['sample_id'])==sid and str(c['source_checkpoint_sha256'])==s['source_checkpoint_sha256']
  assert c['boxes_raw6'].shape==(1,256,6) and c['text_scores'].shape==(1,256) and np.isfinite(c['boxes_raw6']).all() and np.isfinite(c['text_scores']).all()
  q=int(c['query_index']);assert q==int(c['text_scores'][0].argmax())==t['q0'];heads.add(str(c['selected_parameter_sha256']))
  for field,shape in [('base_xyz',(1,256,3)),('center_features',(1,288,256)),('size_features',(1,288,256))]:assert c[field].shape==shape and np.isfinite(c[field]).all()
  p=o['sampled_points_model'];ix=o['raw_indices'];assert p.shape==(16384,3) and p.dtype==np.float32 and np.isfinite(p).all() and ix.shape==(16384,) and np.issubdtype(ix.dtype,np.integer) and (ix>=0).all() and (ix<t['raw_points']).all()
  uid,first,inverse=np.unique(ix,return_index=True,return_inverse=True);assert len(uid)==t['unique_sample_indices'] and np.array_equal(p,p[first][inverse]),'Same raw return must keep exact XYZ across replacement samples'
  unique_xyz=np.unique(p,axis=0);assert len(unique_xyz)<=len(uid)
  counts[platform]['sampling_with_repeated_raw_indices']+=int(len(uid)<len(ix));counts[platform]['distinct_rawIDs_coincident_XYZ_cases']+=int(len(unique_xyz)<len(uid));counts[platform]['replacement_duplicate_rows_total']+=len(ix)-len(uid);counts[platform]['coincident_rawID_excess_total']+=len(uid)-len(unique_xyz)
  ranges[platform+'|unique_raw_indices'].append(len(uid));ranges[platform+'|unique_XYZ'].append(len(unique_xyz));ranges[platform+'|raw_points'].append(t['raw_points'])
  counts[platform]['source_selected_nonpositive_dimensions']+=int((c['boxes_raw6'][0,q,3:]<=0).any())
  for field,hfield in [('sampled_points_model','point_sample_sha256'),('raw_indices','raw_indices_sha256')]:assert arrsha(o[field])==o[hfield].tobytes().hex()==c[hfield].tobytes().hex()
  assert c['actual_model_text_sha256'].tobytes()==hashlib.sha256((r['caption']+plan['sentinel']).encode()).digest()
  assert c['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).digest()
  assert t['cache_sha256']==s['files'][cp] and t['observation_sha256']==s['files'][op]
  assert all(isinstance(t[k],str) and len(t[k])==64 for k in ('point_file_sha256','image_sha256','camera_container_sha256'))
  frame=frameindex[anchor_alias[sid]];assert frame['pointpath']==r['pointpath'] and frame['imagepath']==r['imagepath'] and frame['metapath']==r['metapath']
  assert t['point_file_sha256']==frame['point_sha256'] and t['image_sha256']==frame['image_sha256'] and t['camera_container_sha256']==frame['metadata_sha256']
 assert len(heads)==1 and dict(collections.Counter(r['platform'] for r in rows))=={'drone':50,'quad':96} and sorted(n for group in scenes.values() for n in group.values())==[18,32,32,32,32]
 prior=[]
 for rel in ('research/round9/experiment/diagnostic160_gpu_inputs.json','research/round10/input_prepare/diagnostic160_gpu_inputs.json','research/round11/input_prepare/diagnostic320_gpu_inputs.json','research/round8/experiment/smoke40_gpu_inputs.json','research/round7/experiment/smoke_gpu_inputs.json'):prior.extend(read(B/rel))
 assert len({r['id'] for r in prior})==len({r['pointpath'] for r in prior})==700 and not {r['id'] for r in rows}&{r['id'] for r in prior} and not {r['pointpath'] for r in rows}&{r['pointpath'] for r in prior}
 out={'state':'independent_actual_R12_source146_input_observation_binding_audit_pass','auditor_sha256':sha(__file__),'source_seal_sha256':sha(sp),'manifest_sha256':sha(a.manifest),'capture_program_sha256':sha(B/'crossplatform/round12/capture_source.py'),'n':146,'bound_files_checked':295,'counts':{k:dict(v) for k,v in counts.items()},'scene_counts':{k:len(v) for k,v in scenes.items()},'per_scene_quotas':[32,18,32,32,32],'support_count_ranges':{k:{'min':min(v),'max':max(v),'sum':sum(v)} for k,v in ranges.items()},'all_same_source_head_sha256':next(iter(heads)),'all_original_q0_argmax_checked':True,'all9inputvalues_actualtext_pointandindex_cacheandtrace_bound':True,'repeated_rawID_exact_coordinates_verified':True,'prior700_ID_pointpath_intersection0':True,'source_frozen_scope':'Producer strict fullstate/parameter/grad/eval assertions and firstrepeat; auditor uses sealed cache arrays only, no model instantiation.','camera_point_image_original_files_not_reopened':'Producer originalfile SHA strings checked and bound by trace; no independently regenerated point sample or camera calibration claim.','anchor_input_contenthash_bound_to_frameindex_and_producer_filehashes':True,'input_frame_index_sha256':sha(Path(__file__).with_name('FRAME_INDEX_PRIVATE.json')),'no_GT_annotation_target_scoringindex_oldmetrics_media_pointfile_weight_model_GPU_or_G_access':True}
 assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('source','manifest','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--expected-seal-sha256',required=True);main(p.parse_args())
