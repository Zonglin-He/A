"""Independent actual R13 captures and old8 cache-reuse audit. No GT/model/IoU."""
from pathlib import Path
import ast,collections,hashlib,json,re,sys
import numpy as np
from PIL import Image
B=Path(__file__).resolve().parents[3];V=B/'research/round13/evaluation';D=B/'research/round13/input';S=B/'crossplatform/round13/source108_attempt001';M=B/'crossplatform/round13/masks108_attempt001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
checks=0;membercounts={};flags={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
def ck(x):
 global checks
 assert x;checks+=1
def sealverify(root):
 s=read(root/'SEALED.json')
 for rel,h in s['files'].items():
  p=(root/rel).resolve();ck(p.is_relative_to(root.resolve()) and p.is_file() and sha(p)==h)
 membercounts[str(root.relative_to(B))]=len(s['files']);return s
ss=sealverify(S);ms=sealverify(M);ck(sha(S/'SEALED.json')=='57c5156e2ccf94d3965b37eb7de5ed4a03efa5789af6695aea8d45850a2b3b9d');ck(sha(M/'SEALED.json')=='ab4ada616de7db136c5a568f7da18e6fbde3588def4086e86e4cb9184559e67d')
sp=read(S/'registered_plan.json');mp=read(M/'registered_plan.json');ck(sha(S/'registered_plan.json')==ss['plan_sha256']==sha(B/'research/round13/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));ck(sha(M/'registered_plan.json')==ms['plan_sha256']==sha(B/'research/round13/experiment/MASK_CAPTURE_PLAN_LOCK.json'))
ck(ss['state']=='sealed_R12_source_forward_and_input_observations_no_G_no_GT' and ss['n_samples']==ss['n_source_cache_files']==ss['n_observation_files']==108 and len(ss['files'])==219)
ck(ms['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and ms['frame_n']==108 and len(ms['files'])==110 and not ms['trained_or_updated'] and ms['mask_returned_original_image_pixels'])
ck(ss['source_repeat_bitexact'] and ss['source_unchanged'] and ss['strict_state_entries']==1034 and ss['strict_FP32_flags']==ms['strict_FP32_flags']==flags)
ck(sp['stage']=='source_capture' and sp['source_no_G_or_GT'] and sp['no_source_retraining_or_teacher_EMA'] and sp['expected_samples']==108 and sp['seed']==0)
ck(sp['sentinel']==' . not mentioned' and sp['noun_fallback']=='all_content_tokens')
ck(ss['source_model_state_fingerprint']==sp['source_model_state_fingerprint']=='0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9')
ck(sha(sp['source_checkpoint_path'])==ss['source_checkpoint_sha256']==sp['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd')
for rel,h in sp['capture_program_sha256'].items():ck(sha(B/rel)==h)
ck(mp['stage']=='independent_fullimage_masks' and mp['independent_from_caption_q0_sourceROI_or_GT'] and mp['no_teacher_EMA_training'])
ck(mp['score_threshold']==mp['mask_probability_threshold']==.5 and mp['min_size']==800 and mp['max_size']==1333 and mp['detections_per_img']==100)
ck(mp['program_sha256']==ms['program_sha256']==sha(B/'crossplatform/round12/capture_masks.py'));ck(sha(mp['weights_path'])==ms['weights_sha256']==mp['weights_sha256']=='73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e')
ck(sp['input_plan_sha256']==mp['input_plan_sha256']==sha(D/'INPUT_PLAN_LOCK.json'))
manifest=read(D/'NEW108_GPU_INPUTS.json');cases=read(D/'CASE116_PRIVATE.json');casebyid={x['id']:x for x in cases};fi=read(D/'FRAME_INDEX_PRIVATE.json');maskin=read(D/'MASK108_FRAME_INPUTS.json')
ck(ss['manifest_sha256']==sp['manifest_sha256']==sha(D/'NEW108_GPU_INPUTS.json'));ck(read(S/'manifest_input.json')==manifest);ck(ms['frames_sha256']==mp['frames_sha256']==sha(D/'MASK108_FRAME_INPUTS.json'))
st=read(S/'trace.json');mt=read(M/'TRACE.json');st={r['id']:r for r in st};mt={r['frame_alias']:r for r in mt};ck(len(st)==len(mt)==len(manifest)==len(maskin)==108)
ck({x.name for x in (S/'source_cache').glob('*.npz')}=={r['id']+'.npz' for r in manifest});ck({x.name for x in (S/'observations').glob('*.npz')}=={r['id']+'.npz' for r in manifest});ck({x.name for x in M.glob('*.npz')}=={r['frame_alias']+'.npz' for r in maskin})
# Literal public taxonomy, no torch/model import.
meta=next(Path(p)/'torchvision/models/_meta.py' for p in sys.path if (Path(p)/'torchvision/models/_meta.py').is_file());tree=ast.parse(meta.read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_COCO_CATEGORIES' for t in n.targets));tax=ast.literal_eval(assignment.value);ck(len(tax)==91 and hashlib.sha256(json.dumps(tax,separators=(',',':')).encode()).hexdigest()==mp['COCO_categories_sha256'])
counts=collections.defaultdict(collections.Counter);classcounts=collections.Counter();heads=set();hws=collections.Counter();mask_total=0
for r in manifest:
 sid=r['id'];t=st[sid];c=casebyid[sid];f=fi[c['frame_alias']];ck(t['platform']==r['platform'] and t['scene']==r['scene']);p=r['platform'];counts[p]['source_cases']+=1
 with np.load(S/'source_cache'/(sid+'.npz'),allow_pickle=False) as z:cache={k:z[k].copy() for k in z.files}
 with np.load(S/'observations'/(sid+'.npz'),allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
 ck(str(cache['sample_id'])==sid and str(cache['source_checkpoint_sha256'])==ss['source_checkpoint_sha256'])
 for key,shape in [('boxes_raw6',(1,256,6)),('text_scores',(1,256)),('base_xyz',(1,256,3)),('center_features',(1,288,256)),('size_features',(1,288,256))]:ck(cache[key].shape==shape and cache[key].dtype==np.float32 and np.isfinite(cache[key]).all())
 q=int(cache['query_index']);ck(q==int(cache['text_scores'][0].argmax())==t['q0']);heads.add(str(cache['selected_parameter_sha256']))
 xyz=obs['sampled_points_model'];ix=obs['raw_indices'];ck(xyz.shape==(16384,3) and xyz.dtype==np.float32 and np.isfinite(xyz).all());ck(ix.shape==(16384,) and np.issubdtype(ix.dtype,np.integer) and (ix>=0).all() and (ix<t['raw_points']).all())
 ids,first,inverse=np.unique(ix,return_index=True,return_inverse=True);ck(len(ids)==t['unique_sample_indices'] and np.array_equal(xyz,xyz[first][inverse]));counts[p]['repeated_sampling_cases']+=len(ids)<len(ix);counts[p]['replacement_rows']+=len(ix)-len(ids);counts[p]['selected_nonpositive_size_cases']+=bool((cache['boxes_raw6'][0,q,3:]<=0).any())
 for key,hkey in [('sampled_points_model','point_sample_sha256'),('raw_indices','raw_indices_sha256')]:ck(arsha(obs[key])==obs[hkey].tobytes().hex()==cache[hkey].tobytes().hex())
 ck(cache['actual_model_text_sha256'].tobytes()==hashlib.sha256((r['caption']+sp['sentinel']).encode()).digest());ck(cache['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).digest())
 ck(t['cache_sha256']==ss['files']['source_cache/'+sid+'.npz'] and t['observation_sha256']==ss['files']['observations/'+sid+'.npz']);ck(t['point_file_sha256']==f['point_sha256'] and t['image_sha256']==f['image_sha256'] and t['camera_container_sha256']==f['metadata_sha256'])
for row in maskin:
 alias=row['frame_alias'];f=fi[alias];t=mt[alias];p=f['platform'];ck(set(row)=={'frame_alias','imagepath'} and row['imagepath']==f['imagepath'] and re.fullmatch(r'input_frame_[0-9]{5}',alias) is not None);ck(sha(row['imagepath'])==f['image_sha256']==t['image_sha256'])
 with Image.open(row['imagepath']) as im:W,H=im.size
 with np.load(M/(alias+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 ck(set(d)=={'boxes_xyxy','scores','class_ids','masks_bool','instance_indices','original_hw','resized_hw'});n=len(d['scores']);mask_total+=n;counts[p]['mask_frames']+=1;counts[p]['mask_instances']+=n;counts[p]['zero_detection_frames']+=n==0
 ck(n==t['instance_n'] and 0<=n<=100 and d['boxes_xyxy'].shape==(n,4) and d['masks_bool'].shape==(n,H,W) and d['masks_bool'].dtype==bool and d['class_ids'].shape==d['instance_indices'].shape==(n,))
 ck(np.isfinite(d['scores']).all() and ((d['scores']>=.5)&(d['scores']<=1)).all() and np.isfinite(d['boxes_xyxy']).all());ck(((d['class_ids']>0)&(d['class_ids']<91)).all() and np.issubdtype(d['class_ids'].dtype,np.integer));ck(all(tax[int(x)] not in ('N/A','__background__') for x in d['class_ids']));classcounts.update(tax[int(x)] for x in d['class_ids'])
 ck(np.issubdtype(d['instance_indices'].dtype,np.integer) and (d['instance_indices']>=0).all() and (d['instance_indices']<100).all() and (np.diff(d['instance_indices'])>0).all())
 ck(np.array_equal(d['original_hw'],[H,W]) and t['original_hw']==[H,W] and np.array_equal(d['resized_hw'],t['resized_hw']));ck(min(d['resized_hw'])==800 and max(d['resized_hw'])<=1333)
 bb=d['boxes_xyxy'];ck((bb[:,0]>=0).all() and (bb[:,1]>=0).all() and (bb[:,2]<=W+1e-4).all() and (bb[:,3]<=H+1e-4).all() and (bb[:,2]>bb[:,0]).all() and (bb[:,3]>bb[:,1]).all())
 ck([hashlib.sha256(x.tobytes()).hexdigest() for x in d['masks_bool']]==t['mask_sha256']);counts[p]['empty_masks']+=sum(not x.any() for x in d['masks_bool']);hws[str([H,W])]+=1
 K=np.array(f['image_intrinsic']);E=np.array(f['image_extrinsic']);ck(K.shape==(3,3) and E.shape==(4,4) and np.isfinite(K).all() and np.isfinite(E).all() and K[0,0]>0 and K[1,1]>0 and np.allclose(K[2],[0,0,1]) and 0<=K[0,2]<W and 0<=K[1,2]<H)
ck(len(heads)==1 and dict(collections.Counter(r['platform'] for r in manifest))=={'drone':32,'quad':76})
# The exposed8 keep the old real producer outputs; only named files are checked,
# not relabelled as newR13 forwards and no cross-process slot identity assumed.
oldS=B/'crossplatform/round12/source146_attempt001';oldM=B/'crossplatform/round12/masks501_attempt001';os=read(oldS/'SEALED.json');om=read(oldM/'SEALED.json');ck(os['source_model_state_fingerprint']==ss['source_model_state_fingerprint'] and os['strict_FP32_flags']==om['strict_FP32_flags']==flags);ck(om['program_sha256']==ms['program_sha256'] and om['weights_sha256']==ms['weights_sha256'])
for c in cases[108:]:
 ck(c['source_dir']==str(oldS.relative_to(B)) and c['mask_dir']==str(oldM.relative_to(B)))
 for rel in ['source_cache/'+c['id']+'.npz','observations/'+c['id']+'.npz']:ck(sha(oldS/rel)==os['files'][rel])
 rel=c['frame_alias']+'.npz';ck(sha(oldM/rel)==om['files'][rel])
result={'state':'PASS_R13_ACTUAL_NEW108_SOURCE_MASK_AND_OLD8_REUSE_AUDIT','source_seal_sha256':sha(S/'SEALED.json'),'mask_seal_sha256':sha(M/'SEALED.json'),'source_plan_sha256':ss['plan_sha256'],'mask_plan_sha256':ms['plan_sha256'],'input_plan_sha256':sha(D/'INPUT_PLAN_LOCK.json'),'auditor_sha256':sha(Path(__file__)),'checks':checks,'new_source_n':108,'new_mask_frame_n':108,'new_source_bound_files':219,'new_mask_bound_files':110,'old_regression_cache_observation_mask_files_checked':24,'all_same_head_parameter_sha256':next(iter(heads)),'source_state_fingerprint':ss['source_model_state_fingerprint'],'actual_source_elapsed_seconds':ss['elapsed_seconds'],'actual_mask_elapsed_seconds':ms['seconds'],'mask_instances_total':mask_total,'counts':{k:dict(v) for k,v in counts.items()},'predicted_class_counts_not_GT':dict(classcounts),'original_HW_histogram':dict(hws),'source_actualtext_sampling_ID_coordinate_cache_trace_binding_checked':True,'source_frozen_repeat_proof_scope':'Producer allstate/parameter/requiresgrad/grad/eval assertions plus firstcase repeat and immutable caches; auditor does not instantiate model.','mask_threshold_scope':'Bool masks/frozen .5 plan/code/hash contract; probabilities absent so .5 decision not independently recomputed.','inherited_R12_state_name_is_truthfully_scoped_by_new108_plans_manifest_and_files':True,'old8_are_old_forward_reuse_not_new_captures':True,'no_GT_bbox_class_instance_annotations_scoringindex_scoredrows_new_IoU_pixel_decode_raw_pointfile_model_GPU_or_weight_payload_load':True,'issues':[]}
p=V/'CAPTURE108_INDEPENDENT_AUDIT.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':checks,'mask_instances':mask_total,'counts':result['counts'],'sha256':sha(p)}))
