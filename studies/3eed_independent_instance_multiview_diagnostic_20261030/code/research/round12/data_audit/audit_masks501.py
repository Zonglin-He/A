"""Independent cached MaskRCNN observation binding audit, no model/GT/pixel decode."""
from pathlib import Path
import argparse,ast,collections,hashlib,json,sys,time
import numpy as np
from PIL import Image
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(a):
 started=time.perf_counter();sp=a.masks/'SEALED.json';assert sha(sp)==a.expected_seal_sha256;s=read(sp)
 assert s['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and s['frame_n']==501 and not s['trained_or_updated'] and s['mask_returned_original_image_pixels'] and len(s['files'])==503
 for rel,digest in s['files'].items():
  p=(a.masks/rel).resolve();assert p.is_relative_to(a.masks.resolve()) and sha(p)==digest
 plan=read(a.masks/'registered_plan.json');assert sha(a.masks/'registered_plan.json')==s['plan_sha256'];assert plan['stage']=='independent_fullimage_masks' and plan['score_threshold']==plan['mask_probability_threshold']==.5 and plan['program_sha256']==s['program_sha256']==sha(B/'crossplatform/round12/capture_masks.py') and plan['weights_sha256']==s['weights_sha256']
 assert s['strict_FP32_flags']=={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
 rows=read(D/'MASK_FRAME_INPUTS.json');fi=read(D/'FRAME_INDEX_PRIVATE.json');assert len(rows)==len(fi)==501 and sha(D/'MASK_FRAME_INPUTS.json')==s['frames_sha256']==plan['frames_sha256'] and all(set(r)=={'frame_alias','imagepath'} for r in rows)
 lookup={r['frame_alias']:r for r in rows};assert len(lookup)==501 and set(lookup)==set(fi)
 trace=read(a.masks/'TRACE.json');assert len(trace)==len({r['frame_alias'] for r in trace})==501;trace={r['frame_alias']:r for r in trace};assert set(trace)==set(fi)
 # Read the public torchvision taxonomy literal through AST, never import torch.
 meta=next((Path(p)/'torchvision/models/_meta.py' for p in sys.path if (Path(p)/'torchvision/models/_meta.py').is_file()),None);assert meta
 tree=ast.parse(meta.read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_COCO_CATEGORIES' for t in n.targets));cats=ast.literal_eval(assignment.value);assert len(cats)==91 and hashlib.sha256(json.dumps(cats,separators=(',',':')).encode()).hexdigest()==plan['COCO_categories_sha256']
 names={p.name for p in a.masks.glob('*.npz')};assert names=={r['frame_alias']+'.npz' for r in rows}
 totals=collections.Counter();classes=collections.Counter();counts=[];sizes=collections.Counter();score_min=1.;score_max=0.;perplatform=collections.defaultdict(collections.Counter)
 for row in rows:
  alias=row['frame_alias'];frame=fi[alias];tr=trace[alias];assert row['imagepath']==frame['imagepath'] and sha(row['imagepath'])==frame['image_sha256']==tr['image_sha256']
  # PIL open/size only reads JPEG metadata; no convert/load/np.asarray is called.
  with Image.open(row['imagepath']) as im:W,H=im.size
  with np.load(a.masks/(alias+'.npz'),allow_pickle=False) as z:
   assert set(z.files)=={'boxes_xyxy','scores','class_ids','masks_bool','instance_indices','original_hw','resized_hw'}
   boxes=z['boxes_xyxy'].copy();scores=z['scores'].copy();labels=z['class_ids'].copy();masks=z['masks_bool'].copy();indices=z['instance_indices'].copy();original=z['original_hw'].copy();resized=z['resized_hw'].copy()
  n=len(scores);assert n==tr['instance_n'] and 0<=n<=100 and boxes.shape==(n,4) and labels.shape==indices.shape==(n,) and masks.shape==(n,H,W) and masks.dtype==bool
  assert np.isfinite(boxes).all() and np.isfinite(scores).all() and ((scores>=.5)&(scores<=1)).all() and np.issubdtype(labels.dtype,np.integer) and ((labels>0)&(labels<len(cats))).all()
  assert all(cats[int(k)] not in ('N/A','__background__') for k in labels)
  assert np.issubdtype(indices.dtype,np.integer) and (indices>=0).all() and (indices<100).all() and np.all(np.diff(indices)>0)
  assert np.array_equal(original,[H,W]) and tr['original_hw']==[H,W] and np.array_equal(resized,tr['resized_hw'])
  assert len(resized)==2 and np.isfinite(resized).all() and (resized>0).all() and min(resized)==800 and max(resized)<=1333
  assert (boxes[:,0]>=0).all() and (boxes[:,1]>=0).all() and (boxes[:,2]<=W+1e-4).all() and (boxes[:,3]<=H+1e-4).all() and (boxes[:,2]>boxes[:,0]).all() and (boxes[:,3]>boxes[:,1]).all()
  K=np.asarray(frame['image_intrinsic'],float);E=np.asarray(frame['image_extrinsic'],float);assert K.shape==(3,3) and E.shape==(4,4) and np.isfinite(K).all() and np.isfinite(E).all() and K[0,0]>0 and K[1,1]>0 and np.allclose(K[2],[0,0,1]) and 0<=K[0,2]<W and 0<=K[1,2]<H
  mh=[hashlib.sha256(m.tobytes()).hexdigest() for m in masks];assert mh==tr['mask_sha256']
  empty=sum(not m.any() for m in masks);totals['frames']+=1;totals['instances']+=n;totals['mask_SHA_checked']+=n;totals['empty_mask_instances']+=empty;totals['zero_instance_frames']+=n==0;perplatform[frame['platform']]['frames']+=1;perplatform[frame['platform']]['instances']+=n;classes.update(cats[int(x)] for x in labels);counts.append(n);sizes[str([H,W])]+=1
  if n:score_min=min(score_min,float(scores.min()));score_max=max(score_max,float(scores.max()))
 assert totals['frames']==501
 out={'state':'independent_actual_R12_mask501_observation_binding_audit_pass','n_frames':501,'bound_files_checked':503,'auditor_sha256':sha(__file__),'mask_seal_sha256':sha(sp),'frame_input_sha256':sha(D/'MASK_FRAME_INPUTS.json'),'frame_index_sha256':sha(D/'FRAME_INDEX_PRIVATE.json'),'capture_program_sha256':s['program_sha256'],'weights_sha256_recorded_no_weight_payload_load':s['weights_sha256'],'COCO_taxonomy_sha256':plan['COCO_categories_sha256'],'taxonomy_source_file_sha256':sha(meta),'counts':dict(totals),'detection_count_min_median_max':[min(counts),float(np.median(counts)),max(counts)],'original_HW_histogram':dict(sizes),'score_range':[score_min,score_max],'predicted_class_histogram_not_GT':dict(classes),'by_platform':{k:dict(v) for k,v in perplatform.items()},'original_image_pixels_mask_and_boxes_and_K_contract_checked':True,'all_instance_mask_SHA_equal_trace':True,'mask_threshold_binary_contract':'bool masks with threshold.5bound to frozen producerplan/code; underlying probability tensors notretained, threshold decision not independently recomputed.','image_bytes_rehashed_and_JPEG_header_size_only_no_pixel_decode':True,'independent_fullimage_frontend_no_caption_q0_or_sourcebox_input':True,'semantic_or_identity_correctness_not_established':True,'no_GT_bbox_class_instance_annotation_taskscore_pointarrays_model_GPU_or_weight_load':True,'seconds':time.perf_counter()-started}
 assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--masks',type=Path,required=True);p.add_argument('--expected-seal-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
