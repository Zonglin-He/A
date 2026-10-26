"""Actual four-case same Source/eight RGB, official preprocessing CPU contracts."""
from pathlib import Path
import sys,json,hashlib,time
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/expert_generalization_v1';P=R/'results/expert_generalization_v1'
sys.path[:0]=[str(R/'studies/review_strengthening_v1'),str(R),str(R/'tools')]
import review_common as rc
def run():
 import numpy as np,torch
 import torchvision.transforms as T
 import torch.nn.functional as F
 from PIL import Image
 from stvg.decode import decode
 from stvg.teacher_bridge import evidence_key
 torch.set_num_threads(1);rc.gt_guard();plan=rc.read(D/'intent.json');assert plan['fingerprint']==rc.digest({k:v for k,v in plan.items()if k!='fingerprint'});lock=rc.read(D/'source_lock.json')
 for name,h in lock['files_sha256'].items():assert rc.sha(D/'vendor/ReferFormer'/name)==h
 official=(D/'vendor/ReferFormer/inference_ytvos.py').read_text();assert 'T.Resize(360)'in official and 'pred_scores = pred_scores.mean(0)'in official and 'max_inds = max_ind.repeat(video_len)'in official and 'all_pred_boxes = pred_boxes[range(video_len), max_inds]'in official
 transform=T.Compose([T.Resize(360),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]);runtime=rc.read(rc.RUN/'plan.json')['provider_runtime_fingerprint'];roots=[rc.PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache'];proof=[];start=time.perf_counter()
 for ds in rc.COUNTS:
  for row in rc.records(ds)[:2]:
   cap,parent=rc.selected.suite().load_parent(ds,row);frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];pos=list(cap['positions8']);key,context=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,fingerprint=runtime,dataset=ds);hits=[r for r in roots if(r/(key+'.json')).exists()];assert hits
   root=hits[0];meta=rc.read(root/(key+'.json'));assert meta['key']==key and meta['context']==context and meta['sha256']==rc.sha(root/(key+'.pt'));teacher=torch.load(root/(key+'.pt'),map_location='cpu',weights_only=False,mmap=True)
   assert teacher['native_same_track_all8']and teacher['nested_mask_all_false']and np.array_equal(teacher['boxes'].numpy(),np.asarray(cap['geometry']['native_boxes']))
   tensor=torch.stack([transform(Image.fromarray(f))for f in frames[pos]]);again=torch.stack([transform(Image.fromarray(f))for f in frames[pos]]);assert torch.equal(tensor,again)and list(tensor.shape)==teacher['input_shape']
   # Source and alternative use the same raw eight frames and textual expression;
   # only official alternative model inference is deferred, never fabricated.
   expr=' '.join(row['metadata']['caption'].lower().split());assert hashlib.sha256(expr.encode()).hexdigest()==teacher['text_sha256']
   b=teacher['boxes'].numpy();lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2;back=np.concatenate(((lo+hi)/2,hi-lo),axis=1);assert np.allclose(back,b,atol=1e-7,rtol=0)
   logits=torch.as_tensor(teacher['native_mask_logits']).float();h,w=frames.shape[1:3];binary=F.interpolate(logits[:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0]>0;assert torch.equal(binary,F.interpolate(logits[:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0].sigmoid()>.5)
   proof.append(dict(dataset=ds,actual_eight_RGB_sha256=hashlib.sha256(frames[pos].tobytes()).hexdigest(),parent_capture_sha256=parent['capture_sha256'],original_teacher_payload_sha256=meta['sha256'],original_teacher_context_receipt_sha256=rc.sha(root/(key+'.json')),original_native_boxes_and_eight_frame_inputbinding_exact=True,official_360_preprocessing_repeat_exact_and_matches_current_expert_shape=True,expression_hash_exact=True,cxcywh_conversion_checked=True,selected_query_one_track_rule_source_audited=True,alternative_model_outputs_exist=False))
 assert not torch.cuda.is_initialized();v=dict(status='passed_four_actual_input_contracts_only',cases=proof,worker=rc.identity(),elapsed_wall_seconds=time.perf_counter()-start,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,actual_ReferFormer_model_bridge=False,actual_CUDA_extension_kernel_tested=False);rc.atomic(D/'input_contract_receipt.json',v);rc.atomic(P/'input_contract_receipt.json',v);print(v['status'])
if __name__=='__main__':run()
