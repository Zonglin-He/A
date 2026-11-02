"""R50 full-frame adapter for literal R12 MaskRCNN and camera allowlist.

No source/semantic crop is fed to the detector. Exact registered 693 caches
are reused; all other frames receive one whole-image forward. Chunk receipts
and per-attempt STARTED ledgers permit bounded scheduling, never silent retry.
"""
import sys,time,shutil,hashlib
from pathlib import Path
import numpy as np
from full_common50 import *
from full_frontend50 import known_frame_registry,source_input_hashes,triple

def main():
 a=args(__doc__);plan,rows,out=context(a,'frames');folder=out/'frames';folder.mkdir(parents=True,exist_ok=True)
 if a.finalize:
  assert a.chunk is None
  entries=[]
  for r in rows:
   rp=folder/'cases_PRIVATE'/(r['id']+'.json');b=read(rp);assert b['state']=='R50_FRAME_COMPLETE'
   assert b['id_PRIVATE']==r['id'] and sha(folder/'masks_PRIVATE'/(r['id']+'.npz'))==b['mask_cache_sha256']
   assert sha(folder/'camera_PRIVATE'/(r['id']+'.json'))==b['camera_cache_sha256'];entries.append(b)
  assert sum(b['new_MaskRCNN_forward'] for b in entries)==5219 and sum(b['reused_registered_mask'] for b in entries)==693
  dump(folder/'FRAME_BINDINGS_PRIVATE.json',entries)
  seal(folder,'R50_FULL5912_ORIGINAL_R12_MASKS_CAMERA_COMPLETE',n=5912,model_calls=5219,reused_masks=693,
       no_new_grounder=True,plan_sha256=sha(a.plan));return
 for d in ('masks_PRIVATE','camera_PRIVATE','cases_PRIVATE','attempts_PRIVATE','chunks_PRIVATE'):(folder/d).mkdir(exist_ok=True)
 chunk='all' if a.chunk is None else f'{a.chunk:04d}';receipt=folder/'chunks_PRIVATE'/(chunk+'.json');assert not receipt.exists()
 reuse=known_frame_registry();inputs=source_input_hashes();old=read(ROOT/'research/round12/experiment/MASK_CAPTURE_PLAN_LOCK.json')
 assert old['score_threshold']==old['mask_probability_threshold']==.5
 sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import read_camera_allowlist,CameraSpec
 model=None;calls=0;reused=0;start=time.perf_counter();versions=None;resized=[];hook=None
 entered_total=sum(read(p).get('model_forward_entered',False) for p in (folder/'attempts_PRIVATE').glob('*.json'))
 def detector():
  nonlocal model,versions,hook
  if model is not None:return model
  import torch,torchvision
  from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
  torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  cp=Path(old['weights_path']);assert sha(cp)==old['weights_sha256'];state=torch.load(cp,map_location='cpu',weights_only=True)
  model=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);model.load_state_dict(state,strict=True);del state
  model.requires_grad_(False).eval().cuda();assert model.transform.min_size==(800,) and model.transform.max_size==1333 and model.roi_heads.detections_per_img==100
  cats=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
  assert hashlib.sha256(__import__('json').dumps(cats,separators=(',',':')).encode()).hexdigest()==old['COCO_categories_sha256']
  versions=[(n,p._version) for n,p in model.named_parameters()]
  hook=model.transform.register_forward_hook(lambda module,args,result:resized.__setitem__(slice(None),result[0].image_sizes))
  return model
 try:
  for i,r in enumerate(rows):
   sid=r['id'];rp=folder/'attempts_PRIVATE'/(sid+'.json');assert not rp.exists(),'Existing attempt cannot be retried automatically'
   rec={'status':'started','id_PRIVATE':sid,'model_forward_entered':False,'semantic_or_source_input_to_detector':False};dump(rp,rec)
   inp=inputs[sid];assert sha(r['imagepath'])==inp['image_file_sha256'] and sha(r['metapath'])==inp['camera_metadata_file_sha256']
   mp=folder/'masks_PRIVATE'/(sid+'.npz');cp=folder/'camera_PRIVATE'/(sid+'.json');prior=reuse.get(triple(r))
   if prior:
    assert (prior['image_sha256'],prior['point_sha256'],prior['metadata_sha256'])==(inp['image_file_sha256'],inp['raw_point_file_sha256'],inp['camera_metadata_file_sha256'])
    assert sha(ROOT/prior['mask_path_PRIVATE'])==prior['mask_sha256'];shutil.copyfile(ROOT/prior['mask_path_PRIVATE'],mp)
    fields=prior['camera_fields_PRIVATE'];reused+=1;newcall=False
   else:
    from PIL import Image
    import torch
    rgb=np.array(Image.open(r['imagepath']).convert('RGB'),copy=True);H,W=rgb.shape[:2];m=detector()
    # Budget is global over durable attempts, not reset by process/chunk.
    assert entered_total<plan['max_new_MaskRCNN_calls']==5219
    rec['model_forward_entered']=True;dump(rp,rec);calls+=1;entered_total+=1;newcall=True
    t=torch.from_numpy(rgb).permute(2,0,1).float().div_(255).cuda()
    with torch.inference_mode():o=m([t])[0]
    use=o['scores']>=.5;indices=torch.nonzero(use).flatten()
    np.savez_compressed(mp,boxes_xyxy=o['boxes'][use].cpu().numpy(),scores=o['scores'][use].cpu().numpy(),
      class_ids=o['labels'][use].cpu().numpy(),masks_bool=(o['masks'][use,0]>=.5).cpu().numpy(),
      instance_indices=indices.cpu().numpy(),original_hw=np.array([H,W]),resized_hw=np.array(resized[0]))
    rec.update(status='model_completed_raw_mask_saved',mask_cache_sha256=sha(mp));dump(rp,rec)
    fields=read_camera_allowlist(r['metapath'])
   # Decimal -> float is the independently checked R46 camera JSON bridge.
   dump(cp,fields);CameraSpec.from_camera_fields(r['platform'],read(cp),missing_distortion_policy='assume_zero_declared').validated()
   ins=npz(mp);b=dict(state='R50_FRAME_COMPLETE',id_PRIVATE=sid,platform=r['platform'],
    image_sha256=inp['image_file_sha256'],point_sha256=inp['raw_point_file_sha256'],metadata_sha256=inp['camera_metadata_file_sha256'],
    mask_cache_sha256=sha(mp),camera_cache_sha256=sha(cp),original_hw=ins['original_hw'],resized_hw=ins['resized_hw'],
    instance_n=len(ins['class_ids']),reused_registered_mask=bool(prior),new_MaskRCNN_forward=newcall,
    prior_mask_family=prior['cache_family'] if prior else None,model_weights_sha256=old['weights_sha256'])
   dump(folder/'cases_PRIVATE'/(sid+'.json'),b);rec['status']='completed';dump(rp,rec)
   if (i+1)%16==0:print('R50_FULL_FRAMES',chunk,i+1,len(rows),calls,reused,flush=True)
 except BaseException as e:
  if 'rec' in locals():rec.update(status='error_preserved_no_retry',error_type=type(e).__name__,error=str(e));dump(rp,rec)
  dump(folder/('FAILED_'+chunk+'.json'),{'error_type':type(e).__name__,'error':str(e),'new_calls':calls,'reused':reused});raise
 if model is not None:
  assert versions==[(n,p._version) for n,p in model.named_parameters()];hook.remove()
 dump(receipt,{'state':'R50_FULL_FRAME_CHUNK_COMPLETE','n':len(rows),'new_calls':calls,'reused':reused,
  'seconds':time.perf_counter()-start,'plan_sha256':sha(a.plan),
  'files':{r['id']:{k:sha(folder/k/(r['id']+ext)) for k,ext in [('masks_PRIVATE','.npz'),('camera_PRIVATE','.json'),('cases_PRIVATE','.json')]} for r in rows}})
 print('R50_FULL_FRAMES_CHUNK_SEALED',chunk,len(rows),calls,reused,flush=True)
if __name__=='__main__':main()
