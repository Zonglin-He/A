"""Same frozen whole-image Mask R-CNN observations on newly authorized frames."""
import sys,time,hashlib
import numpy as np
from common46 import *
def main():
 a=args(__doc__);plan,rows,out=context(a);folder=out/'frames';folder.mkdir(parents=True);(folder/'masks_PRIVATE').mkdir();(folder/'camera_PRIVATE').mkdir();(folder/'attempts_PRIVATE').mkdir()
 import torch,torchvision
 from PIL import Image
 from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
 sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import read_camera_allowlist
 old=read(ROOT/'research/round12/experiment/MASK_CAPTURE_PLAN_LOCK.json');assert old['score_threshold']==old['mask_probability_threshold']==.5;cp=Path(old['weights_path']);assert sha(cp)==old['weights_sha256']
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 state=torch.load(cp,map_location='cpu',weights_only=True);m=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);m.load_state_dict(state,strict=True);del state;m.requires_grad_(False).eval().cuda();assert m.transform.min_size==(800,) and m.transform.max_size==1333 and m.roi_heads.detections_per_img==100
 cats=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories'];assert hashlib.sha256(__import__('json').dumps(cats,separators=(',',':')).encode()).hexdigest()==old['COCO_categories_sha256'];versions=[(n,p._version) for n,p in m.named_parameters()];resized=[];hook=m.transform.register_forward_hook(lambda module,args,result:resized.__setitem__(slice(None),result[0].image_sizes));trace=[];start=time.perf_counter()
 calls=0;reused=0
 for i,r in enumerate(rows):
  record={'status':'started','attempt':i+1,'model_forward_entered':False};rp=folder/'attempts_PRIVATE'/(r['id']+'.json');dump(rp,record)
  try:
   image=np.array(Image.open(r['imagepath']).convert('RGB'),copy=True);H,W=image.shape[:2];mp=folder/'masks_PRIVATE'/(r['id']+'.npz');saved=plan.get('reusable_mask_files',{}).get(r['id'])
   if saved:
    import shutil
    assert plan['execution_mode']=='engineering_oldA2' and sha(saved['path'])==saved['sha256'] and sha(r['imagepath'])==saved['image_sha256'];shutil.copyfile(saved['path'],mp);z=npz(mp);boxes=z['boxes_xyxy'];resized[:]=[tuple(z['resized_hw'])];assert tuple(z['original_hw'])==(H,W);reused+=1;record.update(status='raw_mask_reused_after_prior_io_failure',prior_attempt_path=saved['path'],prior_raw_sha256=saved['sha256'],mask_sha256=sha(mp));dump(rp,record)
   else:
    assert calls<plan['max_MaskRCNN_calls'];calls+=1;t=torch.from_numpy(image).permute(2,0,1).float().div_(255).cuda();record['model_forward_entered']=True;dump(rp,record)
    with torch.inference_mode():o=m([t])[0]
    # Same native R12 probability/score cutoffs and original pixel mask output.
    use=o['scores']>=.5;indices=torch.nonzero(use).flatten();boxes=o['boxes'][use].cpu().numpy();scores=o['scores'][use].cpu().numpy();labels=o['labels'][use].cpu().numpy();masks=(o['masks'][use,0]>=.5).cpu().numpy();np.savez_compressed(mp,boxes_xyxy=boxes,scores=scores,class_ids=labels,masks_bool=masks,instance_indices=indices.cpu().numpy(),original_hw=np.array([H,W]),resized_hw=np.array(resized[0]));record.update(status='model_completed_masks_saved',mask_sha256=sha(mp));dump(rp,record)
   fields=read_camera_allowlist(r['metapath']);dump(folder/'camera_PRIVATE'/(r['id']+'.json'),fields);trace.append({'id_PRIVATE':r['id'],'image_sha256':sha(r['imagepath']),'point_sha256':sha(r['pointpath']),'metadata_sha256':sha(r['metapath']),'mask_cache_sha256':sha(mp),'original_hw':[H,W],'resized_hw':list(resized[0]),'instance_n':len(boxes)});record['status']='completed';dump(rp,record)
  except BaseException as e:record.update(status='error_preserved_no_retry',error_type=type(e).__name__,error=str(e));dump(rp,record);raise
  if (i+1)%32==0:dump(folder/'PROGRESS.json',{'completed':i+1,'n':len(rows)});print('R46_MASKS',i+1,flush=True)
 hook.remove();assert versions==[(n,p._version) for n,p in m.named_parameters()];dump(folder/'FRAME_BINDINGS_PRIVATE.json',trace);seal(folder,'R46_SAME_R12_FROZEN_MASKRCNN_AND_CAMERA_ALLOWLIST_NO_GT',n=len(rows),model_calls=calls,reused_raw_masks_after_prior_io_failure=reused,prior_mask_calls=plan.get('prior_mask_calls',0),weights_sha256=old['weights_sha256'],model_unchanged=True,no_caption_sourcebox_query_input=True,seconds=time.perf_counter()-start,torch=torch.__version__,torchvision=torchvision.__version__)
if __name__=='__main__':main()
