"""Frozen whole-image instance observations. No caption/q0/box/GT input API."""
from pathlib import Path
import argparse,hashlib,json,time,re
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 import torch,torchvision
 from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
 from PIL import Image
 plan=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha256 and plan['stage']=='independent_fullimage_masks'
 assert plan['program_sha256']==sha(__file__)
 assert plan['score_threshold']==plan['mask_probability_threshold']==.5
 rows=json.loads(a.frames.read_text());assert sha(a.frames)==plan['frames_sha256'];assert rows and all(set(r)=={'frame_alias','imagepath'} for r in rows)
 assert len({r['frame_alias'] for r in rows})==len(rows) and all(re.fullmatch(r'input_frame_[0-9]{5}',r['frame_alias']) for r in rows)
 assert not a.output.exists();a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 cp=Path(plan['weights_path']);assert sha(cp)==plan['weights_sha256'];state=torch.load(cp,map_location='cpu',weights_only=True)
 assert isinstance(state,dict) and all(isinstance(x,torch.Tensor) for x in state.values())
 m=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None);m.load_state_dict(state,strict=True);del state;m.requires_grad_(False).eval().cuda()
 assert m.transform.min_size==(800,) and m.transform.max_size==1333 and m.roi_heads.detections_per_img==100
 cats=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories'];assert hashlib.sha256(json.dumps(cats,separators=(',',':')).encode()).hexdigest()==plan['COCO_categories_sha256']
 files={};trace=[];start=time.perf_counter();resized=[]
 def hook(module,args,output):resized[:]=output[0].image_sizes
 handle=m.transform.register_forward_hook(hook)
 for i,r in enumerate(rows):
  tick=time.perf_counter();image=np.array(Image.open(r['imagepath']).convert('RGB'),copy=True);H,W=image.shape[:2]
  t=torch.from_numpy(image).permute(2,0,1).float().div_(255).cuda();torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():o=m([t])[0]
  use=o['scores']>=plan['score_threshold'];indices=torch.nonzero(use).flatten();boxes=o['boxes'][use].cpu().numpy();scores=o['scores'][use].cpu().numpy();labels=o['labels'][use].cpu().numpy();masks=(o['masks'][use,0]>=plan['mask_probability_threshold']).cpu().numpy()
  assert masks.shape==(len(boxes),H,W) and np.isfinite(boxes).all() and np.isfinite(scores).all() and np.all((labels>0)&(labels<len(cats)))
  f=r['frame_alias']+'.npz';np.savez_compressed(a.output/f,boxes_xyxy=boxes,scores=scores,class_ids=labels,masks_bool=masks,instance_indices=indices.cpu().numpy(),original_hw=np.array([H,W]),resized_hw=np.array(resized[0]))
  files[f]=sha(a.output/f);trace.append({'frame_alias':r['frame_alias'],'image_sha256':sha(r['imagepath']),'instance_n':len(boxes),'mask_sha256':[hashlib.sha256(x.tobytes()).hexdigest() for x in masks],'original_hw':[H,W],'resized_hw':list(resized[0]),'seconds':time.perf_counter()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
  if (i+1)%40==0:print(json.dumps({'frames':i+1,'seconds':time.perf_counter()-start}),flush=True)
  del o,t,use,masks
 handle.remove();tf=a.output/'TRACE.json';tf.write_text(json.dumps(trace,indent=2)+'\n');files['TRACE.json']=sha(tf);files['registered_plan.json']=sha(a.output/'registered_plan.json')
 seal={'state':'sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT','frame_n':len(rows),'program_sha256':sha(__file__),'weights_sha256':sha(cp),'plan_sha256':sha(a.plan),'frames_sha256':sha(a.frames),'files':files,'seconds':time.perf_counter()-start,'torch':torch.__version__,'torchvision':torchvision.__version__,'trained_or_updated':False,'strict_FP32_flags':{'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True},'mask_returned_original_image_pixels':True,'inference_GPU_only_sequential_root_owner':True};(a.output/'SEALED.json').write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps({k:v for k,v in seal.items() if k!='files'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
