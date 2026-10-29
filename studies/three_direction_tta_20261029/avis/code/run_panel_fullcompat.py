"""Run source and episodic TTA without importing the ground-truth scorer."""
import argparse,pathlib,sys,json,hashlib,time,os,importlib.metadata
ROOT=pathlib.Path(__file__).resolve().parents[1];UP=ROOT/'upstream_fullcompat'
sys.path[:0]=[str(ROOT/'detectron2'),str(UP),str(UP/'mask2former/modeling/pixel_decoder/ops')]
import numpy as np,torch
import pycocotools.mask as mu
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import transforms as T,detection_utils as DU
from mask2former import add_maskformer2_config
from avism import add_avism_config
from tta_adapter import ClipHeadTTA

def sha(path):return hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/AVISM_R50_IN.pth'));p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--method',choices=['source','views','entropy','consistency','zero_features','cyclic_audio'],required=True);p.add_argument('--steps',type=int,default=0);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--device',default='cuda');args=p.parse_args()
 plan_path=pathlib.Path(args.plan);plan=json.loads(plan_path.read_text());assert plan_path.with_suffix(plan_path.suffix+'.sha256').read_text().split()[0]==sha(plan_path)
 assert {'method':args.method,'steps':args.steps} in plan['arms'] and args.lr==plan['optimizer']['lr']
 for package,version in plan['critical_package_versions'].items():
  try:actual=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:actual=None
  assert actual==version,(package,version,actual)
 out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
 manifest=pathlib.Path(args.manifest);spec=json.loads(manifest.read_text());checkpoint=pathlib.Path(args.checkpoint)
 assert json.loads(checkpoint.with_suffix('.pth.download.json').read_text())['state']=='completed'
 assert str(manifest)==plan['manifest'] and sha(manifest)==plan['manifest_sha256']
 assert str(checkpoint)==plan['checkpoint'] and sha(checkpoint)==plan['checkpoint_sha256']
 for file,expected in plan['source_hashes'].items():assert sha(ROOT/file)==expected,file
 for file,expected in plan['input_hashes'].items():assert sha(file)==expected,file
 cfg=get_cfg();add_deeplab_config(cfg);add_maskformer2_config(cfg);add_avism_config(cfg);cfg.merge_from_file(str(UP/'configs/avism/R50/avism_R50_IN.yaml'));cfg.MODEL.DEVICE=args.device;cfg.MODEL.WEIGHTS='';cfg.MODEL.AVISM.TEST_RUN_CHUNK_SIZE=4;cfg.freeze()
 torch.manual_seed(20261029);np.random.seed(20261029);torch.set_num_threads(4)
 model=build_model(cfg);state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 state=state['model'] if 'model' in state else state
 missing,unexpected=model.load_state_dict(state,strict=False)
 # Buffers may differ with modern framework; do not continue with missing model weights.
 allowed={'pixel_mean','pixel_std'}
 if any(k not in allowed for k in missing) or unexpected:raise RuntimeError({'missing':missing,'unexpected':unexpected})
 model.eval();model.requires_grad_(False)
 adapter=ClipHeadTTA(model.avism_module,'source' if args.method in ['zero_features','cyclic_audio'] else args.method,args.steps,args.lr)
 assert adapter.names==plan['parameters'] and sum(p.numel() for p in adapter.params)==plan['parameter_count']
 protocol={'plan_sha256':sha(plan_path),'args':vars(args),'manifest_sha256':sha(manifest),'checkpoint_sha256':sha(checkpoint),'config':cfg.dump(),'upstream_commit':os.popen(f'git -C {UP} rev-parse HEAD').read().strip(),'tta_sha256':sha(ROOT/'code/tta_adapter.py'),'predictor_sha256':sha(__file__),'missing_keys':missing,'unexpected_keys':unexpected,'annotation_access':'no ground-truth files imported by predictor','state':'running'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 predictions=[];records=[];resize=T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MAX_SIZE_TEST,'choice')
 categories=sorted(spec['categories'],key=lambda c:c['id'])
 for video_index,v in enumerate(spec['videos']):
  frames=[]
  for f in v['image_paths']:
   image=DU.read_image(f,format=cfg.INPUT.FORMAT);image=resize.get_transform(image).apply_image(image);frames.append(torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))))
  audio=np.load(v['audio_path'])[v['frame_indices']].astype('float32');audio_control={'method':args.method,'original_feature_l2':float(np.linalg.norm(audio))}
  if args.method=='zero_features':audio=np.zeros_like(audio);audio_control['interpretation']='zero VGGish-feature ablation; not a recording of silence'
  elif args.method=='cyclic_audio':
   donor=spec['videos'][(video_index+1)%len(spec['videos'])];donor_audio=np.load(donor['audio_path'])[donor['frame_indices']].astype('float32');donor_indices=np.floor(np.arange(len(audio))*len(donor_audio)/len(audio)).astype(int);audio=donor_audio[donor_indices];audio_control.update({'donor_video_id':donor['video_id'],'donor_selected_frame_positions':donor_indices.tolist(),'interpretation':'fixed next-video cyclic feature replacement; no GT/category matching'})
  inputs={'image':frames,'audio':audio,'height':v['height'],'width':v['width'],'length':len(frames),'video_id':v['video_id']}
  if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
  tic=time.perf_counter()
  with torch.no_grad():result=model([inputs])
  if args.device.startswith('cuda'):torch.cuda.synchronize()
  elapsed=time.perf_counter()-tic
  for s,l,m in zip(result['pred_scores'],result['pred_labels'],result['pred_masks']):
   segs=[]
   for mask in m.numpy():
    rle=mu.encode(np.asfortranarray(mask.astype('uint8')));rle['counts']=rle['counts'].decode();segs.append(rle)
   predictions.append({'video_id':v['video_id'],'score':s,'category_id':categories[l]['id'],'segmentations':segs})
  record={'video_id':v['video_id'],'frames':len(frames),'seconds':elapsed,'predicted_tracks':len(result['pred_scores']),'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,'tta':adapter.records[-1] if adapter.records else {'method':'source'},'audio_control':audio_control};records.append(record);print(json.dumps(record),flush=True)
  (out/'progress.json').write_text(json.dumps(records,indent=2))
 predfile=out/'predictions.json';predfile.write_text(json.dumps(predictions));protocol.update({'state':'sealed','prediction_sha256':sha(predfile),'videos':len(records),'frames':sum(x['frames'] for x in records),'predicted_tracks':len(predictions)});(out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('SEALED',protocol['prediction_sha256'],flush=True)
if __name__=='__main__':main()
