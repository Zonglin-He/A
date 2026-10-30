"""Sealed video-disjoint AVIS experiments; predictor never opens annotation files."""
import argparse, pathlib,sys,json,hashlib,time,contextlib,copy,math,os
R=pathlib.Path(__file__).resolve().parents[1];A=R.parent;UP=A/'upstream_fullcompat'
sys.path[:0]=[str(A/'detectron2'),str(UP),str(UP/'mask2former/modeling/pixel_decoder/ops'),str(A/'scripts')]
import numpy as np,torch
import torch.nn.functional as F
import pycocotools.mask as mu
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import transforms as T,detection_utils as DU
from mask2former import add_maskformer2_config
from avism import add_avism_config
from tta_adapter import ClipHeadTTA
from latent_adapter import VideoTrace,query_latent,adapt_latent,pair_matches

def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def build(device):
 cfg=get_cfg();add_deeplab_config(cfg);add_maskformer2_config(cfg);add_avism_config(cfg);cfg.merge_from_file(str(UP/'configs/avism/R50/avism_R50_IN.yaml'));cfg.MODEL.DEVICE=device;cfg.MODEL.WEIGHTS='';cfg.MODEL.AVISM.TEST_RUN_CHUNK_SIZE=4;cfg.freeze()
 torch.manual_seed(20261030);np.random.seed(20261030);torch.set_num_threads(4)
 model=build_model(cfg);state=torch.load(A/'checkpoints/AVISM_R50_IN.pth',weights_only=True,map_location='cpu',mmap=True)['model'];model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False)
 return model,cfg

def timed(model,inputs,latent=None,trace=None):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tic=time.perf_counter()
 with torch.no_grad(),(query_latent(model,latent) if latent is not None else contextlib.nullcontext()),(trace if trace is not None else contextlib.nullcontext()):res=model([inputs])
 torch.cuda.synchronize();return res,{'seconds':time.perf_counter()-tic,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}

def raw_sensitivity(left,right):
 result={}
 for k in ['pred_logits','pred_mask_embed','pred_cq_embed']:
  x,y=left.clip[k],right.clip[k]
  result[k]={'relative_l2':float((x-y).norm()/(x.norm()+1e-8)),'max_abs':float((x-y).abs().max())}
 x,y=left.audio_query,right.audio_query
 result['early_audio_query']={'relative_l2':float((x-y).norm()/(x.norm()+1e-8)),'max_abs':float((x-y).abs().max())}
 return result

def result_sensitivity(a,b):
 # Class+mask matching: do not treat slots/order as instance IDs.
 if not len(a['pred_scores']) or not len(b['pred_scores']):return {'tracks_left':len(a['pred_scores']),'tracks_right':len(b['pred_scores']),'matched_mean_iou':None}
 from scipy.optimize import linear_sum_assignment
 am=a['pred_masks'].flatten(1).float();bm=b['pred_masks'].flatten(1).float();inter=am@bm.T;iou=inter/(am.sum(-1)[:,None]+bm.sum(-1)[None,:]-inter+1e-8)
 compat=torch.tensor(a['pred_labels'])[:,None]==torch.tensor(b['pred_labels'])[None,:];row,col=linear_sum_assignment(((1-iou)+2*(~compat)).numpy())
 return {'tracks_left':len(am),'tracks_right':len(bm),'matched_mean_iou':float(iou[row,col].mean()),'matched_class_agreement':float(compat[row,col].float().mean())}

def direct_fusion(model,trace,inputs):
 """Same frozen heads, direct probability blend, ordinary source score readout."""
 device=model.device;probs=trace.clip['pred_logits'][-1,0].softmax(-1);embed=trace.clip['pred_mask_embed'][-1,0]
 scores=probs[:,:-1];num_classes=scores.shape[1]
 values,idx=scores.flatten().topk(model.test_topk_per_image,sorted=False);labels=torch.arange(num_classes,device=device)[None].repeat(model.num_queries,1).flatten()[idx];query_idx=torch.div(idx,num_classes,rounding_mode='floor')
 candidates=((probs.argmax(-1)<num_classes)&(scores.amax(-1)>=.3)).nonzero().flatten()
 interim_size=tuple(s*4 for s in trace.projected.shape[-2:]);image_size=tuple(inputs['image'][0].shape[-2:]);out_size=(inputs['height'],inputs['width'])
 numerator=torch.zeros(len(idx),device=device);denominator=torch.zeros(len(idx),device=device);masks=[];matches=[];global_logits=[]
 for t in range(len(trace.frames)):
  mf=trace.projected[t].to(device);all_masks=torch.einsum('qc,chw->qhw',embed,mf)
  frame=trace.frames[t];pairs=pair_matches(frame['logits'].to(device),frame['masks'].to(device),probs,all_masks,candidates);mapping={cq:fq for cq,fq,_ in pairs};matches.append(len(pairs))
  pred=all_masks[query_idx].clone()
  for j,q in enumerate(query_idx.tolist()):
   if q in mapping and int(labels[j])==int(probs[q,:-1].argmax()):
    soft=(pred[j].sigmoid()+frame['masks'][mapping[q]].to(device).sigmoid())*.5
    pred[j]=torch.logit(soft.clamp(1e-6,1-1e-6))
  global_logits.append(pred)
 # Match native readout interpolation batch size and summation order.
 for start in range(0,len(global_logits),model.test_interpolate_chunk_size):
  pred=torch.stack(global_logits[start:start+model.test_interpolate_chunk_size],dim=1)
  pred=F.interpolate(pred,size=interim_size,mode='bilinear',align_corners=False)[:,:,:image_size[0],:image_size[1]]
  soft=pred.sigmoid();hard=soft>.5;numerator+=(soft.flatten(1)*hard.flatten(1)).sum(1);denominator+=hard.flatten(1).sum(1)
  masks.append((F.interpolate(pred,size=out_size,mode='bilinear',align_corners=False)>0).cpu())
 values=values*numerator/(denominator+1e-6);keep=(values>.3).nonzero().flatten()
 return {'image_size':out_size,'pred_scores':values[keep].tolist(),'pred_labels':labels[keep].tolist(),'pred_masks':torch.cat(masks,dim=1)[keep.cpu()]},{'frame_match_counts':matches,'fusion_weight':.5,'scope':'same teacher, all-frame local proposals; no updates; native readout quality recomputed'}

def encode(result,video,categories):
 rows=[]
 for s,label,masks in zip(result['pred_scores'],result['pred_labels'],result['pred_masks']):
  seg=[]
  for mask in masks.numpy():
   r=mu.encode(np.asfortranarray(mask.astype('uint8')));r['counts']=r['counts'].decode();seg.append(r)
  rows.append({'video_id':video['video_id'],'score':s,'category_id':categories[label]['id'],'segmentations':seg})
 return rows

def load_inputs(v,cfg):
 resize=T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MAX_SIZE_TEST,'choice');frames=[]
 for p in v['image_paths']:
  image=DU.read_image(p,format='RGB');image=resize.get_transform(image).apply_image(image);frames.append(torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))))
 return {'image':frames,'audio':np.load(v['audio_path'])[v['frame_indices']].astype('float32'),'height':v['height'],'width':v['width'],'length':len(frames),'video_id':v['video_id']}

def run(args):
 planpath=pathlib.Path(args.plan);plan=json.loads(planpath.read_text());assert planpath.with_suffix('.json.sha256').read_text().split()[0]==sha(planpath)
 manifest=pathlib.Path(plan['manifest']);assert sha(manifest)==plan['manifest_sha256'];spec=json.loads(manifest.read_text())
 for file,h in plan['source_hashes'].items():assert sha(file)==h,file
 for file,h in plan['input_hashes'].items():assert sha(file)==h,file
 out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
 model,cfg=build('cuda');categories=sorted(spec['categories'],key=lambda c:c['id']);predictions={arm:[] for arm in plan['arms']};records=[]
 protocol={'state':'running','plan_sha256':sha(planpath),'manifest_sha256':sha(manifest),'args':vars(args),'config':cfg.dump(),'annotation_access':'predictor imports no GT file; metadata-only manifest','prediction_policy':'all arms seal before scoring; no optional output replacement'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for vi,v in enumerate(spec['videos']):
  inp=load_inputs(v,cfg);anchors=np.linspace(0,len(inp['image'])-1,8,dtype=int).tolist();anchors=sorted(set(anchors));row={'video_id':v['video_id'],'anchors':anchors,'arms':{},'audio_diagnostics':{}}
  trace=VideoTrace(model,anchors);source,cost=timed(model,inp,trace=trace);predictions['source'].extend(encode(source,v,categories));row['arms']['source']=cost
  # Baseline bridge on the FIRST video, before any adaptation.
  if vi==0:
   zero=torch.zeros((100,256),device=model.device);zt=VideoTrace(model,[],full=False);zres,zcost=timed(model,inp,latent=zero,trace=zt)
   exact=all(torch.equal(trace.clip[k],zt.clip[k]) for k in ['pred_logits','pred_mask_embed']) and torch.equal(source['pred_masks'],zres['pred_masks']) and source['pred_scores']==zres['pred_scores']
   row['zero_bridge']={'bitexact_source':exact,**zcost};assert exact,'zero-latent official source bridge failed';del zt,zres
  tic=time.perf_counter();direct,details=direct_fusion(model,trace,inp);predictions['direct_fusion'].extend(encode(direct,v,categories));row['arms']['direct_fusion']={'seconds':time.perf_counter()-tic,**details};del direct
  orig=model.avism_module.forward
  for name,method in [('old_views','views'),('old_LN','entropy')]:
   adapter=ClipHeadTTA(model.avism_module,method,3 if method=='entropy' else 0,1e-4)
   result,c=timed(model,inp);predictions[name].extend(encode(result,v,categories));row['arms'][name]={**c,'tta':adapter.records[-1]};adapter.reset();model.avism_module.forward=orig
   assert all(torch.equal(p,init) for p,init in zip(adapter.params,adapter.initial)),'LN reset failure'
   del result,adapter
  adapted_trace=None;adapted_result=None;full_latent=None
  for name,identity in [('latent_identity',True),('latent_no_identity',False)]:
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tic=time.perf_counter()
   latent,details=adapt_latent(model,trace,anchors,identity=identity,steps=plan['latent']['steps'],lr=plan['latent']['lr']);adapt_peak=torch.cuda.max_memory_allocated();post_trace=VideoTrace(model,[],full=False) if identity else None
   result,c=timed(model,inp,latent=latent,trace=post_trace);torch.cuda.synchronize();total_time=time.perf_counter()-tic
   predictions[name].extend(encode(result,v,categories));row['arms'][name]={'total_seconds':total_time,'peak_cuda_bytes_including_adaptation':max(adapt_peak,c['peak_cuda_bytes']),'post_update_inference':c,'tta':details,'post_update_vs_source':result_sensitivity(source,result)}
   if identity:adapted_trace=post_trace;adapted_result=result;full_latent=latent
   else:del result,latent
  # Hook is gone; original localizer equal to zero hook under same batch1 call.
  t=anchors[0];c=trace.cache[t];p=model.sem_seg_head.predictor
  with torch.no_grad():
   a=p([x.to(model.device) for x in c['multi']],c['mask'].to(model.device),c['clip_mask'].to(model.device),trace.audio[t:t+1])
   with query_latent(model,torch.zeros((100,256),device=model.device)):b=p([x.to(model.device) for x in c['multi']],c['mask'].to(model.device),c['clip_mask'].to(model.device),trace.audio[t:t+1])
  reset_exact=all(torch.equal(a[0][k],b[0][k]) for k in ['pred_logits','pred_masks']);assert reset_exact;row['episodic_reset_zero_hook_exact']=reset_exact
  del a,b
  # These interventions have NO counterfactual GT. They are sensitivity only.
  donor=spec['videos'][(vi+1)%len(spec['videos'])];da=np.load(donor['audio_path'])[donor['frame_indices']].astype('float32');indices=np.floor(np.arange(len(inp['audio']))*len(da)/len(inp['audio'])).astype(int)
  for control,audio in [('cyclic',da[indices]),('shift3',np.roll(inp['audio'],3,axis=0))]:
   alt=dict(inp,audio=audio);bt=VideoTrace(model,[],full=False);br,bc=timed(model,alt,trace=bt);at=VideoTrace(model,[],full=False);ar,ac=timed(model,alt,latent=full_latent,trace=at)
   row['audio_diagnostics'][control]={'interpretation':'prediction sensitivity only; not known negative labels','donor_video_id':donor['video_id'] if control=='cyclic' else None,'source':{**raw_sensitivity(trace,bt),'instances':result_sensitivity(source,br),'cost':bc},'adapted':{**raw_sensitivity(adapted_trace,at),'instances':result_sensitivity(adapted_result,ar),'cost':ac}}
   del bt,br,at,ar
  row['source_to_adapted_raw']=raw_sensitivity(trace,adapted_trace)
  records.append(row);(out/'progress.json').write_text(json.dumps(records,indent=2));print(json.dumps({'video':v['video_id'],'completed':vi+1,'gated_pairs':row['arms']['latent_identity']['tta']['pair_count'],'latent_delta':row['arms']['latent_identity']['tta'].get('latent_delta_l2')}),flush=True)
  del trace,source,adapted_trace,adapted_result,full_latent,inp;torch.cuda.empty_cache()
 for arm,pred in predictions.items():
  d=out/arm;d.mkdir();pf=d/'predictions.json';pf.write_text(json.dumps(pred));(d/'seal.json').write_text(json.dumps({'state':'sealed','plan_sha256':sha(planpath),'prediction_sha256':sha(pf),'manifest_sha256':sha(manifest),'videos':len(spec['videos']),'tracks':len(pred)},indent=2))
 protocol.update(state='sealed',arms={arm:sha(out/arm/'predictions.json') for arm in predictions},videos=len(spec['videos']));(out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('SEALED',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args)
