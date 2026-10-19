"""Cohort-independent, episodic HC research predictor. No scorer or roster imports."""
import gc,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import prediction
from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
from .runtime import ROOT,parse,load_source
from ._source_forward import frozen_forward
from .initialization import encode,estimate,apply
from .temporal import probabilities,mass_table,final_mbr
from .experts import positions8
from .teacher_bridge import refer_native
from .state import Replay
from .policy import fit
from .direct import fit as direct_fit

class ReferDINOOPDPredictor:
 def __init__(self,config=None,*,check=lambda:None,source=None):
  self.config=json.loads((Path(__file__).with_name('config.json')).read_text())if config is None else dict(config)
  self.check=check;self.source=source;self.loads={}
 def _source(self):
  if self.source is None:
   t=time.perf_counter();self.source=load_source();self.loads['Source_load_seconds']=time.perf_counter()-t
  return self.source.cuda().eval().requires_grad_(False)
 def capture(self,frames,frame_ids,metadata,*,parses=None):
  from methods.decota_final_simplified_v1.backbone import validate_input
  validate_input(frames,frame_ids,metadata);assert len(frame_ids)>=8
  t=time.perf_counter();parsing=0.
  if parses is None:parses,parsing=parse(metadata['caption'])
  row=dict(input=metadata,frame_ids=list(frame_ids),parses=parses);warm=self.source is not None;model=self._source();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  batch,recs,base=frozen_forward(model,frames,row);native=prediction(base.zero['logits'],base.zero['boxes'],recs,frame_ids);torch.cuda.synchronize();source_s=time.perf_counter()-start
  inputs=detached(base.spatial_inputs,'cpu');src=native['boxes'].detach().cpu();logits=detached(base.zero['logits'],'cpu');del batch,base;model.cpu();gc.collect();torch.cuda.empty_cache()
  start=time.perf_counter();expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);torch.cuda.synchronize();dino_load=time.perf_counter()-start
  start=time.perf_counter();ex=observations(expert,parses,frames,frame_ids,native['indices'],audit=True);torch.cuda.synchronize();dino_s=time.perf_counter()-start;del expert;gc.collect();torch.cuda.empty_cache()
  start=time.perf_counter();pool=[]
  for a in ex['anchors']['single4']:
   pos=a['position'];z=ex['observations'][('original',pos)]['probe'];assert z['accepted'];pool.append(dict(position=pos,boxes=np.asarray(z['boxes'],float).reshape(-1,4),scores=np.asarray(z['target_scores'],float)))
  calibration=self.config['E_calibration'];pools=[encode(v['boxes'])-encode(src.double().numpy())[v['position']]for v in pool];est=estimate(pools,[v['scores']for v in pool],calibration['scale'],calibration['truncation'],1/16);E=apply(src.double().numpy(),est['delta']);U,q=mass_table(probabilities(recs,logits));assert U==list(frame_ids);interval=final_mbr(U,q)['interval'];p4=ex['positions4'];p8=positions8(U,p4);init_s=time.perf_counter()-start
  out=refer_native(frames[p8],metadata['caption']);ref_load=out['model_load_seconds'];gc.collect();torch.cuda.empty_cache()
  cost=dict(parse_seconds=parsing,Source_forward_seconds=source_s,DINO_load_seconds=dino_load,DINO_observation_seconds=dino_s,DINO_image_forwards=ex['new_DINO'],E_MBR_seconds=init_s,ReferDINO_load_seconds=ref_load,ReferDINO_seconds=out['cost']['expert_seconds'],ReferDINO_process_wall_seconds=out['process_wall_seconds'],ReferDINO_query_forwards=1,ReferDINO_image_frames=8,Source_queries=1,Source_offset_forwards=2,capture_complete_seconds=time.perf_counter()-t,peak_GPU_bytes=max(torch.cuda.max_memory_allocated(),out['peak_GPU_bytes']),Source_load_seconds=0. if warm else self.loads.get('Source_load_seconds',0.))
  return dict(Source=src,E=E,frame_ids=U,interval=interval,planned_positions=p8,original_positions4=p4,spatial_inputs=inputs,records=detached(recs,'cpu'),native_interval=native['physical_interval'],expert=ex,teacher_native=out,cost=cost,GT_read=False)
 def adapt(self,cache,*,key,arms=('Direct','Fixed','Live'),seeds=None):
  self.check();model=self._source();rp=Replay(model,cache,cache['Source'],cache['E'],cache['planned_positions'],self.check);target=cache['teacher_native']['boxes'];fits={};outputs=dict(Source_MBR=cache['Source'].double().numpy(),E_MBR=cache['E']);cfg=self.config;seeds=cfg['seeds']if seeds is None else seeds
  try:
   for arm in arms:
    for seed in ([None]if arm=='Direct'else seeds):
     out=direct_fit(rp,target,key,arm,seed,cfg)if arm=='Direct'else fit(rp,target,key,arm,seed,cfg);name=arm if arm=='Direct'else f'{arm}_{seed}';fits[name]=out;outputs[name]=out['final'].numpy()
   assert torch.equal(rp.state.detach(),torch.zeros_like(rp.state))
   return dict(outputs=outputs,fits=fits,frame_ids=cache['frame_ids'],interval=cache['interval'],planned_positions=cache['planned_positions'],GT_read=False,cost=dict(trajectory_decoder_calls=rp.calls,offset_decoder_calls=2*rp.calls,actual_updates=rp.backwards,fit_seconds={k:v['seconds']for k,v in fits.items()},temporary_parameters=rp.state.numel(),network_weight_updates=0))
  finally:del rp;gc.collect();torch.cuda.empty_cache()
 def predict(self,frames,frame_ids,metadata,*,key=None,cache=None,parses=None,seed=None):
  tick=time.perf_counter();reused=cache is not None;key=key or hashlib.sha256((metadata['caption']+str(frame_ids)).encode()).hexdigest()
  if cache is None:cache=self.capture(frames,frame_ids,metadata,parses=parses)
  else:assert cache['frame_ids']==list(frame_ids)and not cache['GT_read']
  result=self.adapt(cache,key=key,arms=('Live',),seeds=[self.config['default_seed']if seed is None else seed]);result['sampling_key']=key;result['boxes']=next(v for k,v in result['outputs'].items()if k.startswith('Live_'));result['cost'].update(cache['cost']);result['cost']['complete_predict_seconds']=time.perf_counter()-tick;result['cache_reused']=reused
  return result
