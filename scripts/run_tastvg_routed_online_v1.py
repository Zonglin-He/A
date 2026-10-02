"""Actual on-policy online R, only Sa2VA observation positions changed."""
import os,sys,time,gc,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_routed_common_v1 import *

def run(ds):
 import torch,numpy as np
 from PIL import Image
 from transformers import AutoModel,AutoTokenizer
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease as gpu_lease
 from scripts.run_tastvg_full_b1_experts_v1 import observation
 from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
 from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes
 from vg_tta.tastvg_event_support_v1 import OnlineMethod,rollout_states
 from vg_tta.tastvg_selected_rollout_v1 import central_with_candidates
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state
 from vg_tta.tastvg_reference_selection_v1 import student_frames
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta import exact_frame_decode_audit_v2 as binding
 from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hc_decode
 from vg_tta.exact_frame_decode_audit_v2 import decode as vid_decode
 p=verify(ds);cfg=p['params'];out=BASE/ds/'R';bar=read(POOL/ds/'CAPTURE_BARRIER.json');tick=time.monotonic();actor=lease=None;done=0
 try:
  install_clean_loader();sys.addaudithook(guard);lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  binding.decode=hc_decode if ds=='hc2' else vid_decode
  from scripts.run_spatial_regression_alignment_v1 import model_load
  model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==bar['checkpoint_state_sha256']
  center=central_state(model);states,spec,basis=rollout_states(center,cfg['rho'],cfg['direction_count']);deltas=[{n:v-center[n] for n,v in z.items()} for z in states]
  def new_actor():return OnlineMethod(model,deltas,lr=cfg['lr'],teacher_temperature=cfg['teacher_temperature'],student_temperature=cfg['student_temperature'],steps=cfg['steps'],basis=basis,**read(out/'REQUEST.json')['method'])
  def data_at(parent,cond):
   rf=POOL/ds/'capture'/cond/f'{parent:05}.json';assert sha(rf)==bar['files'][str(rf.relative_to(POOL/ds))];r=read(rf);f=POOL/ds/r['cache'];assert sha(f)==r['sha256'];return device_tree(load(f),'cuda'),r
  def old_expert(stage,parent,cond,pixel):
   r=read(POOL/ds/'experts'/stage/cond/f'{parent:05}.json');f=POOL/ds/'experts'/r['cache'];assert sha(f)==r['cache_sha256'] and r['pixel_sha256']==pixel;return {**load(f),'pixel_sha256':pixel}
  # A exact live control: a write then a future nonexpert under the new runtime.
  actor=new_actor();parity=[];seq=p['splits']['search']['orders']['order1']
  for at,parent in enumerate(seq[:2]):
   data,cr=data_at(parent,'clean');x,_=actor.arrive(data,at%4==0,lambda:old_expert('temporal',parent,'clean',cr['pixel_sha256']),lambda:old_expert('spatial',parent,'clean',cr['pixel_sha256']))
   old=load(PRIOR/ds/'A/online/clean/order1'/f'{at:05}.pt')
   assert x['pre_state_sha256']==old['pre_sha'] and x['post_state_sha256']==old['post_sha']
   assert torch.equal(x['prediction']['boxes'],old['slow']['boxes']) and x['output_prediction']['indices']==old['final_indices']
   for a,b in zip(x['update_steps'],old['update_steps']):
    assert a['pre_state_sha256']==b['pre_state_sha256'] and a['post_state_sha256']==b['post_state_sha256']
    if a['update']:
     assert a['update']['loss_before']==b['update']['loss_before']
     for n,g in a['update']['gradients'].items():assert torch.equal(g,b['update']['gradients'][n])
   parity.append(dict(arrival=at,state_prediction_gradient_bitwise=True));del data,x;gc.collect();torch.cuda.empty_cache()
  actor.close();actor=None;assert state_hash(model.state_dict())==mh
  mp=ROOT/'checkpoints/Sa2VA-4B'
  for r in read(mp/'DOWNLOAD_RECEIPT.json')['files']:assert sha(mp/r['file'])==r['sha256']
  for f,h in read(mp/'OFFICIAL_CODE_RECEIPT.json')['files'].items():assert sha(mp/f)==h['sha256']
  expert,info=AutoModel.from_pretrained(str(mp),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,trust_remote_code=True,local_files_only=True,output_loading_info=True)
  assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
  expert=expert.eval().cuda().requires_grad_(False);tokenizer=AutoTokenizer.from_pretrained(str(mp),trust_remote_code=True,use_fast=False,local_files_only=True)
  expert.preparing_for_generation(tokenizer,max_new_tokens=256,torch_dtype=torch.bfloat16);assert all(float(m.fill_hole_area)==0 for m in expert.modules() if hasattr(m,'fill_hole_area'))
  registry={};calls=[]
  for f in (QUAL/ds/'receipts').glob('*student_routed.json'):
   r=read(f);cf=QUAL/r['cache'];assert sha(cf)==r['cache_sha256'];registry[r['input_sha256']]=(cf,'prior_qualification',sha(f))
  for f in (BASE/ds/'expert_receipts').glob('*.json'):
   r=read(f);cf=BASE/r['cache'];assert sha(cf)==r['cache_sha256'];registry[r['input_sha256']]=(cf,'this_run',sha(f))
  def acquire(row,cond,pixel,pos,key,parity_old=None):
   budget();frames,ids=binding.decode(row['input']);assert ids==row['frame_ids'];shifted,ph,_=observation(row,cond,frames);assert ph==pixel
   samples=[shifted[i] for i in pos];digest=hashlib.sha256(row['input']['caption'].encode()+str(pos).encode()+str(shifted.shape).encode()+b''.join(a.tobytes() for a in samples)).hexdigest()
   if digest in registry and parity_old is None:
    cf,origin,receipt_hash=registry[digest];value=load(cf);receipt=dict(cache=str(cf.relative_to(ROOT)),cache_sha256=sha(cf),input_sha256=digest,pixel_sha256=pixel,positions=pos,origin=origin,original_receipt_sha256=receipt_hash,new_call=False,seconds=0.,GT_read=False)
   else:
    attempted=len(list(BASE.glob('*/attempts/*.json')));assert attempted<read(BASE/'RUNTIME_LOCK.json')['max_new_specialist_calls']
    af=BASE/ds/'attempts'/f'{key}.json';assert not af.exists(),'Saved unreceipted attempt needs root review'
    write(af,dict(key=key,input_sha256=digest,pixel_sha256=pixel,positions=pos,GT_read=False,time=time.time()))
    start=time.monotonic();prompt='<image>Please segment the object described by: '+row['input']['caption'].rstrip('.')+'.'
    with torch.inference_mode():raw=expert.predict_forward(video=[Image.fromarray(a) for a in samples],text=prompt,tokenizer=tokenizer)
    masks=raw['prediction_masks'];mask=masks[0] if len(masks) else None;valid,boxes=mask_boxes(mask,pos,len(ids))
    value=dict(valid=valid,boxes=boxes,positions=pos,mask_count=len(masks),mask_shape=list(np.asarray(mask).shape) if mask is not None else None,mask_bits=np.packbits(np.asarray(mask,dtype=bool)) if mask is not None else None,prediction_text=raw['prediction'],input_sha256=digest,GT_read=False)
    if parity_old:
     old=load(parity_old);assert value['input_sha256']==old['input_sha256'] and value['prediction_text']==old['prediction_text'] and value['mask_shape']==old['mask_shape']
     for k in ['valid','boxes','mask_bits']:assert np.array_equal(value[k],old[k]),k
    cf=BASE/ds/'expert_cache'/f'{key}.pt';save(cf,value)
    receipt=dict(cache=str(cf.relative_to(ROOT)),cache_sha256=sha(cf),input_sha256=digest,pixel_sha256=pixel,positions=pos,origin='new_inference',new_call=True,seconds=time.monotonic()-start,uniform_bitwise_parity=bool(parity_old),GT_read=False)
    registry[digest]=(cf,'this_run',None)
   receipt.update(valid_frames=int(value['valid'].sum()),mask_count=value['mask_count']);write(BASE/ds/'expert_receipts'/f'{key}.json',receipt);calls.append(receipt)
   return {**value,'pixel_sha256':pixel},receipt
  parent=seq[0];row=p['rows'][parent];r=read(POOL/ds/'experts/spatial/clean'/f'{parent:05}.json');pos=load(POOL/ds/'experts'/r['cache'])['positions']
  if not (BASE/ds/'expert_receipts/uniform_parity.json').exists():acquire(row,'clean',r['pixel_sha256'],pos,'uniform_parity',POOL/ds/'experts'/r['cache'])
  write(BASE/ds/'SMOKE_ROOT_EVIDENCE.json',dict(status='pass',A_clean_arrivals=parity,uniform_specialist_bitwise_parity=True,checkpoint_state_sha256=mh,GT_read=False,time=time.time()))
  write(out/'SUPPORT.json',dict(spec=spec,center_sha256=state_hash(center),params=cfg));extra=0;previous=None
  for cond in p['conditions']:
   for order,seq in p['splits']['search']['orders'].items():
    actor=new_actor();previous=state_hash(actor.actor.initial);last=None
    for at,parent in enumerate(seq):
     budget();f=out/'online'/cond/order/f'{at:05}.pt';rf=f.with_suffix('.json')
     if rf.exists():
      rr=read(rf);assert sha(f)==rr['sha256'] and rr['pre_sha']==previous and rr['parent']==parent;previous=rr['post_sha'];last=f;done+=1;continue
     if last:actor.actor.restore(device_tree(load(last)['post_state'],'cuda'));last=None
     data,cr=data_at(parent,cond);assert state_hash(actor.actor.state())==previous;scheduled=at%4==0;observed=[];routing=None;sr=None
     if scheduled:
      with torch.no_grad():_,_,ref,_,tc=central_with_candidates(actor.actor,data)
      routing=student_frames(tc,data['frame_ids']);extra+=1
     def tp():observed.append('temporal');return old_expert('temporal',parent,cond,cr['pixel_sha256'])
     def sp():
      nonlocal sr
      observed.append('spatial');ev,sr=acquire(p['rows'][parent],cond,cr['pixel_sha256'],routing['positions'],f'{cond}_{order}_{at:05}');return ev
     x,_=actor.arrive(data,scheduled,tp,sp);assert observed==(['temporal','spatial'] if scheduled else [])
     if scheduled:
      assert x['temporal']['candidates']==tc and torch.equal(x['prediction']['boxes'],ref['boxes']);assert len(routing['positions'])==len(set(routing['positions']))==5
     assert x['pre_state_sha256']==previous and torch.isfinite(x['output_prediction']['boxes']).all() and all(torch.isfinite(v).all() for v in x['post_state'].values())
     payload=dict(arm='R',parent=parent,condition=cond,order=order,arrival=at,expert_scheduled=scheduled,updated=x['updated'],pixel_sha256=cr['pixel_sha256'],pre_state=x['pre_state'],post_state=x['post_state'],pre_sha=x['pre_state_sha256'],post_sha=x['post_state_sha256'],update=x.get('update'),update_steps=x['update_steps'],compute=x['compute'],source_native=compact_prediction(data['prediction']),slow=compact_prediction(x['prediction']),post_prediction=compact_prediction(x['post_prediction']),final_indices=x['output_prediction']['indices'],temporal=x.get('temporal'),reinsertion=None,routing=routing,routed_expert=sr,GT_read=False)
     save(f,payload);write(rf,dict(sha256=sha(f),parent=parent,pre_sha=payload['pre_sha'],post_sha=payload['post_sha'],updated=x['updated']));previous=payload['post_sha'];done+=1
     status(out/'STATUS.json',dict(status='running',done=done,total=384,condition=cond,order=order,worker_pid=os.getpid(),seconds=time.monotonic()-tick));print('ROUTED',ds,done,384,cond,order,flush=True)
     del data,x,payload;gc.collect();torch.cuda.empty_cache()
    actor.close();actor=None;assert state_hash(model.state_dict())==mh
  assert done==384;verify();write(out/'PREDICTION_BARRIER.json',dict(cells=done,GT_read=False,model_restored=True,files={str(f.relative_to(out)):sha(f) for f in (out/'online').rglob('*.json')},time=time.time()))
  write(BASE/ds/'RESOURCES.json',dict(worker_wall_seconds=time.monotonic()-tick,wall_includes_loading_IO=True,extra_routing_suffix_replays=extra,uniform_parity_calls=1,live_A_parity_arrivals=2,peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),expert_receipts=len(list((BASE/ds/'expert_receipts').glob('*.json'))),new_specialist_calls=len(list((BASE/ds/'attempts').glob('*.json'))),GT_read=False))
  status(out/'STATUS.json',dict(status='completed',done=done,total=done,seconds=time.monotonic()-tick))
 finally:
  if actor:actor.close()
  if lease:lease.close()

if __name__=='__main__':
 try:run(sys.argv[1])
 except BaseException as e:
  ds=sys.argv[1];write(BASE/ds/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));status(BASE/ds/'R/STATUS.json',dict(status='failed',error=repr(e),worker_pid=os.getpid(),time=time.time()));raise
