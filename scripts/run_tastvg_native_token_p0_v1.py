"""Observe old A native attention, score fixed old nine candidates; no update."""
import os,sys,time,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_routed_common_v1 import *
def run():
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
 from vg_tta.tastvg_native_token_binding_v1 import score
 from methods.decota_final_simplified_v1.tensors import state_hash
 verify();assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['new_R_cells']==768
 lock=read(BASE/'TOKEN_RUNTIME_LOCK.json')
 pins=dict(lock['pins'])
 for rev in sorted((BASE/'token/revisions').glob('*.json')):pins.update(read(rev)['pin_overrides'])
 for f,h in pins.items():assert sha(ROOT/f)==h,f
 for f,h in lock['inputs'].items():assert sha(ROOT/f)==h,f
 install_clean_loader();sys.addaudithook(guard);fh=lease();tick=time.monotonic();done=0
 torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 from scripts.run_spatial_regression_alignment_v1 import model_load
 for ds in DATASETS:
  model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());cb=read(POOL/ds/'CAPTURE_BARRIER.json');assert mh==cb['checkpoint_state_sha256']
  qp=read(QUAL/ds/'PLAN.json');checks=[]
  layer=model.ground_decoder.decoder.layers[-1];sm=layer.cross_attn if layer.cross_attn is not None else layer.cross_attn_image;tm=model.ground_decoder.time_decoder.layers[-1].cross_attn_image
  for c in qp['cells']:
   f=BASE/'token'/ds/f'{c["cell"]:03}.pt';rf=f.with_suffix('.json')
   if rf.exists():assert sha(f)==read(rf)['sha256'];done+=1;continue
   budget();r=read(POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json');assert sha(POOL/ds/r['cache'])==r['sha256'];data=device_tree(load(POOL/ds/r['cache']),'cuda');old=load(ROOT/c['payload']);state=device_tree(old['pre_state'],'cuda');assert state_hash(state)==c['pre_state_sha256'] and data['pixel_sha256']==c['pixel_sha256']
   attention={'spatial':[],'temporal':[]};hooks=[]
   for branch,mod in [('spatial',sm),('temporal',tm)]:
    hooks.append(mod.register_forward_hook(lambda mod,args,out,branch=branch:attention[branch].append(out[1].detach().cpu())))
   try:_,_,pre=predict(model,data,state)
   finally:
    for h in hooks:h.remove()
   assert torch.equal(pre['boxes'].cpu(),old['slow']['boxes']) and pre['indices']==old['slow']['indices']
   assert all(torch.equal(a.cpu(),b) for a,b in zip(pre['logits'],old['update_steps'][0]['prediction']['logits']))
   assert len(attention['spatial'])==len(attention['temporal'])==4
   second={k:[v[1].numpy(),v[3].numpy()] for k,v in attention.items()};cache=load(POOL/ds/r['cache']);tubes=np.stack([x['prediction']['boxes'].numpy() for x in old['update_steps'][0]['candidates']]);scores=score(cache['views'],second,tubes,old['slow']['indices'])
   value=dict(cell=c['cell'],source_id=c['source_id'],parent=c['parent'],condition=c['condition'],order=c['order'],arrival=c['arrival'],pre_state_sha256=c['pre_state_sha256'],pixel_sha256=c['pixel_sha256'],capture_sha256=r['sha256'],attention=second,scores=scores,prediction_bitwise_parity=True,parameter_update=False,GT_read=False)
   save(f,value);write(rf,dict(sha256=sha(f),cell=c['cell'],GT_read=False,parity=True));done+=1;checks.append(dict(cell=c['cell'],parity=True))
   status(BASE/'token/STATUS.json',dict(status='running_native_token',dataset=ds,done=done,total=60,worker_pid=os.getpid(),time=time.time()));print('TOKEN',ds,c['cell'],done,60,flush=True)
   del data,state,old,attention,scores,cache,value;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.state_dict())==mh;del model;gc.collect();torch.cuda.empty_cache()
 assert done==60;write(BASE/'token/PREDICTION_BARRIER.json',dict(cells=60,candidates=540,GT_read=False,parameter_updates=0,files={str(f.relative_to(BASE)):sha(f) for f in (BASE/'token').glob('*/*.json')},time=time.time()))
 write(BASE/'token/RESOURCES.json',dict(worker_wall_seconds=time.monotonic()-tick,native_suffix_replays=60,new_backbone_forwards=0,new_experts=0,parameter_updates=0,peak_vram_bytes=torch.cuda.max_memory_allocated(),GT_read=False))
 status(BASE/'token/STATUS.json',dict(status='sealed_pending_root_GT',done=60,total=60,time=time.time()));fh.close()
if __name__=='__main__':
 try:run()
 except BaseException as e:write(BASE/'token'/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));status(BASE/'token/STATUS.json',dict(status='failed',error=repr(e),time=time.time()));raise
