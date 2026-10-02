"""Replay HC saved A/R states on locked different queries; no update/expert/GT."""
import os,sys,time,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_transfer_aligned_common_v1 import *
def run():
 import torch,numpy as np
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_final_simplification_v1 import lease
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
 lock=verify('TRANSFER');p=read(OLD/'hc2/PLAN.json');cb=read(POOL/'hc2/CAPTURE_BARRIER.json');tick=time.monotonic();done=replays=0
 install_clean_loader();sys.addaudithook(guard);fh=lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 from scripts.run_spatial_regression_alignment_v1 import model_load
 model=model_load('vidstg_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==cb['checkpoint_state_sha256']
 if not (BASE/'transfer/TARGET_LOCK.json').exists():
  vectors=[];parents=sorted(p['splits']['search']['orders']['order1'])
  # Frozen text-only context BEFORE new GT. No multimodal/backbone forward.
  with torch.no_grad():
   for parent in parents:
    text=p['rows'][parent]['input']['caption'];enc=model.text_encoder;tok=enc.tokenizer([text],return_tensors='pt').to('cuda');raw=enc.body(**tok).last_hidden_state[0];keep=tok.attention_mask[0].bool();keep[0]=False;keep[int(tok.attention_mask[0].sum())-1]=False;z=raw[keep].double().mean(0);vectors.append((z/z.norm()).cpu())
  z=torch.stack(vectors);sim=(z@z.T).numpy();assert len(parents)==32
  commit(BASE/'transfer/TEXT_CONTEXT.pt',dict(parents=parents,vectors=z,cosine=sim,checkpoint_state_sha256=mh,GT_read=False));table=[]
  for order,seq in p['splits']['search']['orders'].items():
   for i in range(0,32,4):
    future=list(range(i+1,32));d=seq[i];ix=parents.index(d);scores={j:float(sim[ix,parents.index(seq[j])]) for j in future}
    roles={'self':i,'next':next(j for j in future if j%4!=0),'near':min(future,key=lambda j:(-scores[j],j)),'far':min(future,key=lambda j:(scores[j],j))}
    table.append(dict(order=order,donor_arrival=i,donor_parent=d,roles={k:dict(arrival=j,parent=seq[j],cosine=1. if j==i else scores[j],scheduled_in_original=j%4==0) for k,j in roles.items()},eligible_future=sorted(scores),all_future_cosines=scores))
  write(BASE/'transfer/TARGET_LOCK.json',dict(rows=table,context_sha256=sha(BASE/'transfer/TEXT_CONTEXT.pt'),GT_read=False,tie='earliest future arrival',selection_time=time.time()))
 targets=read(BASE/'transfer/TARGET_LOCK.json');receipt(BASE/'transfer/TEXT_CONTEXT.pt');assert not targets['GT_read']
 def data_at(parent,cond):
  f=POOL/'hc2/capture'/cond/f'{parent:05}.json';r=read(f);assert sha(f)==cb['files'][str(f.relative_to(POOL/'hc2'))];cf=POOL/'hc2'/r['cache'];assert sha(cf)==r['sha256'];return device_tree(load(cf),'cuda'),r
 parity=0
 for cond in p['conditions']:
  for t in targets['rows']:
   i=t['donor_arrival'];order=t['order']
   for arm in ['A','R']:
    f=BASE/'transfer/pairs'/arm/cond/order/f'{i:05}.pt'
    if f.with_suffix('.json').exists():receipt(f);done+=1;continue
    budget();oldf=OLD/'hc2'/arm/'online'/cond/order/f'{i:05}.pt';old=load(oldf);rr=read(oldf.with_suffix('.json'));assert sha(oldf)==rr['sha256'] and old['expert_scheduled'];assert state_hash(old['pre_state'])==old['pre_sha'] and state_hash(old['post_state'])==old['post_sha']
    predictions={};target_receipts={};frozen_intervals={}
    for parent in sorted({v['parent'] for v in t['roles'].values()}):
     data,cr=data_at(parent,cond);frozen_intervals[parent]=data['prediction']['indices'];out={}
     for phase in ['pre','post']:
      st=device_tree(old[phase+'_state'],'cuda');_,_,pred=predict(model,data,st);out[phase]=compact_prediction(pred);replays+=1
     if parent==t['donor_parent']:
      for phase,saved in [('pre','slow'),('post','post_prediction')]:assert torch.equal(out[phase]['boxes'],old[saved]['boxes']) and out[phase]['indices']==old[saved]['indices'],(cond,order,i,arm,phase)
      parity+=1
     predictions[parent]=out;target_receipts[parent]=dict(capture_sha256=cr['sha256'],pixel_sha256=cr['pixel_sha256']);del data;gc.collect();torch.cuda.empty_cache()
    assert state_hash(model.state_dict())==mh
    commit(f,dict(arm=arm,condition=cond,order=order,donor_arrival=i,donor_parent=t['donor_parent'],roles=t['roles'],predictions=predictions,frozen_intervals=frozen_intervals,self_interval=old['slow']['indices'],donor_payload_sha256=sha(oldf),pre_sha=old['pre_sha'],post_sha=old['post_sha'],updated=old['updated'],target_receipts=target_receipts,GT_read=False,parameter_updates=0,new_experts=0,checkpoint_restored=True))
    done+=1;status(BASE/'transfer/STATUS.json',dict(status='running',done=done,total=192,worker_pid=os.getpid(),suffix_replays=replays,time=time.time()));print('TRANSFER',done,192,cond,order,arm,i,flush=True)
 assert done==192;verify('TRANSFER');assert state_hash(model.state_dict())==mh
 write(BASE/'transfer/PREDICTION_BARRIER.json',dict(pairs=768,logical_cross_query_pairs=576,write_payloads=192,files={str(f.relative_to(BASE)):sha(f) for f in (BASE/'transfer/pairs').rglob('*.json')},target_lock_sha256=sha(BASE/'transfer/TARGET_LOCK.json'),GT_read=False,time=time.time()))
 write(BASE/'transfer/RESOURCES.json',dict(worker_wall_seconds=time.monotonic()-tick,wall_includes_loading_IO=True,suffix_replays=replays,live_self_parity=parity,text_only_forwards=32,new_backbone_forwards=0,new_experts=0,parameter_updates=0,peak_vram_bytes=torch.cuda.max_memory_allocated()))
 status(BASE/'transfer/STATUS.json',dict(status='sealed_pending_root_score',done=192,total=192,time=time.time()));fh.close()
if __name__=='__main__':
 try:run()
 except BaseException as e:
  write(BASE/'transfer'/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));status(BASE/'transfer/STATUS.json',dict(status='failed',error=repr(e),time=time.time()));raise
