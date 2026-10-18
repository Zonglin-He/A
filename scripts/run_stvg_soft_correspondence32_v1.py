"""Feature-only capture and sealed candidate feedback; no GT reader in measure."""
from pathlib import Path
import sys,time,os,collections,gc,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_soft_correspondence32_v1';OLD=ROOT/'artifacts/stvg_dino_rollback32_v1'
def prepare():
 from vg_tta.stvg_soft_correspondence_v1 import contract
 records=read(OLD/'PLAN_LOCK.json')['records'];own=['scripts/run_stvg_soft_correspondence32_v1.py','vg_tta/stvg_soft_correspondence_v1.py','protocols/stvg_soft_correspondence32_v1.md']
 put(D/'PLAN_LOCK.json',dict(status='locked_before_features_rewards_GT',count=32,records=records,correspondence_temperature=.07,far_distance_fraction=.25,deadline_epoch=time.time()+7200,GPU_seconds_cap=900,GT_online=False,phase2_conditional=True))
 put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in own},registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']},old_predictions={r['alias']:sha(OLD/'predictions'/(r['alias']+'.pt'))for r in records}));put(D/'CPU_CONTRACT_COMPLETION.json',contract())
def verify():
 p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];l=read(D/'RUNTIME_LOCK.json')
 for f,h in {**l['pins'],**l['registries']}.items():assert sha(ROOT/f)==h,f
 from main_method.runtime import resource_guard
 resource_guard();return p

def measure():
 import numpy as np,torch
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash,detached
 from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates,rollout
 from vg_tta.stvg_soft_correspondence_v1 import propagate,soft_iou
 from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
 p=verify();lease=gpu();start=time.perf_counter();files=[];domain=None;model=None;cache=collections.OrderedDict();quals=[]
 try:
  for i,r in enumerate(p['records']):
   verify();assert time.perf_counter()-start<p['GPU_seconds_cap'];alias=r['alias'];ds=r['dataset'];put(D/'STATUS.json',dict(status='running',stage='frozen_features_and_feedback',done=i,total=32,GT_read=False),True)
   dest=D/'predictions'/(alias+'.pt')
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   if ds!=domain:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
   old=load(OLD/'predictions'/(alias+'.pt'));assert sha(OLD/'predictions'/(alias+'.pt'))==read(D/'RUNTIME_LOCK.json')['old_predictions'][alias];src=load(OLD/'source'/(alias+'.pt'));row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,cache);appearance=[];masks=[]
   def apphook(module,args,out):appearance.append(out.detach().float().cpu())
   def maskhook(module,args,out):
    raw=out[0];x,mask,*_=raw.decompose();masks.append((~mask).detach().cpu())
   hooks=[model.input_proj.register_forward_hook(apphook),model.vis_encoder.register_forward_hook(maskhook)];tick=time.perf_counter()
   try:batch,recs,base=frozen_forward(model,frames,row)
   finally:
    for h in hooks:h.remove()
   assert len(appearance)==len(masks)==2
   frozen=old['arms']['Frozen_native']['boxes'];b=base.zero['boxes'].cpu().double();xy=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1).numpy();assert np.array_equal(xy,frozen)and ids==list(src['frame_ids'])
   for (_,pos),ob in old['expert']['observations'].items():assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==ob['receipt']['rgb_sha256']
   H,W=appearance[0].shape[-2:];maps=torch.empty(len(ids),appearance[0].shape[1],H,W);valid=torch.empty(len(ids),H,W,dtype=torch.bool)
   for off in [0,1]:
    assert len(appearance[off])==len(ids[off::2]);maps[off::2]=appearance[off];valid[off::2]=masks[off]
   feature_seconds=time.perf_counter()-tick;del batch,base,appearance,masks;gc.collect();torch.cuda.empty_cache();tick=time.perf_counter()
   anchors=old['expert']['anchors']['single4'];ap=[int(a['position'])for a in anchors];ab=[torch.tensor(a['box'])for a in anchors];soft=propagate(maps.cuda(),valid.cuda(),ap,ab,p['correspondence_temperature']);planned=old['expert']['positions4'];unobs=[j for j in range(len(ids))if j not in planned];perm=list(range(len(ids)))
   if len(unobs)>1:
    for j,k in enumerate(unobs):perm[k]=unobs[(j+1)%len(unobs)]
   central=old['fit']['before'].float();cfg=old['fit']['config'];rr=rollout(mean_coordinates(central),central,row['key'],0,config=cfg);actions=torch.sigmoid(rr['samples']);oldround=old['fit']['rounds'][0]
   if 'rollout'in oldround:
    for j,pos in enumerate(old['fit']['positions']):actions[pos]=torch.sigmoid(oldround['rollout']['samples'][j]);rr['samples'][pos]=oldround['rollout']['samples'][j]
   if anchors:
    real=torch.stack([soft_iou(actions,sm[:,None])for sm in soft]).float();shuffled=torch.stack([soft_iou(actions,sm[perm,None])for sm in soft]).float();order=np.argsort(ap);aa=np.array(ap)[order];boxes=np.array([a['box']for a in anchors])[order];physical=np.array(ids);ref=np.stack([np.interp(physical,physical[aa],boxes[:,c])for c in range(4)],1);interpolated=aligned_iou(actions,torch.tensor(ref,dtype=actions.dtype)[:,None]);rewards=dict(real=real.mean(0),shuffled=shuffled.mean(0),interpolation=interpolated)
   else:real=shuffled=torch.empty(0,len(ids),32);rewards={a:torch.zeros(len(ids),32)for a in ['real','shuffled','interpolation']}
   weights={a:torch.softmax(v/cfg['tau'],-1)for a,v in rewards.items()};assert all(torch.isfinite(v).all()for v in weights.values());assert state_hash(model.state_dict())==mh
   if not any(q['dataset']==ds for q in quals):
    # Repeated matching is deterministic on frozen cached tensors; no new model.
    again=propagate(maps.cuda(),valid.cuda(),ap,ab,p['correspondence_temperature']);assert torch.equal(soft,again);quals.append(dict(dataset=ds,matching_replay_exact=True,Source_complete_output_exact=True,all_anchor_pixels_exact=True,pre_text_projection_maps=True))
   out=dict(alias=alias,dataset=ds,query_ordinal=r['query_ordinal'],frame_ids=ids,Native_interval=src['Native_interval'],planned_positions=planned,admitted_positions=ap,maps=maps,valid=valid,soft_maps_per_anchor=soft,permutation=perm,central=central,samples=rr['samples'],actions=actions,rewards=rewards,weights=weights,per_anchor_rewards=real,per_anchor_disagreement=real.std(0,unbiased=False)if anchors else torch.zeros(len(ids),32),config=cfg,GT_read=False,compute=dict(feature_capture_seconds=feature_seconds,correspondence_sampling_reward_seconds=time.perf_counter()-tick,new_DINO_calls=0,new_Student_updates=0,new_Source_capture=1))
   files.append(save(dest,out));print(alias,i+1,'features',list(maps.shape),'anchors',len(ap),flush=True);del out,maps,valid,soft,old,src,frames,actions,rr,weights,rewards;gc.collect();torch.cuda.empty_cache()
  put(D/'GLOBAL_FEEDBACK_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,new_DINO_calls=0,new_Student_updates=0,GPU_lease_seconds=time.perf_counter()-start));put(D/'FEATURE_QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,rows=quals));put(D/'STATUS.json',dict(status='complete',stage='feedback_sealed_before_GT',done=32,total=32),True)
 finally:lease.close()
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','measure']);stage=a.parse_args().stage
 try:globals()[stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',type=type(e).__name__,error=str(e),partial_preserved=True));raise
