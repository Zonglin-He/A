"""Matched Source-episodic identity/trajectory OPD32. No target labels in measure."""
from pathlib import Path
import argparse,collections,gc,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
BASE=ROOT/'artifacts/stvg_identity_trajectory32_v1';OLD=ROOT/'artifacts/stvg_joint_event_critic_v1';ARMS=['top1_frame','multi_frame','multi_tube','multi_direct']
def status(stage,done,**kw):put(BASE/'STATUS.json',dict(status='running',stage=stage,done=done,total=32,pid=os.getpid(),timestamp=time.time(),**kw),True)
def prepare():
 records=read(OLD/'PLAN_LOCK.json')['target'];cfg=read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets'];own=['scripts/run_stvg_identity_trajectory32_v1.py','vg_tta/stvg_identity_trajectory_opd_v1.py','vg_tta/stvg_interval_mbr_v1.py','protocols/stvg_interval_identity_opd_v1.md','vg_tta/decota_spatial_opd_tunable_v1.py']
 put(BASE/'PLAN_LOCK.json',dict(status='locked_before_models',count=32,records=records,arms=ARMS,configurations={d:cfg[d]['config']for d in cfg},cohort='same old exposed target16/domain, not original P6_64',initial='identical frozen Source1792 in all arms; episodic/no writeback',temporal='Source Native primary and exact Native posterior MBR secondary; no actionness',new_DINO_calls=0,new_Qwen_calls=0,new_training=0,deadline_epoch=time.time()+7200,GPU_seconds_cap=3600,GT_online=False,GT_after_complete_all32_allarms_seal=True))
 put(BASE/'RUNTIME_LOCK.json',dict(status='locked',pins={p:sha(ROOT/p)for p in own},registries={p:sha(ROOT/p)for p in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']},inputs={r['alias']:sha(OLD/'target'/(r['alias']+'.pt'))for r in records}));status('prepared',0,GT_read=False)
def verify():
 p=read(BASE/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];lock=read(BASE/'RUNTIME_LOCK.json')
 for f,h in {**lock['pins'],**lock['registries']}.items():assert sha(ROOT/f)==h,f
 from main_method.runtime import resource_guard
 resource_guard();return p

def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from scripts.stvg_pipeline_speed_v1 import full_prefix_readout
 from vg_tta.stvg_interval_mbr_v1 import event_membership,mbr
 from vg_tta.stvg_identity_trajectory_opd_v1 import hypotheses,fit
 from vg_tta.decota_spatial_opd_tunable_v1 import fit as original_fit
 p=verify();lease=gpu();started=time.perf_counter();domain=None;model=None;cache=collections.OrderedDict();files=[]
 try:
  for i,r in enumerate(p['records']):
   verify();assert time.perf_counter()-started<p['GPU_seconds_cap'];alias=r['alias'];ds=r['dataset'];dest=BASE/'predictions'/(alias+'.pt')
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   status('Source_capture_and_matched_fits',i,GT_read=False)
   if ds!=domain:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
   tick=time.perf_counter();row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,cache);batch,recs,base=frozen_forward(model,frames,row);old=load(OLD/'target'/(alias+'.pt'));assert sha(OLD/'target'/(alias+'.pt'))==read(BASE/'RUNTIME_LOCK.json')['inputs'][alias]
   native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);assert torch.equal(native['boxes'],old['boxes'][0])and native['physical_interval']==old['native_interval']and ids==old['frame_ids']
   for (_,pos),ob in old['expert']['observations'].items():assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==ob['receipt']['rgb_sha256']
   memb=event_membership([x.numpy()for x in native['logits']],recs,ids);mb=[mbr(z.numpy(),v['frame_ids'])['interval']for z,v in zip(native['logits'],recs)];mb=[min(x[0]for x in mb),max(x[1]for x in mb)]
   h=hypotheses(base,old['expert'],ids,memb);initial=detached(base.initial,'cpu');outputs={};times={};capture=time.perf_counter()-tick
   for arm in ARMS:
    status('fit_'+arm,i,GT_read=False);base.restore(initial);tick=time.perf_counter()
    result=original_fit(base,initial,old['expert'],ids,row['key'],'on_policy',config=p['configurations'][ds])if arm=='top1_frame'else fit(base,initial,h,ids,row['key'],arm,p['configurations'][ds])
    assert torch.equal(result['before'],native['boxes']);full=full_prefix_readout(base,ids,recs,result['state']);assert torch.equal(full['boxes'],result['final']);assert all(torch.equal(initial[n],v)for n,v in result['initial'].items())
    outputs[arm]=dict(fit=detached(result,'cpu'),complete_output=detached(full,'cpu'));times[arm]=time.perf_counter()-tick
   assert state_hash(model.state_dict())==mh
   payload=dict(alias=alias,dataset=ds,query_ordinal=r['query_ordinal'],frame_ids=ids,Source_boxes=native['boxes'],Native_interval=native['physical_interval'],MBR_interval=mb,hypotheses=detached(h,'cpu'),membership=memb,initial=initial,arms=outputs,compute=dict(capture_decode_association_seconds=capture,arm_seconds=times,new_Source_captures=1,new_DINO_calls=0),GT_read=False)
   files.append(save(dest,payload));print(alias,i+1,'complete',flush=True);status('sealed_query',i+1,GT_read=False)
   del batch,base,frames,old,native,payload,outputs,h,result,full;gc.collect();torch.cuda.empty_cache()
  put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,GPU_lease_seconds=time.perf_counter()-started,new_DINO_calls=0));status('predictions_complete',32,GT_read=False)
 finally:lease.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure']);a=p.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  put(BASE/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),partials_preserved=True));raise
