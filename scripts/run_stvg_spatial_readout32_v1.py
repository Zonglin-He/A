"""Raw DINO score consistency + independent six-layer readout, finite32."""
from pathlib import Path
import argparse,collections,copy,gc,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
BASE=ROOT/'artifacts/stvg_spatial_readout32_v1';OLD=ROOT/'artifacts/stvg_joint_event_critic_v1';PREV=ROOT/'artifacts/stvg_identity_trajectory32_v1'
def status(stage,n,**kw):put(BASE/'STATUS.json',dict(status='running',stage=stage,done=n,total=32,pid=os.getpid(),timestamp=time.time(),**kw),True)
def prepare():
 records=read(OLD/'PLAN_LOCK.json')['target'];cfg=read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets'];own=['vg_tta/stvg_spatial_readout_v1.py','scripts/run_stvg_spatial_readout32_v1.py','protocols/stvg_spatial_readout32_v1.md','methods/decota_final_simplified_v1/observations.py','vg_tta/decota_spatial_opd_tunable_v1.py']
 put(BASE/'PLAN_LOCK.json',dict(status='locked_before_models',count=32,records=records,arms=['original_OPD','target_rank_OPD','layer_select_original_state'],time='same Source Native physical interval',state='same episodic frozen Source1792, no LN inheritance/writeback',configurations={d:cfg[d]['config']for d in cfg},max_new_DINO_calls=128,max_new_fits=32,GPU_seconds_cap=3600,deadline_epoch=time.time()+7200,GT_online=False,all32_prediction_seal_before_new_GT=True,independent_confirmation=False))
 put(BASE/'RUNTIME_LOCK.json',dict(status='locked',pins={f:sha(ROOT/f)for f in own},registries={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']},old_payloads={r['alias']:sha(OLD/'target'/(r['alias']+'.pt'))for r in records},original_matched_fits={r['alias']:sha(PREV/'predictions'/(r['alias']+'.pt'))for r in records}));status('prepared',0,GT_read=False)
def verify():
 p=read(BASE/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch']
 for f,h in {**read(BASE/'RUNTIME_LOCK.json')['pins'],**read(BASE/'RUNTIME_LOCK.json')['registries']}.items():assert sha(ROOT/f)==h,f
 from main_method.runtime import resource_guard
 resource_guard();return p

def measure():
 import numpy as np,torch
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.observations import SpatialExpert,ContextView,tensor_hash
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.tensors import state_hash,detached
 from vg_tta.stvg_spatial_readout_v1 import context_probe,full_layers,select_layer
 from vg_tta.decota_spatial_opd_tunable_v1 import fit
 p=verify();lock=read(BASE/'RUNTIME_LOCK.json');budget_file=BASE/'recovery/empty_observation_cleanup002/BUDGET_READBACK.json';budget_offset=read(budget_file)['old_GPU_lease_upper_bound_seconds']if budget_file.exists()else 0.;lease=gpu();started=time.perf_counter();model=None;domain=None;cache=collections.OrderedDict();files=[];calls=0;dino=SpatialExpert(ROOT/EXPERT_SNAPSHOT);eh=state_hash(dino.model.state_dict())
 try:
  for i,r in enumerate(p['records']):
   verify();assert time.perf_counter()-started+budget_offset<p['GPU_seconds_cap'];dest=BASE/'predictions'/(r['alias']+'.pt');alias=r['alias'];ds=r['dataset'];status('raw_DINO_capture',i,GT_read=False)
   if dest.exists():z=load(dest);files.append(read(dest.with_suffix('.receipt.json')));calls+=z['compute']['new_DINO_calls'];continue
   if ds!=domain:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    domain=ds;model=model_for('vidstg'if ds=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());cache.clear()
   tick=time.perf_counter();row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,cache);batch,recs,base=frozen_forward(model,frames,row);old=load(OLD/'target'/(alias+'.pt'));prev=load(PREV/'predictions'/(alias+'.pt'));assert sha(OLD/'target'/(alias+'.pt'))==lock['old_payloads'][alias]and sha(PREV/'predictions'/(alias+'.pt'))==lock['original_matched_fits'][alias];assert ids==old['frame_ids']and torch.equal(base.zero['boxes'].cpu(),old['boxes'][0]);capture_seconds=time.perf_counter()-tick
   expert=copy.deepcopy(old['expert']);expert['anchors']={'single4':[]};raws={};newcalls=0;dino_seconds=0;inputchecks=0
   for (tag,pos),ob in sorted(old['expert']['observations'].items()):
    receipt=ob['receipt'];assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==receipt['rgb_sha256'];inputs={}
    def hook(module,args,kw):
     for k in ['pixel_values','pixel_mask','input_ids','attention_mask']:
      if k in kw:inputs[k]=dict(sha256=tensor_hash(kw[k]),shape=list(kw[k].shape),dtype=str(kw[k].dtype))
    h=dino.model.register_forward_pre_hook(hook,with_kwargs=True)
    try:
     with torch.no_grad():
      d=ContextView(dino,row['parses']['context'])(frames[pos],receipt['text'],row['parses']['context']['entity'])if receipt['context_active']else dino(frames[pos],row['parses']['old']['phrase'],row['parses']['old']['entity'])
    finally:h.remove()
    assert inputs==receipt['inputs'];inputchecks+=len(inputs);assert d['text']==receipt['text'];newcalls+=1;calls+=1;assert calls<=128;dino_seconds+=d['seconds']
    if receipt['context_active']:
     original,stage=context_probe(d,pos,ids[pos],'mixed');new,stage_new=context_probe(d,pos,ids[pos],'target')
     assert torch.equal(torch.as_tensor(original['boxes']),torch.as_tensor(ob['probe']['boxes']));assert torch.equal(torch.as_tensor(original['target_scores']),torch.as_tensor(ob['probe']['target_scores']));assert original['candidate_ids']==ob['probe']['candidate_ids']and original['accepted']==ob['probe']['accepted']and original['reason']==ob['probe']['reason']
     expert['observations'][tag,pos]={'probe':new,'receipt':receipt};raws[pos]=dict(detection=detached(d,'cpu'),old_stage=detached(stage,'cpu'),new_stage=detached(stage_new,'cpu'),old_probe=detached(original,'cpu'),new_probe=detached(new,'cpu'),context=True)
    else:
     original=ob['probe'];assert d['accepted']==original['accepted']and d['reason']==original['reason'];assert (not d['box']and len(original['boxes'])==0)or torch.equal(torch.tensor(d['box']),torch.as_tensor(original['boxes'])[0]);new=original;raws[pos]=dict(detection=detached(d,'cpu'),old_probe=original,new_probe=new,context=False)
    if new['accepted']:
     j=int(np.argmax(new['target_scores']));expert['anchors']['single4'].append(dict(position=pos,frame_id=ids[pos],box=torch.as_tensor(new['boxes'][j]).tolist(),score=float(new['target_scores'][j]),weight=1.))
   initial=detached(base.initial,'cpu');before=prev['arms']['top1_frame']['fit'];assert all(torch.equal(initial[k],v)for k,v in before['initial'].items());tick=time.perf_counter();layers,full=full_layers(base,ids,recs,before['state']);assert torch.equal(full['boxes'],before['final']);selected=select_layer(layers,old['expert']['anchors']['single4']);layer_seconds=time.perf_counter()-tick
   status('target_rank_OPD_fit',i,GT_read=False);tick=time.perf_counter();newfit=fit(base,initial,expert,ids,row['key'],'on_policy',config=p['configurations'][ds]);newlayers,newfull=full_layers(base,ids,recs,newfit['state']);assert torch.equal(newfull['boxes'],newfit['final']);fit_seconds=time.perf_counter()-tick
   assert state_hash(model.state_dict())==mh;assert all(torch.equal(base.state()[k].cpu(),v)for k,v in initial.items())
   out=dict(alias=alias,dataset=ds,query_ordinal=r['query_ordinal'],frame_ids=ids,Native_interval=old['native_interval'],Source_boxes=old['boxes'][0],initial=initial,old_expert=old['expert'],new_expert=expert,raw=raws,original_fit=before,new_fit=detached(newfit,'cpu'),original_layers=layers,selected_layer=selected['layer'],layer_scores=selected['scores'],arms=dict(original_OPD=before['final'],target_rank_OPD=newfit['final'],layer_select_original_state=selected['boxes'].cpu()),original_complete_output=full,new_complete_output=newfull,GT_read=False,compute=dict(new_DINO_calls=newcalls,DINO_forward_seconds=dino_seconds,Source_capture_decode_seconds=capture_seconds,layer_complete_replay_seconds=layer_seconds,new_reference_OPD_and_complete_readback_seconds=fit_seconds,input_tensor_exact_checks=inputchecks))
   files.append(save(dest,detached(out,'cpu')));status('sealed_query',i+1,GT_read=False);print(alias,i+1,'calls',calls,flush=True)
   del out,newfit,base,batch,frames,old,prev,raws,layers,newlayers,full,newfull,expert;gc.collect();torch.cuda.empty_cache()
  assert state_hash(dino.model.state_dict())==eh
  put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,new_DINO_calls=calls,GPU_lease_seconds=budget_offset+time.perf_counter()-started,recovery_GPU_lease_upper_bound_included=budget_offset,DINO_state_unchanged=True));status('predictions_complete',32,GT_read=False)
 finally:lease.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure']);a=p.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  put(BASE/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partial_preserved=True));raise
