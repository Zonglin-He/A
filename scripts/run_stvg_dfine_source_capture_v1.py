"""Bounded source-only intake/capture for D-FINE-inspired conditional refinement."""
from pathlib import Path
import sys,os,time,json,hashlib,collections,gc,argparse,zipfile,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,load,save
D=ROOT/'artifacts/stvg_dfine_refinement32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/dfine_refinement32';OLD=ROOT/'artifacts/stvg_joint_event_critic_v1';TARGET=ROOT/'artifacts/stvg_discriminative_dense32_v1';V=None

def verify():
 global V
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'CAPTURE_RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 if V is None:V=PinVerifier(l['pins'])
 V.verify();resource_guard();assert time.time()<p['deadline_epoch'];return p

def status(stage,n,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=n,pid=os.getpid(),epoch=time.time(),**kw),True)

def prepare():
 from methods.decota_final_simplified_v1.observations import QuerySubjectParser,visual_query,parse_context
 targets=read(TARGET/'PLAN_LOCK.json')['records'];old=read(OLD/'PLAN_LOCK.json');lookup={(r['dataset'],r['parent']):r for r in old['source']};parser=None;rows=[];provenance={};excluded={};source_labels=[]
 raw=read(ROOT/'external/VidSTG-Dataset/annotations/train_annotations.json');raw_index=collections.defaultdict(list)
 for i,v in enumerate(raw):raw_index[v['vid']].append((i,v))
 vidor=zipfile.ZipFile(ROOT/'downloads/vidor/training-annotation.zip');names={Path(n).stem:n for n in vidor.namelist()if n.endswith('.json')}
 from scripts.prepare_vidstg_wrong_domain_support import trajectory_for_target
 for ds,di in [('hc2','hc2_to_vid'),('vidstg','vid_to_hc1')]:
  mf=ROOT/f'artifacts/decota_paper_execution_20260917/vitta/{di}/SOURCE_MANIFEST.json';m=read(mf);assert m['official_split']=='train';exclude={r['parent']for r in targets if r['dataset']==ds};pool=[r for r in m['records']if r['input']['source']not in exclude];excluded[ds]=len(m['records'])-len(pool)
  pool.sort(key=lambda r:hashlib.sha256(('dfine-source-v1|'+r['input']['source']).encode()).hexdigest());assert len(pool)>=59
  provenance[ds]=dict(manifest=str(mf.relative_to(ROOT)),sha256=sha(mf),annotation=m['annotation'],annotation_sha256=m['annotation_sha256'],available_after_target_exclusion=len(pool),source_checkpoint='hcstvg2'if ds=='hc2'else'vidstg')
  hc=read(m['annotation'])if ds=='hc2'else None
  for k,r in enumerate(pool[:59]):
   q=dict(r['input']);assert Path(q['video_path']).is_file()and sha(q['video_path'])==q['video_sha256'];parent=q['source'];prior=lookup.get((ds,parent));split='source_train'if k<43 else'source_validation';alias=('DH'if ds=='hc2'else'DV')+f'{k+1:03}'
   if prior:assert prior['input']['caption']==q['caption'];parses=prior['parses']
   else:
    if parser is None:parser=QuerySubjectParser(ROOT/'.cache/stanza')
    parses=parser(q['caption']);parses.update(old=visual_query(parser,q['caption']),context=parse_context(parser,q['caption']))
   if 'fps'not in q:
    rate=subprocess.check_output([str(ROOT/'.conda/tubedetr/bin/ffprobe'),'-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','-of','default=nw=1:nk=1',q['video_path']],text=True).strip();a,b=rate.split('/');q['fps']=float(a)/float(b)
   rec=dict(alias=alias,dataset=ds,parent=parent,split=split,input=q,frame_ids=q['frame_ids'],key=r['key'],parses=parses,annotation_key=r['official_annotation_key'],reuse_source_alias=prior['alias']if prior else None);rows.append(rec)
   if ds=='hc2':
    v=hc[r['official_annotation_key']];assert v['English'].strip().lower()==q['caption'].strip().lower();s=int(v['st_frame'])-1;e=s+len(v['bbox'])-1;gt={s+j:[x,y,min(x+w,q['width']),min(y+h,q['height'])]for j,(x,y,w,h)in enumerate(v['bbox'])};binding=dict(official_annotation_key=r['official_annotation_key'])
   else:
    matches=[]
    for i,v in raw_index[parent]:
     if int(v['used_segment']['begin_fid'])!=q['start_frame']or int(v['used_segment']['end_fid'])!=q['end_frame']:continue
     for field in ['captions','questions']:
      for j,c in enumerate(v.get(field,[])):
       if c['description'].strip().lower()==q['caption'].strip().lower():matches.append((i,field,j,int(c['target_id'])))
    assert matches,alias;identities={(raw[i]['temporal_gt']['begin_fid'],raw[i]['temporal_gt']['end_fid'],t)for i,f,j,t in matches};assert len(identities)==1,('Ambiguous source binding',alias)
    i,field,j,target=matches[0];v=raw[i];ann=json.loads(vidor.read(names[parent]));assert ann['width']==q['width']and ann['height']==q['height'];s,e=int(v['temporal_gt']['begin_fid']),int(v['temporal_gt']['end_fid']);gt={int(fid):[t['bbox'][0],t['bbox'][1],t['bbox'][0]+t['bbox'][2],t['bbox'][1]+t['bbox'][3]]for fid,t in trajectory_for_target(ann,target).items()if s<=int(fid)<e};binding=dict(original_annotation_index=i,field=field,query_index=j,target_id=target,duplicate_equivalent_bindings=len(matches))
   assert gt and e>s;source_labels.append((alias,dict(alias=alias,dataset=ds,split=split,gt=gt,span=[s,e],binding=binding,source_GT_only=True,missing_GT_unknown=True)))
 p=dict(status='locked',source=rows,target=targets,counts=dict(source_train_per_domain=43,source_validation_per_domain=16,target_per_domain=16),source_provenance=provenance,excluded_target_parent_counts=excluded,
  seed=20261018,GPU_cap_seconds=3600,deadline_epoch=time.time()+3*3600,created_epoch=time.time(),source_only_training=True,source_validation_historically_exposed=True,target_development_exposed=True,
  module=dict(bins=33,roi_size=7,context_scale=1.5,project_channels=32,hidden=128,refinement_stages=2,bin_range_source_train_quantile=.95,bin_range_floor=.25,bin_range_ceiling=2.,initial_nonzero_logit=-4.),
  training=dict(epochs=30,validation_every=5,steps_per_epoch=6,batch_size=64,lr=.001,weight_decay=.0001,gradient_clip=5.,loss='mean two-stage (interpolated boundary CE +5L1+2GIoU)',start_kinds=['E','Native','E-center-noise'],select='source-validation complete vIoU maximum among trained checkpoints, ties earliest'),
  target_configuration=dict(rounds=2,candidates_per_round=17,teacher_samples=32,center_scale=.25,temperature=.1,step_size=.5,planned_nodes_width_height_time_fixed=True,source_gate='both source domains center-only direct2 vIoU and unobserved actual sIoU improve E; otherwise stop before target',no_new_DINO=True),
  registries_unchanged=True,DFINE_scope='inspired two-stage nonuniform edge-distribution refinement, no GO-LSD, not a full D-FINE reproduction')
 put(D/'PLAN_LOCK.json',p)
 for a,z in source_labels:save(D/'source_supervision'/(a+'.pt'),z)
 pins={f:sha(ROOT/f)for f in ['scripts/run_stvg_dfine_source_capture_v1.py','protocols/stvg_dfine_refinement32_v1.md','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json']}
 for ds,v in provenance.items():pins[v['manifest']]=v['sha256']
 pins['external/VidSTG-Dataset/annotations/train_annotations.json']=sha(ROOT/'external/VidSTG-Dataset/annotations/train_annotations.json');pins['downloads/vidor/training-annotation.zip']=sha(ROOT/'downloads/vidor/training-annotation.zip')
 put(D/'CAPTURE_RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));status('source_prepared',0,source_GT_only=True);print('SOURCE_LOCK',len(rows),excluded,flush=True)

def capture():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from vg_tta.stvg_shared_text_query32_v1 import input_pool
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 p=verify();lease=None
 while lease is None:
  try:lease=gpu()
  except BlockingIOError:status('waiting_shared_GPU_lease',0);time.sleep(10);verify()
 prior_spent=sum(read(f)['actual_GPU_elapsed_seconds']for f in D.glob('SOURCE_GPU_STOP*.json')if f.name!='SOURCE_GPU_STOP_LATEST.json');tick=time.perf_counter();put(D/'SOURCE_GPU_RUNTIME002.json',dict(pid=os.getpid(),epoch=time.time(),cap_seconds=3600));model=None;domain=None;dino=None;files=[];cost=[];cache=collections.OrderedDict();cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json')
 try:
  for i,r in enumerate(p['source']):
   verify();assert prior_spent+time.perf_counter()-tick<3600;ds=r['dataset'];a=r['alias']
   existing=D/'source_features'/(a+'.pt')
   if existing.exists():
    old=load(existing);assert old['alias']==a and old['dataset']==ds and old['frame_ids']==r['frame_ids'];files.append(read(existing.with_suffix('.receipt.json')));cost.append(dict(alias=a,dataset=ds,seconds=None,new_Source_captures=1,new_DINO_calls=0 if r['reuse_source_alias']else old['expert']['new_DINO'],expert_reused=bool(r['reuse_source_alias']),retained_original_capture=True,individual_seconds_unknown=True));continue
   status('source_feature_capture',i,alias=a,total=118,target_GT_read=False)
   if ds!=domain:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('hcstvg2'if ds=='hc2'else'vidstg');mh=state_hash(model.state_dict());domain=ds;cache.clear()
   row=dict(input=r['input'],frame_ids=r['frame_ids'],parses=r['parses'],key=r['key'],ordinal=i);frames,ids,_=frames_for(ds,row,cache);appearance=[];masks=[]
   def hook(m,args,out):appearance.append(out.detach().float().cpu())
   def mask(m,args,out):_,v,*_=out[0].decompose();masks.append((~v).detach().cpu())
   hs=[model.input_proj.register_forward_hook(hook),model.vis_encoder.register_forward_hook(mask)];t=time.perf_counter()
   try:batch,recs,base=frozen_forward(model,frames,row)
   finally:
    for h in hs:h.remove()
   native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);nb=native['boxes'].double().cpu().numpy();reuse=False
   if r['reuse_source_alias']:
    old=load(OLD/'source'/(r['reuse_source_alias']+'.pt'));assert torch.equal(native['boxes'].cpu(),old['boxes'][0])and ids==old['frame_ids']and native['physical_interval']==old['native_interval'];expert=old['expert'];reuse=True
    for (_,pos),v in expert['observations'].items():assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==v['receipt']['rgb_sha256']
   else:
    if dino is None:dino=SpatialExpert(ROOT/EXPERT_SNAPSHOT)
    expert=observations(dino,row['parses'],frames,ids,native['indices'],audit=True)
   pool=input_pool(expert);pools=[encode(f['boxes'])-encode(nb)[f['position']]for f in pool];setting=cal['source_domains'][ds];est=estimate(pools,[f['scores']for f in pool],setting['scale'],setting['truncation'],1/16,diagnostics=False);E=apply(nb,est['delta'])
   assert len(appearance)==len(masks)==2;maps=torch.empty(len(ids),*appearance[0].shape[1:]);valid=torch.empty(len(ids),*masks[0].shape[1:],dtype=torch.bool)
   for off in [0,1]:maps[off::2]=appearance[off];valid[off::2]=masks[off]
   grids=probabilities(recs,base.zero['logits']);assert sorted(set(sum([g['ids']for g in grids],[])))==ids;q=independent_cdf(grids,ids);iv=final_mbr(ids,q)['interval'];assert state_hash(model.state_dict())==mh
   files.append(save(D/'source_features'/(a+'.pt'),dict(alias=a,dataset=ds,split=r['split'],maps=maps,valid=valid,frame_ids=ids,E=torch.tensor(E),Source=torch.tensor(nb),frames=pool,selected=est['selected'],planned_positions=expert['positions4'],interval=iv,expert=expert,model_hash=mh,source_GT_input=False,target_GT_read=False)))
   cost.append(dict(alias=a,dataset=ds,seconds=time.perf_counter()-t,new_Source_captures=1,new_DINO_calls=0 if reuse else expert['new_DINO'],expert_reused=reuse));print(a,i+1,round(time.perf_counter()-tick,3),flush=True)
   del frames,batch,base,native,appearance,masks,maps,valid,expert;gc.collect();torch.cuda.empty_cache()
   if i==1:put(D/'PILOT2_TIMING.json',dict(count=2,GPU_elapsed_seconds=time.perf_counter()-tick,rough118_seconds=59*(time.perf_counter()-tick),not_a_guarantee=True))
  put(D/'SOURCE_FEATURE_BARRIER.json',dict(status='complete',count=118,files=files,costs=cost,GPU_allocation_seconds=prior_spent+time.perf_counter()-tick,prior_capture_GPU_seconds=prior_spent,peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_target_DINO_calls=0));status('source_features_complete',118)
 finally:
  lease.close()
  put(D/'SOURCE_GPU_STOP002.json',dict(actual_GPU_elapsed_seconds=time.perf_counter()-tick,scope='source captures only',all_partials_preserved=True));put(D/'SOURCE_GPU_STOP_LATEST.json',dict(actual_GPU_elapsed_seconds=prior_spent+time.perf_counter()-tick,scope='cumulative all source capture segments',prior_segment_seconds=prior_spent),True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,error=str(e),type=type(e).__name__,partials_preserved=True));raise
