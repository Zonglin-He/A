"""Original128/domain episodic TTFT pilot with source-only frozen settings."""
from pathlib import Path
import os,sys,json,time,gc,argparse,hashlib,traceback
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(S),str(R),str(R/'tools')]
import calibration_runtime_v1 as c
import numpy as np,torch
from core import adapt,decoder_boxes,to_device,state_digest
from model_utils import cpu_source,spatial_modules,restore
visibility=os.environ.get('CUDA_VISIBLE_DEVICES');sys.path.insert(0,str(R/'studies/reviewer_key_controls_v1'))
import common_v1 as legacy
if visibility is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=visibility
D=c.D/'target_pilot_v1';P=R/'results/expert_guided_ttft_v1/target_pilot_v1'
read=c.read;sha=c.sha;atomic=c.atomic;identity=c.identity;save=c.save;load=c.load
ARMS=['Source','SourceTR','TTFT','Current'];PAIRS=[('TTFT','Source'),('TTFT','SourceTR'),('Current','TTFT')]
def panel(ds):return read(D/'selection.json')[ds]
def cfg(ds):return read(D/'plan.json')['configs'][ds]
def preflight():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1);selection=read(R/'outputs/matched_pipeline_timing_v1/selection.json');cases=[]
 for ds in c.TARGETS:
  for row in selection[ds][:2]:
   old,cap,a,proof=references(ds,row);q,_=c.one_step(old['features']['AuxSourceROI'],old['q0'],kappa=0,beta=1);iv=np.asarray(c.map_interval(list(a['frame_ids']),q));assert np.array_equal(iv,a['SourceTR_interval'])
   assert np.array_equal(c.native_prior(cap['records'],cap['native_logits'],cap['frame_ids']),old['q0']);cases.append(dict(dataset=ds,reference_binding_sha256=c.digest(proof),original_Source_native_heads_prior_SourceROI_update_interval_exact=True))
 atomic(D/'reference_preflight.json',dict(cases=cases,actual_new_Source=0,new_model=0,new_expert=0,new_GT=0,not_actual_GPU_bridge=True));print('four real cached target references passed',flush=True)
def prepare():
 assert not(D/'plan.json').exists();accept=read(c.C/'root_acceptance.json');assert accept['source_functioning_accepted']and accept['no_target_scores_used'];assert accept['source_audit_sha256']==sha(c.C/'audit_receipt.json')
 selection=read(R/'outputs/matched_pipeline_timing_v1/selection.json')
 for ds,rr in selection.items():assert len(rr)==128 and len({r['metadata']['original_video_id']for r in rr})==128
 atomic(D/'selection.json',selection)
 assert len(read(D/'reference_preflight.json')['cases'])==4
 files=[S/'target_pilot_v1.py',S/'target_pilot_controller_v1.py',c.C/'root_acceptance.json',c.C/'audit_receipt.json',c.P/'report.json',c.P/'ROOT_diagnostics.json',R/'results/expert_guided_ttft_v1/report_code/root_source_accept_v1.py',D/'reference_preflight.json',D/'selection.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',legacy.selected.RUN/'plan.json',R/'outputs/reviewer_key_controls_v1/plan_v1.json']
 files += [R/f for f in read(c.C/'plan.json')['code_and_input_sha256']]
 files += [c.C/'plan.json']
 for ds in c.TARGETS:files += [legacy.selected.RUN/ds/'sealed.json',legacy.selected.D/f'components_{ds}_sealed.json',legacy.selected.D/'components'/ds/'metrics.json',legacy.D/'temporal'/ds/'sealed.json',legacy.D/'temporal'/ds/'metrics.json',legacy.D/'temporal'/ds/'reporting_receipt.json',legacy.selected.PREV/'spatial'/f'{ds}_sealed.json']
 p=dict(schema='expert_guided_TTFT_source_selected_original128_target_pilot_v1',configs=accept['selected_configs'],queries_per_domain=128,source_config_choice='Frozen only by corresponding source-validation36/36; no target config/threshold/step choice. Exposed fixed original128 distinct parents/domain, not unknown confirmation.',method='Same raw8ReferDINO boxes/physicalpositions, complete Source pos_fc+PosDecoder+bbox head, frozen backbone/text/time; L1+GIoU episodic AdamW exact reset, full sampled-frame output, adapted-box native pretextROI one-pair kappa0/beta1/context1.5/jointMAP. No shared correction/transport/candidates/DP.',controls='Original Source, Source-box TR and Current full arrays/scalars reused by exact references. Zero-update Source replay/initial SourceROI pooling/q0/nativeheads and zero-update TR interval exact every query.',acquisition='At most256 necessary Source runs to obtain actual spatial decoder context and frames_cls observer, 0newexpert. Every paid full native state saved/SHA BEFORE compatibility/gradient checks. Keep all paid raw/context/selected states throughout pilot. No fallback/skip/hash relaxation or blind retry.',bridge='After first2 NEW Source+adaptation commits/domain wait for actual ROOT CPU paid/context/nativeheads/initialROI/selectedprediction cold input and interval audit. Root cannot handwrite ACK. Independent GPU process then replays all256 saved states and reset exactly without Source or updates.',score='Both128 predictions seal and independent GPU replay before separate officialCPUscore and another CPU allinput/feature/prototype/interval/independent dense scalar/means audit. Allsigned/shortlong/severe/rescue/parent10000CI and actual costs, no prefixGT.',new_Source_cap=256,new_expert=0,optimizer_steps_cap=128*accept['selected_configs']['hc2']['steps']+128*accept['selected_configs']['vidstg']['steps'],Source_preflight_and_calibration_costs_nonzero=True,precision='Original Source224 visualFP16/FP32 head+gradients, cudnnTF32True/matmulFalse, Torch4, unchanged optimizer/reduction math.',resources='SingleGPU/globallease/foreigncompute gate;CPU12-15 nice19 idleIO SourceTorch4, CPUaudit1thread, RAM8GiB/disk80GiB/private160GiB/72h. No other process signal. Source/update/trace/checkpoint costs nested, not matched deployment speed.',code_and_input_sha256={str(f.relative_to(R)):sha(f)for f in dict.fromkeys(files)})
 p['fingerprint']=c.digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='code_and_input_sha256'});print(p['fingerprint'],flush=True)
def check():
 p=read(D/'plan.json');assert p['fingerprint']==c.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in p['code_and_input_sha256'].items():assert sha(R/f)==h,f
 return p
def references(ds,row):
 old,proof=legacy.source(ds,row);cap,binding=legacy.selected.suite().load_parent(ds,row);e=read(legacy.D/'temporal'/ds/'sealed.json')['files'][row['id']];assert sha(legacy.D/e['path'])==e['sha256']
 with np.load(legacy.D/e['path'])as z:a={k:z[k].copy()for k in ['Source','Source_interval','SourceTR','SourceTR_interval','Ours','Ours_interval','frame_ids','positions8']}
 a['Current']=a.pop('Ours');a['Current_interval']=a.pop('Ours_interval')
 assert np.array_equal(a['Source'],old['Source'])and np.array_equal(a['frame_ids'],cap['frame_ids'])and np.array_equal(a['positions8'],cap['positions8'])
 return old,cap,a,dict(compact=proof,parent=binding,temporal_prediction_sha256=e['sha256'])
def resource(start):
 c.probe.resources(start);assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<160*2**30
def context(model,ds,row,folder,parser,refcap):
 from stvg.decode import decode
 from stvg.native import frozen_native_forward
 from stvg.representation import observe_native_grid
 from stvg.tensors import detached
 frames,ids=decode(row['metadata'],ds);assert ids==refcap['frame_ids'];parsed={'subject':parser(row['metadata']['caption'])['subject']}
 atomic(folder/'attempt.json',dict(worker=identity(),before_Source=True,GT_read=False,video_sha256=row['metadata']['video_sha256'],RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest()))
 calls=[];hook=model.ground_decoder.pos_fc.register_forward_pre_hook(lambda m,a:calls.append(detached(a[0],'cpu')))
 torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920);model.cuda();torch.cuda.synchronize();tick=time.perf_counter()
 try:
  with torch.no_grad(),observe_native_grid(model)as grid:batch,recs,native=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=parsed))
 finally:hook.remove()
 torch.cuda.synchronize();seconds=time.perf_counter()-tick;paid=dict(Source=native.zero['boxes'].cpu().numpy(),native_logits=detached(native.zero['logits'],'cpu'),records=recs,spatial_inputs=detached(native.spatial_inputs,'cpu'),frames_cls_calls=calls,native_local_grid=grid,frame_ids=ids,GT_read=False)
 save(folder/'paid_Source_complete.pt',paid);atomic(folder/'paid_Source_checkpoint.json',dict(worker=identity(),sha256=sha(folder/'paid_Source_complete.pt'),Source_seconds=seconds,checkpoint_before_validation=True,GT_read=False))
 assert np.array_equal(paid['Source'],refcap['Source'])and recs==refcap['records'];assert len(paid['native_logits'])==len(refcap['native_logits'])==2 and all(torch.equal(x,y)for x,y in zip(paid['native_logits'],refcap['native_logits']));assert len(calls)>=4
 ctx={k:paid[k]for k in ['Source','spatial_inputs','native_local_grid','frame_ids','native_logits','records']};ctx['frames_cls']=[calls[-3],calls[-1]];ctx['q0']=c.native_prior(recs,paid['native_logits'],ids)
 for cls,kw in zip(ctx['frames_cls'],ctx['spatial_inputs']):
  with torch.no_grad():v=model.ground_decoder.pos_fc(cls.cuda()).sigmoid().unsqueeze(1)
  assert torch.equal(v.cpu(),kw['pred_boxes'])
 save(folder/'context.pt',ctx);atomic(folder/'context_checkpoint.json',dict(sha256=sha(folder/'context.pt'),paid_Source_sha256=sha(folder/'paid_Source_complete.pt'),native_Source_logits_records_exact=True,actual_frames_cls_observer=True,GT_read=False));model.cpu();del batch,native,frames,paid,calls;gc.collect();torch.cuda.empty_cache();return ctx,seconds
def GPU():
 plan=check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'and not(D/'worker.json').exists()
 from stvg.runtime import setup,lease
 from stvg.query import QuerySubjectParser
 c.probe.foreign();lock=lease();setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;atomic(D/'worker.json',identity());start=time.perf_counter();counts=dict(Source=0,expert=0,optimizer_steps=0,queries=0);model=None;seals={}
 try:
  parser=QuerySubjectParser(R/'checkpoints/stanza')
  for ds in c.TARGETS:
   resource(start);c.probe.foreign();tick=time.perf_counter();model,meta=cpu_source(ds);atomic(D/ds/'model_load.json',dict(worker=identity(),seconds=time.perf_counter()-tick,checkpoint_load=meta.get('checkpoint_load')));files={};config=cfg(ds)
   for i,row in enumerate(panel(ds),1):
    resource(start);c.probe.foreign();tick=time.perf_counter();key=row['id'];folder=D/ds/'paid'/key;assert not folder.exists();folder.mkdir(parents=True);old,cap,a,proof=references(ds,row);ctx,secs=context(model,ds,row,folder,parser,cap);counts['Source']+=1
    assert np.array_equal(ctx['q0'],old['q0']);iv,features,_,_,_=c.finish(ctx,ctx['Source']);assert np.array_equal(features,old['features']['AuxSourceROI'])and np.array_equal(iv,a['SourceTR_interval'])
    boxes=np.asarray(old['geometries']['main']['native_boxes']);pos=a['positions8'];del cap,old;model.cpu();torch.cuda.empty_cache();torch.set_num_threads(4)
    result=adapt(model,ctx,boxes,pos,config,checkpoints=(config['steps'],),scratch=folder/'states');counts['optimizer_steps']+=config['steps'];out=result[config['steps']];iv,features,q,proto,pool=c.finish(ctx,out['boxes']);a['TTFT']=out['boxes'];a['TTFT_interval']=iv
    fp=folder/'adapted_evidence.pt';save(fp,dict(features=features,q1=q,prototype=proto,pooling=pool,GT_read=False));state=folder/'states'/f"step{config['steps']}.pt";tp=folder/'adaptation_trace.json';atomic(tp,{k:v for k,v in out.items()if k not in ['boxes','parameter_update_norms']})
    pp=D/ds/'predictions'/(key+'.npz');pp.parent.mkdir(parents=True,exist_ok=True)
    with pp.open('xb')as f:np.savez_compressed(f,**a)
    with np.load(pp)as z:assert set(z.files)==set(a)and all(np.array_equal(z[k],v)for k,v in a.items())
    record=dict(path=str(pp.relative_to(D)),sha256=sha(pp),context_path=str((folder/'context.pt').relative_to(D)),context_sha256=sha(folder/'context.pt'),paid_path=str((folder/'paid_Source_complete.pt').relative_to(D)),paid_sha256=sha(folder/'paid_Source_complete.pt'),state_path=str(state.relative_to(D)),state_sha256=sha(state),parameter_sha256=out['adapted_parameter_sha256'],evidence_path=str(fp.relative_to(D)),evidence_sha256=sha(fp),trace_path=str(tp.relative_to(D)),trace_sha256=sha(tp),proof=proof,Source_seconds=secs,query_wall_seconds=time.perf_counter()-tick,worker=identity(),GT_read=False)
    rp=D/ds/'receipts'/(key+'.json');atomic(rp,record);files[key]=dict(record,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1;atomic(D/'progress.json',dict(dataset=ds,completed_in_domain=i,total_in_domain=128,counts=counts,worker=identity(),GT_read=False))
    if i==2:
     atomic(D/ds/'bridge_waiting.json',dict(keys=list(files),worker=identity(),status='WAIT_actual_ROOT_two_paid_Source_bridge'))
     while not(D/ds/'root_actual_bridge.json').exists():resource(start);time.sleep(3)
     assert read(D/ds/'root_actual_bridge.json')['plan_fingerprint']==plan['fingerprint']
    del ctx,result,a,features,q,proto;gc.collect();torch.cuda.empty_cache()
   atomic(D/ds/'sealed.json',dict(files=files,queries=128,GT_read=False,plan_fingerprint=plan['fingerprint'],worker=identity()));seals[ds]=sha(D/ds/'sealed.json');model.cpu();del model;model=None;gc.collect();torch.cuda.empty_cache()
  assert counts['queries']==256 and counts['Source']==256 and counts['expert']==0 and counts['optimizer_steps']==plan['optimizer_steps_cap'];atomic(D/'sealed.json',dict(domains=seals,counts=counts,GT_read=False,elapsed_wall_seconds=time.perf_counter()-start,worker=identity(),plan_fingerprint=plan['fingerprint']));print('TTFT target256 sealed',flush=True)
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),counts=counts,worker=identity(),all_paid_state_preserved=True));raise
 finally:
  if model is not None:model.cpu()
  lock.close()
def checked(ds,key,sealed=True):
 e=read(D/ds/'sealed.json')['files'][key]if sealed else read(D/ds/'receipts'/(key+'.json'))
 for p,h in [('path','sha256'),('context_path','context_sha256'),('paid_path','paid_sha256'),('state_path','state_sha256'),('evidence_path','evidence_sha256'),('trace_path','trace_sha256')]:assert sha(D/e[p])==e[h]
 if sealed:assert sha(D/e['receipt_path'])==e['receipt_sha256']
 return e
def cold_math(ds,row,e):
 old,cap,ref,proof=references(ds,row);assert proof==e['proof'];ctx=load(D/e['context_path']);paid=load(D/e['paid_path']);assert not paid['GT_read'];assert np.array_equal(paid['Source'],ctx['Source'])and np.array_equal(ctx['Source'],ref['Source'])and paid['records']==ctx['records']==cap['records']
 assert len(paid['native_logits'])==len(cap['native_logits'])==2 and all(torch.equal(x,y)for x,y in zip(paid['native_logits'],cap['native_logits']))
 assert len(paid['frames_cls_calls'])>=4 and all(torch.equal(x,y)for x,y in zip(ctx['frames_cls'],[paid['frames_cls_calls'][-3],paid['frames_cls_calls'][-1]]));assert all(torch.equal(a,b)and torch.equal(va,vb)for(a,va),(b,vb)in zip(ctx['native_local_grid'],paid['native_local_grid']))
 assert np.array_equal(ctx['q0'],c.native_prior(ctx['records'],ctx['native_logits'],ctx['frame_ids']))and np.array_equal(ctx['q0'],old['q0'])
 with np.load(D/e['path'])as z:
  for k,v in ref.items():assert np.array_equal(v,z[k])
  iv,features,_,_,_=c.finish(ctx,z['Source']);assert np.array_equal(iv,z['SourceTR_interval'])and np.array_equal(features,old['features']['AuxSourceROI'])
  first=c.finish(ctx,z['TTFT']);second=c.finish(ctx,z['TTFT']);evidence=load(D/e['evidence_path']);assert all(np.array_equal(first[j],second[j])for j in [0,1,2]);assert np.array_equal(first[0],z['TTFT_interval'])and np.array_equal(first[1],evidence['features'])and np.array_equal(first[2],evidence['q1'])
 t=read(D/e['trace_path']);assert t['reset_all_parameters_and_Source_boxes_exact']and t['checkpoint_cold_forward_exact']and t['parameter_delta_L2']>0;assert all(x['gradient_norm_before_clip']>0 and np.isfinite(x['loss'])for x in t['trace'])
 return True
def bridge(ds):
 plan=check();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and not(D/ds/'root_actual_bridge.json').exists();torch.set_num_threads(1);wait=read(D/ds/'bridge_waiting.json');assert len(wait['keys'])==2;cases=[];lookup={r['id']:r for r in panel(ds)}
 for key in wait['keys']:
  e=checked(ds,key,False);assert cold_math(ds,lookup[key],e);cases.append(dict(paid_Source_sha256=e['paid_sha256'],context_sha256=e['context_sha256'],adapted_state_sha256=e['state_sha256'],all_inputs_native_heads_actual_ROI_prototype_interval_double_cold_exact=True))
 atomic(D/ds/'root_actual_bridge.json',dict(status='passed_actual_two_paid_Source_and_TTFT_complete_input_native_Source_EROIfree_temporal_cold_exact',plan_fingerprint=plan['fingerprint'],cases=cases,worker=identity(),new_Source=0,new_expert=0,new_GT=0));print(ds,'actual ROOT target TTFT bridge passed',flush=True)
def GPU_audit():
 check();seal=read(D/'sealed.json');assert not seal['GT_read']and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 from stvg.runtime import setup,lease
 c.probe.foreign();lock=lease();setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();n=0;model=None
 try:
  for ds in c.TARGETS:
   model,_=cpu_source(ds);original={k:{f:v.clone()for f,v in m.state_dict().items()}for k,m in spatial_modules(model).items()}
   for m in spatial_modules(model).values():m.cuda()
   assert seal['domains'][ds]==sha(D/ds/'sealed.json')
   for row in panel(ds):
    resource(start);c.probe.foreign();e=checked(ds,row['id']);ctx=load(D/e['context_path']);dev=to_device(ctx,'cuda');state=load(D/e['state_path']);assert state_digest(state)==e['parameter_sha256']
    with np.load(D/e['path'])as z:
     restore(model,original)
     with torch.no_grad():before=decoder_boxes(model,dev).cpu().numpy()
     assert np.array_equal(before,z['Source']);restore(model,state)
     with torch.no_grad():out=decoder_boxes(model,dev).cpu().numpy()
     assert np.array_equal(out,z['TTFT']);restore(model,original)
     with torch.no_grad():after=decoder_boxes(model,dev).cpu().numpy()
     assert np.array_equal(after,z['Source']);n+=1
    del ctx,dev,state;gc.collect();torch.cuda.empty_cache()
   model.cpu();del model;model=None;gc.collect();torch.cuda.empty_cache()
  assert n==256;atomic(D/'GPU_audit.json',dict(status='all256_adapted_state_Source_before_after_reset_bitwise_exact_independent_GPU_process',queries=n,new_Source=0,new_expert=0,optimizer_steps=0,seal_sha256=sha(D/'sealed.json'),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print('TTFT target256 independent GPU replay exact',flush=True)
 finally:
  if model is not None:model.cpu()
  lock.close()
def derive(ds,independent=False):
 assert read(D/'GPU_audit.json')['seal_sha256']==sha(D/'sealed.json');labels=c.label_reader(ds);prior={r['id']:r for r in read(legacy.D/'temporal'/ds/'metrics.json')['private_rows']};rr=[]
 for row in panel(ds):
  key=row['id'];e=checked(ds,key);g=labels[key]
  if independent:assert cold_math(ds,row,e)
  values={}
  for arm,oldname in [('Source','Source'),('SourceTR','SourceTR'),('Current','Ours')]:
   v=prior[key]['values'][oldname];values[arm]={k:v[k]for k in ['v','t','s','unobserved_actual_sIoU']};values[arm].update(R03=float(v['v']>=.3),R05=float(v['v']>=.5))
  with np.load(D/e['path'])as z:values['TTFT']=c.scalar(z['TTFT'],z['frame_ids'],z['TTFT_interval'],z['positions8'],row['metadata'],g,ds,independent)
  rr.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
 return rr
def report(rr):
 sys.path.insert(0,str(R/'studies/reviewer_key_controls_v1'));from statistics_v1 import pair_stats
 means={a:{k:float(np.mean([r['values'][a][k]for r in rr if r['values'][a][k]is not None]))if any(r['values'][a][k]is not None for r in rr)else None for k in rr[0]['values'][a]}for a in ARMS}
 groups={g:{a+'-'+b:pair_stats([r for r in rr if g=='all'or r['duration_group']==g],a,b)for a,b in PAIRS}for g in ['all','short_le5s','long_gt5s']}
 return dict(queries=len(rr),parents=len({r['parent']for r in rr}),means=means,paired_by_duration=groups,config_frozen_before_target=True,exposed_original128=True,cached_update_cost_not_deployment=True)
def score(ds):
 check();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and not(D/ds/'score_receipt.json').exists();torch.set_num_threads(1);rr=derive(ds);atomic(D/ds/'metrics.json',dict(private_rows=rr));atomic(P/ds/'report.json',report(rr));atomic(D/ds/'score_receipt.json',dict(new_official_checks=128,reused_official_scalars=384,metrics_sha256=sha(D/ds/'metrics.json'),report_sha256=sha(P/ds/'report.json'),seal_sha256=sha(D/ds/'sealed.json'),worker=identity()));print(ds,'TTFT pilot official score complete pending audit',flush=True)
def audit(ds):
 check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1);rr=derive(ds,True);old=read(D/ds/'metrics.json')['private_rows'];assert len(rr)==len(old)==128;mx=0
 for r,s in zip(rr,old):
  assert all(r[k]==s[k]for k in ['id','parent','duration_group'])
  for a in ARMS:
   for k,v in r['values'][a].items():
    w=s['values'][a][k];assert(v is None)==(w is None)
    if v is not None:mx=max(mx,abs(v-w));assert abs(v-w)<1e-12
 assert report(old)==read(P/ds/'report.json');ack=dict(status='all128_Input_SHA_arrays_savedweights_native_Source_features_posterior_intervals_independent_manual_scores_and_means_verified',new_manual_scalar_checks=128,max_scalar_discrepancy=mx,report_sha256=sha(P/ds/'report.json'),metrics_sha256=sha(D/ds/'metrics.json'),score_receipt_sha256=sha(D/ds/'score_receipt.json'),worker=identity());atomic(D/ds/'reporting_receipt.json',ack);atomic(P/ds/'reporting_receipt.json',ack);print(ds,'TTFT target pilot whole audit complete',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','prepare','GPU','GPU_audit','bridge','score','audit']);ap.add_argument('--dataset',choices=c.TARGETS);a=ap.parse_args();globals()[a.stage](*([a.dataset]if a.stage in ['bridge','score','audit']else[]))
