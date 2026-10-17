"""Bounded restoration check of the historical P7-A DINO + online Gaussian OPD.

The original predictor and fitter are imported unchanged. No prior controller is
resumed. The 32 fixed chronological-prefix predictions seal before CPU scoring.
"""
from pathlib import Path
import argparse,collections,gc,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_dino_rollback32_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v,mutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists()and not mutable:raise FileExistsError(p)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def save(p,v):
 import torch
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();t=p.with_name(p.name+'.tmp');torch.save(v,t);t.replace(p)
 f=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size);write(p.with_suffix('.receipt.json'),dict(status='sealed',**f,GT_read=False));return f
def status(stage,done,**kw):write(BASE/'STATUS.json',dict(status='running',stage=stage,done=done,total=32,**kw),True)
def prepare():
 assert not(BASE/'PLAN_LOCK.json').exists()
 d=read(ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2/DESIGN_LOCK.json');records=[]
 for ds in ['hc2','vidstg']:
  s=d['stages']['P0_'+ds];order=next(iter(s['orders']));seq=s['orders'][order][:16]
  plan=ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json';p=read(plan)
  for i,q in enumerate(seq):
   r=p['rows'][q];records.append(dict(alias=('H'if ds=='hc2'else'V')+f'{i+1:03d}',dataset=ds,query_ordinal=q,sequence_position=i,original_order=order,parent=r['source'],row_key=r['key'],plan_path=str(plan.relative_to(ROOT)),plan_sha256=sha(plan)))
 cfg=read(ROOT/'methods/decota_spatial_opd_v1/configs.json');assert cfg['datasets']['hc2']['config']==dict(lr=.01,sigma=.025,tau=.05,steps=40,writeback=1/16,samples=32);assert cfg['datasets']['vidstg']['config']==dict(lr=.03,sigma=.1,tau=.25,steps=10,writeback=1/8,samples=32)
 own=['scripts/verify_stvg_dino_rollback_v1.py','methods/stvg_opd_p7a_v1/predictor.py','methods/decota_spatial_opd_v1/configs.json','methods/decota_final_simplified_v1/config.py','methods/decota_final_simplified_v1/objectives.py','methods/decota_final_simplified_v1/observations.py','vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/decota_spatial_opd_tunable_audit_v1.py','scripts/stvg_pipeline_speed_v1.py']
 write(BASE/'PLAN_LOCK.json',dict(status='locked_before_model_and_GT',count=32,records=records,entrypoint='methods.stvg_opd_p7a_v1.predictor.P7AOPDPredictor',configurations={ds:cfg['datasets'][ds]['config']for ds in ['hc2','vidstg']},temporal=dict(center_fraction=dict(hc2=1.,vidstg=.5),prior_weight=.1,epsilon=1e-6,two_offset_physical_envelope=True),one_frozen_DINO=True,Native_I0_observation_projected_I1_output=True,online_LN_carry_Query_and_Adam_reset=True,verification_only_no_official_full_restart=True,cohort='historically exposed original locked P0 order1 first16 per target; no outcome selection',GT_read=False))
 write(BASE/'RUNTIME_LOCK.json',dict(status='locked',pins={p:sha(ROOT/p)for p in own},plan_sha256=sha(BASE/'PLAN_LOCK.json'),GT_read=False));print('prepared fixed32')
def projection_check(proj,config):
 import numpy as np
 intervals=[];error=0.
 for o in proj['offsets']:
  z=o['logp0'].detach().cpu().numpy();x=o['raw_logits'].detach().cpu().numpy();ij=o['ij'].detach().cpu().numpy();omega=o['omega'].detach().cpu().numpy()
  u=(x-config.center_fraction*np.median(x))/(np.median(abs(x-np.median(x)))+config.epsilon)
  logp=-np.logaddexp(0.,-u);logn=-np.logaddexp(0.,u)
  scores=np.array([(omega[s:e+1]*logp[s:e+1]).sum()+(omega[:s]*logn[:s]).sum()+(omega[e+1:]*logn[e+1:]).sum()+config.prior_weight*z[k]for k,(s,e)in enumerate(ij.T)])
  saved=o['score'].detach().cpu().numpy();error=max(error,float(abs(scores-saved).max()));assert error<1e-10
  assert int(scores.argmax())==int(saved.argmax());intervals.append(o['interval'])
 return dict(status='pass',max_direct_enumeration_error=error,selected_offsets_and_physical_envelope_exact=True)
def worker():
 import torch,numpy as np
 from methods.stvg_opd_p7a_v1.predictor import P7AOPDPredictor
 from methods.decota_final_simplified_v1.observations import SpatialExpert
 from methods.decota_final_simplified_v1.config import EXPERT_SHA256,EXPERT_SNAPSHOT,MethodConfig
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from scripts.run_decota_paper_main_v1 import gpu,model_for,frames_for,read_row
 from scripts.stvg_pipeline_speed_v1 import full_prefix_readout
 from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit
 from main_method.runtime import resource_guard
 import vg_tta.stvg_opd_frozen_mask_cache012 as capture
 assert 'tubedetr'in sys.prefix
 lock=read(BASE/'RUNTIME_LOCK.json');assert sha(BASE/'PLAN_LOCK.json')==lock['plan_sha256'];assert all(sha(ROOT/p)==v for p,v in lock['pins'].items())
 assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors')==EXPERT_SHA256
 lease=gpu();dino=SpatialExpert(ROOT/EXPERT_SNAPSHOT);expert_hash=state_hash(dino.model.state_dict());plan=read(BASE/'PLAN_LOCK.json');files=[];casefiles=[];reviews=[];model=None;domain=None;predictor=None;seen={};cache=collections.OrderedDict();started=time.time();total_DINO=0;original_capture=capture.frozen_forward
 def tracked(*args,**kw):
  result=original_capture(*args,**kw);seen['value']=result;return result
 capture.frozen_forward=tracked
 try:
  for i,r in enumerate(plan['records']):
   resource_guard();assert time.time()-started<3600
   if domain!=r['dataset']:
    if model is not None:del predictor,model;gc.collect();torch.cuda.empty_cache()
    domain=r['dataset'];model=model_for('vidstg'if domain=='hc2'else'hcstvg2');before_hash=state_hash(model.state_dict());predictor=P7AOPDPredictor(model,dino,dataset=domain);predictor.reset_stream();cache.clear()
   assert sha(ROOT/r['plan_path'])==r['plan_sha256'];row=read_row(domain,r['query_ordinal']);frames,ids,_=frames_for(domain,row,cache);prior=detached(predictor.previous,'cpu')if predictor.previous is not None else None
   torch.cuda.synchronize();tick=time.perf_counter();result=predictor.predict(frames,ids,row['input'],parses=row['parses'],key=row['key']);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
   batch,records,base=seen.pop('value');fit=result['spatial_audit'];native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
   assert result['Native_interval']==native['physical_interval']and result['Native_observation_indices']==native['indices']
   assert sum(v.numel()for v in fit['initial'].values())==1792
   assert torch.count_nonzero(fit['initial']['spatial.query_residual'])==0
   for name,v in fit['initial'].items():
    expected=torch.zeros_like(v)if name=='spatial.query_residual'else(prior[name]if prior is not None else base.initial[name].detach().cpu())
    assert torch.equal(v,expected),('online_initial',name)
    wanted=torch.zeros_like(v)if name=='spatial.query_residual'else v+(fit['state'][name]-v)*predictor.config['writeback']
    assert torch.equal(predictor.previous[name],wanted),('carry',name)
   arithmetic=audit(fit,result['expert']);pc=projection_check(result['projection'],predictor.temporal_config)
   full=full_prefix_readout(base,ids,records,fit['state']);assert torch.equal(full['boxes'],result['boxes'])
   assert result['physical_interval']==[min(o['interval'][0]for o in result['projection']['offsets']),max(o['interval'][1]for o in result['projection']['offsets'])]
   def xyxy(b):
    b=b.detach().cpu().double();return torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1).numpy()
   c=dict(case_id=r['alias'],dataset=domain,query_ordinal=r['query_ordinal'],frame_ids=np.asarray(ids),input={k:v for k,v in row['input'].items()if k!='caption'},Native_interval=native['physical_interval'],native_boxes_xyxy_normalized=xyxy(native['boxes']))
   arms=dict(Frozen_native=dict(boxes=xyxy(native['boxes']),interval=native['physical_interval']),Frozen_projection=dict(boxes=xyxy(native['boxes']),interval=result['physical_interval']),Before_projection=dict(boxes=xyxy(fit['before']),interval=result['physical_interval']),OPD_native_readout=dict(boxes=xyxy(result['boxes']),interval=native['physical_interval']),Full_P7A=dict(boxes=xyxy(result['boxes']),interval=result['physical_interval']))
   casefiles.append(save(BASE/'source'/(r['alias']+'.pt'),c));files.append(save(BASE/'predictions'/(r['alias']+'.pt'),dict(case_id=r['alias'],dataset=domain,arms=arms,positions=fit['positions'],fit=detached(fit,'cpu'),expert=detached(result['expert'],'cpu'),projection=detached(result['projection'],'cpu'),prior=prior,committed=detached(predictor.previous,'cpu'),full_frozen_prefix_decoder_readout=detached(full,'cpu'),projection_check=pc,math_audit=arithmetic,compute=dict(complete_predict_seconds=elapsed,new_DINO_calls=result['expert']['new_DINO']),GT_read=False)))
   reviews.append(dict(dataset=domain,status='pass',active_parameters=1792,observed_anchors=len(fit['positions']),configured_rounds=fit['config']['steps'],actual_gradient_calls=fit['gradient_calls'],math=arithmetic,projection=pc,QR_and_Adam_reset=True,online_LN_writeback_exact=True,complete_both_decoder_readout_exact=True))
   assert state_hash(model.state_dict())==before_hash
   total_DINO+=result['expert']['new_DINO'];status('DINO_P7A_validation',i+1,new_DINO_calls=total_DINO,GT_read=False)
   del frames,base,batch,result,fit,full;gc.collect();torch.cuda.empty_cache()
  assert state_hash(dino.model.state_dict())==expert_hash
  write(BASE/'SOURCE_BARRIER.json',dict(status='complete',count=32,files=casefiles,GT_read=False));write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_predictions_sealed_before_GT=True,GT_read=False));write(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,checks=reviews,actual_new_Source_captures=32,actual_new_spatial_fits=32,original_predictor_and_fitter_unchanged=True,GT_read=False));status('prediction_sealed_pending_CPU_score',32,GT_read=False)
 finally:
  capture.frozen_forward=original_capture;lease.close()
def score():
 import torch
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 assert read(BASE/'ROOT_READBACK.json')['status']=='pass';bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32
 def load(f):assert sha(ROOT/f['path'])==f['sha256'];return torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
 cases={};pred=[]
 for a,b in zip(read(BASE/'SOURCE_BARRIER.json')['files'],bar['files']):
  c=load(a);p=load(b);assert c['case_id']==p['case_id'];cases[c['case_id']]=c;pred.append(p)
 targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[]
 for p in pred:
  c=cases[p['case_id']]
  for arm,v in p['arms'].items():
   m,_=metric._score_student_arm(v,c,targets[p['case_id']],set(p['positions']),official,independent);m.update(R_vIoU_at_0p3=float(m['m_vIoU']>.3),R_vIoU_at_0p5=float(m['m_vIoU']>.5));rows.append(dict(case_id=p['case_id'],dataset=c['dataset'],arm=arm,metrics=m))
 summary,paired=subset_aggregate(rows);write(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,rows=rows,summary=summary,paired=paired,GT_only_after_entire_32_prediction_seal=True,not_official_full_or_independent_confirmation=True));write(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));status('verification_complete_root_publication_pending',32,GT_read=True)
 for x in summary:
  if x['metric']=='m_vIoU':print(x['dataset'],x['arm'],100*x['mean'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','score']);a=p.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  if BASE.exists():write(BASE/('FAILURE_'+a.stage+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),outputs_preserved=True,GT_read=a.stage=='score'))
  raise
