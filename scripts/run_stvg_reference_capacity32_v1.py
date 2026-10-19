"""Explicit GT-assisted whole-tube reference/propagation diagnosis. Not TTA."""
import os,sys,time,itertools,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_reference_capacity32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/reference_capacity32'
OLD=ROOT/'artifacts/stvg_sparse_state_route32_v1';PRE=ROOT/'artifacts/stvg_dino_residual_advantage64_v1'
ARMS=['E_MBR','Fine_center','Old_DINO_state','Candidate_combo_oracle','Local_best_mean','GT_E_known4','GT_mean_known4','GT_subset_oracle_known4','GT_E_matched','GT_state_known4','GT_state_matched','DINO_state_matched'];VER=None
CODE=['scripts/run_stvg_reference_capacity32_v1.py','protocols/stvg_reference_capacity32_v1.md']

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 pins=dict(l['pins'])
 if (D/'PREFLIGHT_METADATA_CORRECTION001.json').exists():
  amendment=read(D/'PREFLIGHT_METADATA_CORRECTION001.json');assert sha(D/'RUNTIME_LOCK.json')==amendment['original_runtime_sha256'];pins.update(amendment['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def prepare():
 assert not D.exists();D.mkdir();P.mkdir();rs=read(OLD/'PLAN_LOCK.json')['records'];assert len(rs)==len({r['parent']for r in rs})==32
 prior=read(OLD/'RUNTIME_LOCK.json')['pins'];files=[k for k in prior if k.startswith(('external/TA-STVG/','methods/decota_final_simplified_v1/','checkpoints/'))or k in ['scripts/run_decota_paper_main_v1.py','scripts/run_spatial_regression_alignment_v1.py','scripts/run_spatial_ssl_gpu_v1.py','scripts/run_final_simplification_v1.py']]
 files +=CODE+['vg_tta/stvg_sparse_state_route32_v1.py','vg_tta/stvg_dino_query_closedloop32_v1.py','vg_tta/stvg_dino_supported_correction32_v1.py','vg_tta/stvg_joint_robust_correction32_v1.py','vg_tta/stvg_error_state_correction_v1.py','vg_tta/tastvg_oracle_event5_v1.py','vg_tta/stvg_opd_frozen_capture_cpu007.py','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py','artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','configs/stvg_fine_center_dataset_candidate_v1.json','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json','results/stvg_error_state/2026-10-19/sparse_state_route32/RESULTS_PUBLIC.json']
 for r in rs:
  a=r['alias'];files +=[f'artifacts/stvg_dino_endpoint_confirm32_v1/{sub}/{a}.pt'for sub in ['source','E']]+[f'artifacts/stvg_dino_query_closedloop32_v1/replay_inputs/{a}.pt',f'artifacts/stvg_sparse_state_route32_v1/predictions/{a}.pt',f'artifacts/stvg_dino_residual_advantage64_v1/extracted002/{a}.pt']
 p=dict(status='locked_explicit_GT_interventions',count=32,records=rs,arms=ARMS,GT_assisted_not_deployable=True,known_GT_only_no_label_interpolation=True,rounds=2,Adam=dict(hc2=.01,vidstg=.03,betas=[.9,.999],eps=1e-8),capacity_screen_pp=5.,GPU_cap_seconds=900,CPU_phase_cap_seconds=900,deadline_epoch=time.time()+3600,created_epoch=time.time(),new_Source_backbone_capture_experts_DINO_training=0,no_OPD_parameter_search=True,formal64_excluded=True,other_pauses_preserved=True)
 put(D/'PLAN_LOCK.json',p);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in sorted(set(files))}));print('locked32 explicit offline GT only',flush=True)

def enumerate_means(base,positions,pools,truth,row,span,iv,ds,incumbent=None,incumbent_assignment=None):
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 residual=[encode(v)-encode(base)[pos]for pos,v in zip(positions,pools)];land=[];best=incumbent.copy()if incumbent is not None else base.copy();bv=DenseTube(best,row,truth,span,clip=ds=='hc2').score(iv)['v'];which=incumbent_assignment
 for assign in itertools.product(*[range(-1,len(v))for v in residual]):
  active=[v[k]for v,k in zip(residual,assign)if k>=0];delta=np.mean(active,axis=0)if active else np.zeros(4);out=apply(base,delta);v=DenseTube(out,row,truth,span,clip=ds=='hc2').score(iv)['v'];land.append(dict(assignment=list(assign),delta=delta,v=v))
  if v>bv+1e-14:best=out;bv=v;which=list(assign)
 return dict(boxes=best,v=bv,assignment=which,landscape=land)

def inputs():
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.stvg_error_state_correction_v1 import encode
 from vg_tta.stvg_joint_robust_correction32_v1 import apply
 from vg_tta.stvg_dino_supported_correction32_v1 import estimate
 from vg_tta.stvg_dino_residual_advantage64_v1 import normalized_truth,ious
 torch.set_num_threads(4);p=verify(True);put(D/'OFFLINE_GT_INPUT_ACCESS.json',dict(authorized_offline_GT_intervention=True,GT_used_for_oracle_and_state_labels=True,time=time.time(),protocol_sha256=sha(ROOT/CODE[1])))
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');files=[];start=time.perf_counter()
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];x=load(PRE/'extracted002'/(a+'.pt'));src=load(ROOT/f'artifacts/stvg_dino_endpoint_confirm32_v1/source/{a}.pt');e=load(ROOT/f'artifacts/stvg_dino_endpoint_confirm32_v1/E/{a}.pt');old=load(OLD/'predictions'/(a+'.pt'));b=src['Source'].double().numpy();E=e['E'];truth=gt[ds][0][q];span=gt[ds][1][q];row=plans[ds][q];iv=e['interval'];assert np.array_equal(E,x['E'])and np.array_equal(old['outputs']['E_MBR'],E)
  ff=[f for f in x['frames']if f['frame_admitted']];pos=[f['position']for f in ff];pools=[f['retained_boxes']for f in ff];scores=[f['retained_scores']for f in ff];pp=[encode(v)-encode(b)[po]for po,v in zip(pos,pools)];setting=cal['source_domains']['vidstg'if ds=='hc2'else'hc2'];rep=estimate(pp,scores,setting['scale'],setting['truncation'],1/16,False);assert np.array_equal(apply(b,rep['delta']),E)and rep['selected']==e['estimate']['selected']
  oracle=enumerate_means(b,pos,pools,truth,row,span,iv,ds,E,rep['selected']);actual_delta=rep['delta'];assert any(np.max(abs(v['delta']-actual_delta))<1e-12 for v in oracle['landscape'])
  known=[po for po in e['planned_positions']if e['frame_ids'][po]in truth];targets=np.array([normalized_truth(truth[e['frame_ids'][po]],row)for po in known]).reshape(-1,4);tm={po:v for po,v in zip(known,targets)};matched=[po for po in old['positions']if po in tm];mt=np.array([tm[po]for po in matched]).reshape(-1,4);dm=np.array([old['targets'][old['positions'].index(po)]for po in matched]).reshape(-1,4)
  outputs=dict(E_MBR=E,Fine_center=old['outputs']['Fine_center'],Old_DINO_state=old['outputs']['Direct_anchor'],Candidate_combo_oracle=oracle['boxes']);gt_est={}
  for key,ps,ts in [('known4',known,targets),('matched',matched,mt)]:
   residual=[(encode(t[None])-encode(b)[po])for po,t in zip(ps,ts)];est=estimate(residual,[np.ones(1)for _ in ps],setting['scale'],setting['truncation'],1/16,False);outputs['GT_E_'+key]=apply(b,est['delta'])if ps else E.copy();gt_est[key]=est
  if known:
   outputs['GT_mean_known4']=apply(b,np.mean(encode(targets)-encode(b)[known],axis=0));gg=enumerate_means(b,known,[v[None]for v in targets],truth,row,span,iv,ds);outputs['GT_subset_oracle_known4']=gg['boxes']
  else:outputs['GT_mean_known4']=E.copy();outputs['GT_subset_oracle_known4']=E.copy();gg=None
  local=[]
  for f in ff:
   fid=e['frame_ids'][f['position']]
   if fid in truth:local.append(encode(f['retained_boxes'])[int(ious(f['retained_boxes'],truth[fid],row,ds).argmax())]-encode(b)[f['position']])
  outputs['Local_best_mean']=apply(b,np.mean(local,axis=0))if local else E.copy()
  z=dict(alias=a,dataset=ds,frame_ids=e['frame_ids'],planned_positions=e['planned_positions'],interval=iv,Source=b,E=E,outputs=outputs,positions=known,targets=targets,matched_positions=matched,matched_GT_targets=mt,matched_DINO_targets=dm,original_positions=old['positions'],original_targets=old['targets'],candidate_positions=pos,candidate_oracle=oracle,GT_subset_oracle=gg,GT_estimates=gt_est,GT_assisted_diagnostic=True,GT_free=False,baseline_E_replay_exact=True)
  files.append(save(D/'inputs'/(a+'.pt'),z));print(a,'GT inputs and CPU oracle sealed',len(oracle['landscape']),len(known),len(matched),flush=True)
 verify(True);put(D/'INPUT_BARRIER.json',dict(status='complete',count=32,files=files,GT_assisted_diagnostic=True,CPU_seconds=time.perf_counter()-start,no_GPU_models_or_experts=True));print('GT-assisted input32 complete',flush=True)

def measure():
 import torch
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_sparse_state_route32_v1 import Decoder,fit,noise
 p=verify(True);bar=read(D/'INPUT_BARRIER.json');assert bar['count']==32
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 lease=gpu();start=time.perf_counter();files=[];domain=None;model=None;dec=None;done=False;pilots=[]
 def check():
  verify();assert time.time()<p['deadline_epoch'];assert time.perf_counter()-start+3<p['GPU_cap_seconds'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  records=[next(r for r in p['records']if r['alias']==a)for a in ['NH001','NV001']]+[r for r in p['records']if r['alias']not in ['NH001','NV001']]
  for ix,r in enumerate(records):
   check();a=r['alias'];ds=r['dataset'];tick=time.perf_counter();x=load(D/'inputs'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));rc=load(ROOT/f'artifacts/stvg_dino_query_closedloop32_v1/replay_inputs/{a}.pt')
   if domain!=ds:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict())
   assert torch.equal(rc['Source'],torch.as_tensor(x['Source']).to(rc['Source']));inp=transfer(rc['spatial_inputs'],'cuda');dec=Decoder(model.ground_decoder.decoder,inp,rc['Source'],x['E'],check);assert dec.scale==old['sigma']and mh==old['model_hash'];probe=dec.gradient_check(a)if ix<2 else None;fits={};dedup={};outputs=x['outputs'].copy();computes=[]
   specs=[('GT_state_known4',x['positions'],x['targets']),('GT_state_matched',x['matched_positions'],x['matched_GT_targets']),('DINO_state_matched',x['matched_positions'],x['matched_DINO_targets'])]
   for arm,ps,ts in specs:
    k=(tuple(ps),np.asarray(ts).tobytes());reused=False
    if arm=='DINO_state_matched'and ps==old['positions']and np.array_equal(ts,old['targets']):f=old['fits']['Direct_anchor'];reused=True;source='exact_old_Direct_anchor'
    elif k in dedup:f=dedup[k];reused=True;source='identical_support_labels_within_query'
    else:
     f=fit(dec,noise(a),ps,ts,'Direct_anchor',p['Adam'][ds],check);dedup[k]=f;source='actual_two_step_decoder_gradient'
    f=dict(f);f['GT_used']=(arm!='DINO_state_matched');fits[arm]=dict(**f,oracle_GT_input=(arm!='DINO_state_matched'),GT_assisted_diagnostic=True,GT_used_for_this_diagnostic=True,reused_exact=reused,reuse_source=source);outputs[arm]=f['final'];computes.append(dict(arm=arm,source=source,updated_steps=sum(t['updated']for t in f['rounds'])))
   assert set(outputs)==set(ARMS)and all(np.isfinite(v).all()and(v[:,2:]>0).all()for v in outputs.values());assert state_hash(model.state_dict())==mh and all(v.grad is None and not v.requires_grad for v in model.parameters())
   cost=dict(seconds=time.perf_counter()-tick,decoder_trajectory_calls=dec.calls,PosDecoder_calls=2*dec.calls,decoder_seconds=dec.model_seconds,backward_calls=dec.backward_calls,probe_backward_calls=int(probe is not None),prefix_checks=dec.prefix_checks,arms=computes)
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],Source=x['Source'],E=x['E'],outputs=outputs,fits=fits,sigma=dec.scale,model_hash=mh,gradient_probe=probe,compute=cost,GT_assisted_diagnostic=True,GT_free=False,zero_Source_E_exact=True)));dec.close();dec=None;del inp,rc,x,old,fits;gc.collect();torch.cuda.empty_cache();print(a,ix+1,round(time.perf_counter()-start,3),cost['decoder_trajectory_calls'],flush=True)
   if ix<2:pilots.append(cost['seconds'])
   if ix==1:
    estimate=time.perf_counter()-start+15*sum(pilots)+60;put(D/'PILOT2_TIMING.json',dict(case_seconds=pilots,total_estimate_seconds=estimate,within_900=estimate<900));assert estimate<900,'cost pilot exceeds fixed budget'
  check();verify(True);put(D/'GLOBAL_OUTPUT_BARRIER.json',dict(status='complete',count=32,files=files,all_GT_assisted_outputs_sealed=True,not_GT_free=True,GPU_allocation_seconds=time.perf_counter()-start+3,actual_worker_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_Source_DINO_experts_training=0));done=True
 finally:
  if dec is not None:dec.close()
  put(D/'GPU_COST_RECEIPT.json',dict(status='released',complete=done,count=len(files),GPU_allocation_seconds=time.perf_counter()-start+3));lease.close()
 print('all32 GT assisted outputs sealed',flush=True)

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_sparse_state_route32_v1 import box_loss
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_OUTPUT_BARRIER.json');assert bar['count']==32
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 start=time.perf_counter();gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];land=[]
 for r in p['records']:
  verify();a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];x=load(D/'inputs'/(a+'.pt'));z=load(D/'predictions'/(a+'.pt'));m=metrics(z,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);assert abs(m['values']['Candidate_combo_oracle']['v']-x['candidate_oracle']['v'])<1e-10;assert m['values']['Candidate_combo_oracle']['v']>=m['values']['E_MBR']['v']-1e-10
  old=next(v for v in read(ROOT/'results/stvg_error_state/2026-10-19/sparse_state_route32/RESULTS_PUBLIC.json')['rows']if v['alias']==a)
  for new,prior in [('E_MBR','E_MBR'),('Fine_center','Fine_center'),('Old_DINO_state','Direct_anchor')]:
   for k,v in m['values'][new].items():assert v is None and old['values'][prior][k]is None or abs(v-old['values'][prior][k])<1e-10
  fits={k:dict(updated_steps=sum(t['updated']for t in f['rounds']),loss_before=f['rounds'][0]['loss_before'],loss_after=f['rounds'][-1]['loss_after'],state_L2=float(np.linalg.norm(f['path'][-1]['state'])),output_L2_from_E=float(np.linalg.norm(f['final']-z['E'])),reused_exact=f['reused_exact'],reuse_source=f['reuse_source'])for k,f in z['fits'].items()};diag={}
  for arm in ['GT_E_known4','GT_mean_known4','GT_subset_oracle_known4','GT_state_known4']:
   ps=x['positions'];ts=x['targets'];diag[arm]=dict(correct_anchor_loss=float(box_loss(torch.tensor(z['outputs'][arm]),ps,ts)))
  land.append(dict(alias=a,dataset=ds,candidate_combinations=[dict(assignment=v['assignment'],v=v['v'])for v in x['candidate_oracle']['landscape']],chosen_assignment=x['candidate_oracle']['assignment'],GT_subset_combinations=[]if x['GT_subset_oracle']is None else[dict(assignment=v['assignment'],v=v['v'])for v in x['GT_subset_oracle']['landscape']]))
  rows.append(dict(alias=a,dataset=ds,known_planned_GT=len(x['positions']),matched_GT_DINO_support=len(x['matched_positions']),original_DINO_support=len(x['original_positions']),candidate_admitted_frames=len(x['candidate_positions']),candidate_combinations=len(x['candidate_oracle']['landscape']),GT_E_estimator_support={k:v['support_frames']for k,v in x['GT_estimates'].items()},fits=fits,GT_anchor_fit=diag,compute=z['compute'],gradient_probe=z['gradient_probe'],**m));print(a,'scored offline oracle',flush=True)
 pairs=[(a,'E_MBR')for a in ARMS[1:]]+[('GT_state_matched','DINO_state_matched'),('GT_state_known4','GT_state_matched'),('GT_state_known4','GT_E_known4'),('GT_subset_oracle_known4','GT_E_known4'),('Candidate_combo_oracle','Local_best_mean')];eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(parents=len(rr),means={a:{k:mean([r['values'][a][k]for r in rr])for k in eps}for a in ARMS},paired={a+'_minus_'+b:{k:paired([100*(r['values'][a][k]-r['values'][b][k])if r['values'][a][k]is not None and r['values'][b][k]is not None else None for r in rr])for k in eps}for a,b in pairs},known_GT_anchor_frames=sum(r['known_planned_GT']for r in rr),matched_support_frames=sum(r['matched_GT_DINO_support']for r in rr),empty_GT_known_queries=sum(r['known_planned_GT']==0 for r in rr),empty_matched_queries=sum(r['matched_GT_DINO_support']==0 for r in rr))
 put(P/'RESULTS_PUBLIC.json',dict(status='complete_GT_assisted_diagnostic_only',count=32,rows=rows,groups=groups,all_parent_denominators_preserved=True,GT_used_for_generation_selection_and_state_supervision=True,not_deployable_or_OPD_gain=True,not_independent_confirmation=True));put(P/'CANDIDATE_LANDSCAPES_PUBLIC.json',dict(status='complete',rows=land,GT_oracle=True,coordinates_excluded=True));put(P/'COST_BREAKDOWN.json',dict(CPU_oracle_input_seconds=read(D/'INPUT_BARRIER.json')['CPU_seconds'],CPU_score_seconds=time.perf_counter()-start,GPU_allocation_seconds=read(D/'GPU_COST_RECEIPT.json')['GPU_allocation_seconds'],actual_decoder_trajectory_calls=sum(r['compute']['decoder_trajectory_calls']for r in rows),actual_PosDecoder_calls=sum(r['compute']['PosDecoder_calls']for r in rows),backward_calls=sum(r['compute']['backward_calls']for r in rows),total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_experts_training=0,old_cache_cost_excluded=True));put(D/'SCORE_COMPLETION.json',dict(status='complete',count=32));verify(True);print('all oracle score32 complete',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','inputs','measure','score']);args=ap.parse_args()
 try:globals()[args.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=args.stage,error_type=type(e).__name__,error=str(e),partials_preserved=True));raise
