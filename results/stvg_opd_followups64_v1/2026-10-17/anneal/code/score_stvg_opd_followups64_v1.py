"""GT-free structural/math root, then strictly postseal common-support scoring."""
from pathlib import Path
import gc,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_followups64_v1 import BASE,read,sha,put,verify
def root(family):
 import torch,numpy as np
 from scripts.stvg_target_teacher_student_v1 import _tree_equal
 from scripts import run_stvg_adjacent_motion_student_v1 as h
 from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
 from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates
 verify();bar=read(BASE/family/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==64 and not bar['GT_read'];rows={r['case_id']:r for r in read(BASE/'INPUT_LOCK.json')['records']};checks=0;stage=[];usage=[]
 for f in bar['files']:
  assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);cid=f['case_id'];r=rows[cid];assert d['source_case_sha256']==r['source_case_sha256'] and d['reference_record']==r['reference_record'];old=torch.load(ROOT/r['old_student_path'],map_location='cpu',weights_only=False);anchors=h._anchors(r['reference_record'],r['native_frame_ids'],torch);ap=r['original_anchor_positions'];e=torch.stack([a['box'] for a in anchors]) if anchors else None
  for key,b in [('P0_F','T'),('P0_O','TS_M2')]:assert np.array_equal(d['arms'][key]['boxes'],old['arms'][b]['boxes']) and d['arms'][key]['interval']==old['arms'][b]['interval'];checks+=1
  for name,pred in d['arms'].items():assert pred['interval']==r['final_Temporal_interval'] and np.isfinite(pred['boxes']).all();checks+=1
  for arm,fit in d['fits'].items():
   if fit is None:assert np.array_equal(d['arms'][arm]['boxes'],d['arms']['P0_F']['boxes']);continue
   assert fit['active_parameters']==1792 and fit['protection_positions']==r['protection_positions'];actual=d['readouts'][arm];assert torch.equal(actual['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu()) and np.array_equal(h._xyxy(actual['boxes']),d['arms'][arm]['boxes']);previous=None;initial_iou=None
   for j,p in enumerate(fit['path']):
    if previous is not None:assert _tree_equal(p['state_before'],previous['state_after']) and _tree_equal(p['optimizer_before'],previous['optimizer_after'])
    previous=p
    if p.get('rollout') is not None:
     sigma=p['sigma'] if family=='anneal' else fit['config']['sigma'];assert torch.equal(mean_coordinates(p['boxes_before'][fit['observation_positions']].cuda()).cpu(),p['rollout']['mean']);pol=p['rollout'];mean=pol['mean'];logp=-(pol['samples']-mean[:,None,:]).square().sum(-1)/(2*sigma*sigma)-2*math.log(2*math.pi*sigma*sigma);loss=-(pol['weights']*logp).sum(-1).mean();assert math.isclose(float(loss),p['loss'],rel_tol=2e-5,abs_tol=2e-5);checks+=1
    if p['boxes_before'] is None:continue
    before=aligned_iou(p['boxes_before'][ap].cuda(),e.cuda()).cpu();after=aligned_iou(p['boxes_after'][ap].cuda(),e.cuda()).cpu();assert torch.equal(before,p['reference_iou_before'])
    if initial_iou is None:initial_iou=before
    for candidate in p['candidates']:
     ci=aligned_iou(candidate['boxes'][ap].cuda(),e.cuda()).cpu();assert torch.equal(ci,candidate['anchor_IoU']);assert candidate['reference_pass']==(float((ci.cuda()-before.cuda()).mean())>=-1e-7);checks+=1
    if p['rollback']:assert _tree_equal(p['state_before'],p['state_after']) and _tree_equal(p['optimizer_before'],p['optimizer_after'])
    if family=='anneal':
     assert not p['skipped'] or p['reason']=='no_information';assert p['sigma']==fit['scale_schedule'][j];assert p['noise_slot']==fit['noise_slots'][j]
    move=mean_coordinates(p['boxes_after'][ap])-mean_coordinates(p['boxes_before'][ap]);u=r['protection_positions'];stage.append(dict(case_id=cid,dataset=d['dataset'],arm=arm,stage=j,sigma=p.get('sigma',fit['config']['sigma']),accepted=p['accepted'],rollback=p['rollback'],proposal_count=len(p['candidates']),mean_reference_iou_before=float(before.mean()),mean_reference_iou_after=float(after.mean()),actual_anchor_logit_displacement_rms=float(move.square().mean().sqrt()),actual_nonanchor_box_displacement_rms=float((p['boxes_after'][u]-p['boxes_before'][u]).square().mean().sqrt()) if u else None,minimum_anchor_delta_from_initial=float((after-initial_iou).min()),high_initial_reference_anchor_count=int((initial_iou>=.5).sum()),high_initial_reference_worst_delta=float((after-initial_iou)[initial_iou>=.5].min()) if bool((initial_iou>=.5).any()) else None,GT_read=False))
   if family=='anneal':assert len(fit['path'])==3
   usage.append(dict(case_id=cid,dataset=d['dataset'],arm=arm,actual_fit=True,accepted_steps=fit['optimizer_steps'],proposal_count=fit['proposed_optimizer_steps'],reward_positions=len(fit['observation_positions']),nominal_reward_actions=32*len(fit['observation_positions'])*len(fit['path']),actual_rollout_actions=32*len(fit['observation_positions'])*sum(p.get('rollout') is not None for p in fit['path']),original_protection_count=len(fit['protection_positions']),decoder_calls=fit['decode_calls_total'],solver_calls=fit['solver_calls'],solver_seconds=fit['solver_seconds_total'],fit_wall_seconds=fit['wall_seconds'],GT_read=False));checks+=1
  del d,old;gc.collect()
 if family=='anneal':assert read(BASE/family/'QUALIFICATION_COMPLETION.json')['count']==2
 put(BASE/family/'ROOT_READBACK.json',dict(status='pass',count=64,independent_checks=checks,parameters_optimizer_rollback_and_next_stage_chain=True,same_original_sparse_acceptance_and_protection=True,all_final_intervals_and_complete_decodes_verified=True,GT_read=False));put(BASE/family/'ROOT_STAGE_DIAGNOSTICS.json',dict(status='complete',count=64,records=stage,usage=usage,GT_read=False));put(BASE/family/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,files=bar['files'],root_sha256=sha(BASE/family/'ROOT_READBACK.json'),GT_read=False));print(family.upper()+'_ROOT64_PASS',flush=True)

def score(family):
 import torch,numpy as np
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 verify();b=BASE/family;bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete' and bar['count']==64 and bar['root_sha256']==sha(b/'ROOT_READBACK.json')
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];supports=[];loo=[];tick=__import__('time').perf_counter()
 for f in bar['files']:
  cid=f['case_id'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);case=cases[cid];target=targets[cid];ids=list(map(int,case['frame_ids']));alltruth=metric._truth_map(target);anchor_ids={ids[p] for p in d['original_anchor_positions']};native=set(ids);strata=dict(anchor=[i for i in alltruth if i in anchor_ids],native_nonanchor=[i for i in alltruth if i in native and i not in anchor_ids],interpolated=[i for i in alltruth if i not in native])
  for name,pred in d['arms'].items():
   vals,_=metric._score_student_arm(pred,case,target,set(d['original_anchor_positions']),official,independent);vals.update(R_vIoU_at_0p3=float(vals['m_vIoU']>=.3),R_vIoU_at_0p5=float(vals['m_vIoU']>=.5))
   for group,known in strata.items():
    if not known:v=None
    else:
     z=truth.score_tube_xyxy(pred['boxes'],ids,int(case['input']['width']),int(case['input']['height']),pred['interval'],{i:alltruth[i] for i in known},target['span'],case['dataset'],official=official,independent=independent);v=z['metrics']['m_sIoU_dense_GT']
    vals['m_sIoU_'+group+'_GT']=v
   rows.append(dict(case_id=cid,dataset=d['dataset'],arm=name,metrics=vals))
  supports.append(dict(case_id=cid,dataset=d['dataset'],known_counts={k:len(v) for k,v in strata.items()},same_support_for_every_arm=True))
  if family=='tube':
   row=next(r for r in read(BASE/'INPUT_LOCK.json')['records'] if r['case_id']==cid);p=torch.load(ROOT/row['proposal_path'],map_location='cpu',weights_only=False);q=p['leave_one_reference']
   if q:
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    io=float(aligned_iou(torch.tensor(q['heldout_prediction_cxcywh'])[None],torch.tensor(q['heldout_pseudoreference_cxcywh'])[None])[0]);loo.append(dict(case_id=cid,dataset=d['dataset'],heldout_pseudoreference_iou=io,not_GT=True))
  del d;gc.collect()
 summary,paired=subset_aggregate(rows);put(b/'POSTSEAL_RESULTS.json',dict(status='complete',count=64,per_domain=32,rows=rows,summary=summary,paired=paired,supports=supports,leave_one_reference_diagnostics=loo,GT_bindings=bindings,P0_R_is_external_diagnostic_not_Student_TTA=family=='tube',no_reference_in_denominator=True,GT_read=True,bootstrap=False,development_not_confirmation=True,coarse_to_fine_or_tube_feasibility_not_diffusion_effect=True));put(b/'CPU_COMPLETION.json',dict(status='complete',count=64,result_sha256=sha(b/'POSTSEAL_RESULTS.json'),wall_seconds=__import__('time').perf_counter()-tick,GT_read=True));print(family.upper()+'_POSTSEAL64_COMPLETE',flush=True)

if __name__=='__main__':{'root':root,'score':score}[sys.argv[2]](sys.argv[1])
