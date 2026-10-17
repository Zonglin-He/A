"""Saved QCQP math/rollback/full-readout audit, then all32 postseal scoring."""
from pathlib import Path
import gc,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_output_budget_qp_v1 import BASE,read,sha,put,verify

def root():
 import torch,numpy as np
 from scripts.stvg_target_teacher_student_v1 import _tree_equal
 from scripts import run_stvg_adjacent_motion_student_v1 as h
 from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
 verify();bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json');assert bar['count']==32 and read(BASE/'QUALIFICATION_COMPLETION.json')['count']==2;rr={r['case_id']:r for r in read(BASE/'INPUT_LOCK.json')['records']};usage=[];checks=0
 for f in bar['files']:
  assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);r=rr[d['case_id']];old=torch.load(ROOT/r['old_student_path'],map_location='cpu',weights_only=False);assert d['reference_record']==r['reference_record']
  for arm,key in [('Source_T','T'),('M0','TS_M2')]:assert np.array_equal(d['arms'][arm]['boxes'],old['arms'][key]['boxes']) and d['arms'][arm]['interval']==old['arms'][key]['interval']
  for arm,fit in d['fits'].items():
   assert d['arms'][arm]['interval']==d['arms']['Source_T']['interval'];checks+=1
   if fit is None:assert np.array_equal(d['arms'][arm]['boxes'],d['arms']['Source_T']['boxes']);usage.append(dict(case_id=d['case_id'],dataset=d['dataset'],arm=arm,actual_fit=False,no_fit_reason=d['no_fit_reason'],GT_read=False));continue
   assert fit['active_parameters']==1792 and fit['protection_positions']==r['protection_positions'];full=d['readouts'][arm];assert torch.equal(full['boxes'].cpu(),fit['final_cached_decoder_boxes'].cpu()) and np.array_equal(h._xyxy(full['boxes']),d['arms'][arm]['boxes']);prev=None
   for p in fit['path']:
    if prev is not None:assert _tree_equal(p['state_before'],prev['state_after']) and _tree_equal(p['optimizer_before'],prev['optimizer_after'])
    prev=p
    if p['rollback']:assert _tree_equal(p['state_before'],p['state_after']) and _tree_equal(p['optimizer_before'],p['optimizer_after'])
    if p['solver']:
     q=p['solver'];A,U,v,adam=[np.asarray(q[k]) for k in ('JA','JU','target','adam')];z=np.asarray(q['solution']);lam=q['lambda_dual'];rhs=adam if arm=='M1' else A.T@v;g=1.;grad=g*z-rhs+lam*U.T@(U@z)+(A.T@(A@z) if arm=='M2' else 0);res=np.linalg.norm(grad)/max(1.,np.linalg.norm(rhs));assert res<1e-6 and float(np.sum((U@z)**2)/2)<=.01*(1+1e-8);assert np.allclose(z,p['filtered_delta'].numpy(),atol=1e-7,rtol=1e-6);checks+=1
    for c in p['candidates']:
     assert c['accepted']==(c['finite'] and c['reference_pass'] and c['output_budget_pass'])
     if c['accepted']:assert c['actual_output_KL']<=.01*(1+1e-7)
     checks+=1
   usage.append(dict(case_id=d['case_id'],dataset=d['dataset'],arm=arm,actual_fit=True,accepted_steps=fit['optimizer_steps'],proposed_steps=fit['proposed_optimizer_steps'],solver_calls=fit['solver_calls'],solver_seconds=fit['solver_seconds_total'],fit_wall_seconds=fit['wall_seconds'],GT_read=False))
  del d,old;gc.collect()
 put(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,independent_checks=checks,all_parameter_Adam_rollbacks_and_full_decode_exact=True,all_accepted_actual_KL_budgets_checked=True,GT_read=False));put(BASE/'ROOT_USAGE.json',dict(status='complete',count=32,records=usage,GT_read=False));put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=bar['files'],root_sha256=sha(BASE/'ROOT_READBACK.json'),GT_read=False));print('OUTPUT_BUDGET32_ROOT_PASS')

def score():
 import torch
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 verify();bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['root_sha256']==sha(BASE/'ROOT_READBACK.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];valid=set()
 for f in bar['files']:
  assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);cid=d['case_id'];case=cases[cid];pos=[a['position'] for a in __import__('scripts.run_stvg_adjacent_motion_student_v1',fromlist=['_anchors'])._anchors(d['reference_record'],d['frame_ids'],torch)]
  if d['pre_adaptation_chart_valid']:valid.add(cid)
  for arm,p in d['arms'].items():
   v,_=metric._score_student_arm(p,case,targets[cid],set(pos),official,independent);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=d['dataset'],arm=arm,metrics=v))
  del d;gc.collect()
 s,p=subset_aggregate(rows);vs,vp=subset_aggregate([r for r in rows if r['case_id'] in valid]);put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,per_domain=16,rows=rows,summary=s,paired=p,pre_adaptation_chart_valid_diagnostic=dict(count=len(valid),summary=vs,paired=vp),GT_bindings=bindings,GT_read=True,not_paper_method=True,development_not_confirmation=True,no_parameter_search=True));put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('OUTPUT_BUDGET32_POSTSEAL_COMPLETE')
if __name__=='__main__':{'root':root,'score':score}[sys.argv[1]]()
