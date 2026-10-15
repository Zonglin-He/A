"""Independent CPU geometry, reference binding, rollback and final-output readback."""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_weightfree_coordinate_v1/confirmation'
def main():
 import torch,numpy as np
 from scripts.review_stvg_instance_step_diagnostic_v1 import iou_cxcywh
 from scripts import stvg_target_teacher_student_v1 as core
 seal=read(BASE/'STUDENT_GLOBAL_BARRIER.json');assert seal['status']=='complete'and seal['count']==32 and not seal['GT_read'];checks=0;steps=0;err=0.;fitcount=0;refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']}
 selection=read(BASE.parent/'SPATIAL_SELECTION_LOCK.json')
 from scripts.run_stvg_coordinate_confirmation_v1 import roster,load_case
 from scripts.review_stvg_coordinate_temporal_v1 import verify as temporal_verify
 cohort={r['probe_alias']:r for r in roster()};temporal_config=read(BASE.parent/'TEMPORAL_SELECTION_LOCK.json')['configs'];temporal_checks=0;temporal_error=0.
 teacher=read(BASE/'TEACHER_GLOBAL_BARRIER.json')
 assert teacher['input_barrier_sha256']==sha(BASE/'CANDIDATE_INPUT_BARRIER.json') and teacher['job_lock_sha256']==sha(BASE/'TEACHER_JOB_LOCK.json')
 assert selection['confirmation_used_for_selection'] is False
 for item in seal['files']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert d['reference_record']==refs[cid]and not d['GT_read']and d['source_restored'];checks+=3
  case,_=load_case(cohort[cid]);temp=torch.load(BASE/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False)['temporal'];tf=temp['selected_temporal_fit']
  if 'temporal_fit'in temp:
   oldfit=temp['temporal_fit'];assert oldfit['final_interval']==temp['final_interval']
   n,e=temporal_verify(case,oldfit);temporal_checks+=n;temporal_error=max(temporal_error,e)
  else:assert temp['final_interval']==case['Native_interval']
  assert set(d['arms'])=={'F','T','original','selected','T_original','T_selected','T_tuned','T_tuned_selected'}
  assert d['arms']['F']['interval']==d['arms']['original']['interval']==d['arms']['selected']['interval']==case['Native_interval']
  assert d['arms']['T']['interval']==d['arms']['T_original']['interval']==d['arms']['T_selected']['interval']==temp['final_interval']
  for plain,joint in [('F','T'),('original','T_original'),('selected','T_selected')]:assert np.array_equal(d['arms'][plain]['boxes'],d['arms'][joint]['boxes'])
  if tf is not None:
   assert tf['config']==temporal_config[d['dataset']] and tf['final_interval']==temp['selected_output_interval']
   n,e=temporal_verify(case,tf);temporal_checks+=n;temporal_error=max(temporal_error,e)
  else:assert temp['selected_output_interval']==case['Native_interval']
  assert d['arms']['T_tuned']['interval']==d['arms']['T_tuned_selected']['interval']==temp['selected_output_interval']
  assert np.array_equal(d['arms']['T_tuned']['boxes'],d['arms']['F']['boxes']) and np.array_equal(d['arms']['T_tuned_selected']['boxes'],d['arms']['selected']['boxes'])
  for variant in ['original','selected']:
   aa=refs[cid][variant]['anchors'];fit=d['fits'].get(variant)
   if not aa:
    assert np.array_equal(d['arms'][variant]['boxes'],d['arms']['F']['boxes']) and d['arms'][variant]['interval']==d['arms']['F']['interval'];assert np.array_equal(d['arms']['T_'+variant]['boxes'],d['arms']['T']['boxes']) and d['arms']['T_'+variant]['interval']==d['arms']['T']['interval'];checks+=2;continue
   expected_cfg=read(ROOT/'artifacts/stvg_weightfree_tuning_v1/METHOD_SELECTION_LOCK.json')['configs'][d['dataset']]
   if variant=='selected':expected_cfg=selection['configs'][d['dataset']]
   assert fit['config']==expected_cfg and fit['declared_steps']==expected_cfg['steps'] and d['selection_sha256']==sha(BASE.parent/'SPATIAL_SELECTION_LOCK.json')
   fitcount+=1;assert torch.equal(d['readouts'][variant]['boxes'].cpu(),fit['final_cached_decoder_boxes']);checks+=1
   pos=[a['position']for a in aa];e=np.asarray([a['box_cxcywh']for a in aa]);w=np.asarray([a['weight']for a in aa]);stopped=False
   for s in fit['path']:
    if s.get('skipped'):continue
    assert not stopped
    q0=iou_cxcywh(s['boxes'].numpy()[pos],e);q1=iou_cxcywh(s['proposed_boxes'].numpy()[pos],e)
    err=max(err,float(np.max(abs(q0-s['reference_iou_before'].numpy()))),float(np.max(abs(q1-s['reference_iou_proposed'].numpy()))));assert err<3e-6
    if s['accepted']:
     assert float(w@(q1-q0)/w.sum())>=-3e-6
     ww=torch.tensor(w,dtype=s['reference_iou_before'].dtype);assert float((ww*(s['reference_iou_proposed']-s['reference_iou_before'])).sum()/ww.sum())>=-1e-7
    if s['proposed']and not s['accepted']:
     assert core._tree_equal(s['optimizer_after'],s['optimizer_before'])and core._tree_equal(s['update_state'],s['state']);stopped=True
    steps+=1;checks+=3
   xy=np.asarray(d['arms'][variant]['boxes']);final=fit['final_cached_decoder_boxes'].numpy();ind=np.concatenate([final[:,:2]-final[:,2:]/2,final[:,:2]+final[:,2:]/2],axis=1);assert np.allclose(ind,xy,atol=1e-7,rtol=0);checks+=1
 write(BASE/'ROOT_READBACK.json',dict(status='pass',case_count=32,fit_count=fitcount,step_count=steps,check_count=checks,max_iou_abs_error=err,temporal_independent_checks=temporal_checks,temporal_max_theta_abs_error=temporal_error,GT_read=False,CPU_only=True,scope='independent geometry, saved Adam/parameter rollback, full-vs-cached final decoder equality; not independent backward replay'))
 files=list(seal['files'])+list(read(BASE/'TEMPORAL_GLOBAL_BARRIER.json')['files'])
 files.append(dict(path=str((BASE.parent/'SPATIAL_SELECTION_LOCK.json').relative_to(ROOT)),sha256=sha(BASE.parent/'SPATIAL_SELECTION_LOCK.json')))
 assert read(BASE/'ROOT_INPUT_RENDER_READBACK.json')['status']=='pass'
 for p in [BASE/'REFERENCE_LOCK.json',BASE/'TEACHER_GLOBAL_BARRIER.json',BASE/'ROOT_READBACK.json',BASE/'ROOT_INPUT_RENDER_READBACK.json',BASE/'CANDIDATE_INPUT_BARRIER.json',BASE/'TEACHER_JOB_LOCK.json',BASE.parent/'TEMPORAL_SELECTION_LOCK.json',BASE.parent/'JOINT_METHOD_LOCK.json',BASE.parent/'NO_BOOTSTRAP_AUTHORITY004.json']:files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 assert read(BASE/'boundary_adapter005/ROOT_REVISION_READBACK.json')['status']=='pass'
 for p in [BASE/'boundary_adapter005/ROOT_REVISION_READBACK.json',BASE/'boundary_adapter005/REVISION_RUNTIME.json',BASE/'boundary_adapter005/BOUNDARY_READBACK.json',BASE/'boundary_adapter005/execution006/REVISION_RUNTIME.json']:files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,GT_read=False,files=files,scorer_sha256=sha(ROOT/'scripts/score_stvg_coordinate_confirmation_v1.py')))
 print('root pass',fitcount,steps,checks,err)
if __name__=='__main__':main()
