"""Synthetic scorer/release checks; no actual GT/taskIoU/source-model read."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,tempfile,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_transfer as s


def fake_case(pid,scene,signal,shift):
 source=np.array([shift,0,0,2,2,2],np.float32);raw=np.tile(source,(7,1))
 if signal:
  raw[1,0]-=.1;raw[2,0]-=.05;raw[3,0]-=.08;raw[4]=raw[3]
  raw[5,0]-=.16;raw[6]=raw[5];raw[6,0]+=np.float32(1e-6)
 d={'arm_boxes_raw6':raw,'arm_boxes_eval6':raw.copy(),'q_source':np.array(0),
    'A_nonzero_residual':np.array(signal),'A_gate':np.array(signal),
    'A_frozen_G_status':np.array('synthetic_G_or_zero'),'arm_status':np.array(['synthetic']*7)}
 pair={'pair_id':pid,'A_id':'A_'+pid,'B_id':'B_'+pid}
 row={'scene':scene};target=np.array([0,0,0,2,2,2,0],np.float32)
 return s.evaluate_case(pair,row,d,target)[0]


def check():
 checks={}
 rows=[fake_case('synthetic_1','scene1',True,.5),fake_case('synthetic_2','scene1',False,2.),fake_case('synthetic_3','scene2',False,0.)]
 summary,contrasts,tails,numerical=s.summarize(rows)
 assert len(summary)==14
 assert {x['n'] for x in summary if x['population']=='all76_primary'}=={3}
 assert {x['n'] for x in summary if x['population']=='A_nonzero_signal_descriptive'}=={1}
 checks['all_cases_and_A_signal_descriptive_denominators_separate']=True
 source=next(x for x in summary if x['population']=='all76_primary' and x['arm']=='source')
 assert source['mean_iou']!=source['equal_scene_mean_iou']
 checks['unequal_scene_micro_and_equal_scene_both_reported']=True
 assert all('coordinate_errors' in a and len(a['coordinate_errors']['abs_error6_m'])==6 for row in rows for a in row['arms'].values())
 checks['selected_raw_center_size_abs6_errors']=True
 assert len(contrasts)>0 and all(len(t['largest_negative_PRIVATE'])==3 if t['population']=='all76_primary' else len(t['largest_negative_PRIVATE'])==1 for t in tails)
 checks['signed_positive_negative_zero_tails_not_filtered']=True
 assert numerical['half']['coordinate_Linf_max_m']==0 and 0<numerical['full']['coordinate_Linf_max_m']<2e-4
 assert numerical['full']['mathematical_equivalence_not_a_contribution_gate']
 checks['parameter_kernel_numerical_and_task_metrics_separate']=True
 # Exact thresholds, no comparison tolerance secretly changes them.
 target=np.array([0,0,0,2,2,2,0.]);boxes=np.array([[0,0,0,1,1,2],[0,0,0,1,2,2.]])
 v,_=s.candidate_ious(boxes,target)
 assert v.tolist()==[.25,.5] and not bool(v[0]>.25) and not bool(v[1]>.5)
 checks['strict_native_yaw0_IoU25_50']=True
 drone={'state':'sealed_R14_Drone64_seven_direct_arms_before_GT','n_cases':64,
   'arm_names':list(s.DRONE_ARMS),'no_parameter_update_GT_or_newbox':True,'model_state_unchanged':True}
 s.validate_drone_final(drone,64)
 for bad in [dict(drone,n_cases=2),dict(drone,state='sealed_R14_Drone_first2_engineering_no_GT'),
             dict(drone,arm_names=['source']),dict(drone,no_parameter_update_GT_or_newbox=False),dict(drone,model_state_unchanged=False)]:
  try:s.validate_drone_final(bad,64)
  except ValueError:pass
  else:raise AssertionError('Incomplete/nonfrozen/wrongarms Drone seal accepted')
 checks['actual_Drone_n_cases64_exact_state_sevenarms_and_frozen_flags']=True
 with tempfile.TemporaryDirectory(prefix='scorer_release_synthetic_',dir=Path(__file__).resolve().parent) as temp:
  tmp=Path(temp);release=tmp/'release.json';poison=tmp/'INDEX_POISON.json';poison.write_text('NOT JSON: must not parse before branch gate')
  s.dump(release,{'CPU_SCORE_RELEASE':True,'both_branches_complete_and_sealed':False,'stage':'R14_both_branches_complete_before_target_GT'})
  a=SimpleNamespace(release=release,release_sha256=s.sha(release),output=tmp/'out')
  try:s.prelabel_gate(a)
  except ValueError as err:assert 'BOTH-branches' in str(err)
  else:raise AssertionError('Single-branch release accepted')
  assert not a.output.exists();checks['both_branch_boolean_gate_before_any_label_index_payload']=True
  pred=tmp/'quad';pred.mkdir();s.dump(pred/'SEALED.json',{})
  pair=tmp/'pairs.json';pair.write_text('[]');bm=tmp/'Bmanifest.json';bm.write_text('[]')
  r={'CPU_SCORE_RELEASE':True,'both_branches_complete_and_sealed':True,'stage':'R14_both_branches_complete_before_target_GT',
    'quad_prediction_seal_sha256':s.sha(pred/'SEALED.json'),'pairs_sha256':s.sha(pair),'B_manifest_sha256':s.sha(bm),
    'scoring_index_sha256':s.sha(poison),'quad_scorer_sha256':s.sha(Path(s.__file__)),
    'drone_prediction_dir':str(tmp/'missing_Drone'),'drone_prediction_seal_sha256':'0'*64}
  s.dump(release,r)
  a=SimpleNamespace(release=release,release_sha256=s.sha(release),predictions=pred,pairs=pair,B_manifest=bm,scoring_index=poison,output=tmp/'out2')
  try:s.prelabel_gate(a)
  except FileNotFoundError:pass
  else:raise AssertionError('Unsealed missing Drone branch accepted')
  assert not a.output.exists();checks['specified_Drone_seal_required_before_GT_index_parse']=True
  artifact=tmp/'mask';artifact.mkdir();(artifact/'receipt.txt').write_text('synthetic')
  producer=s.sha(Path(s.__file__))
  s.dump(artifact/'SEALED.json',{'files':{'receipt.txt':s.sha(artifact/'receipt.txt')},'program_sha256':producer})
  item={'directory':str(artifact),'seal_sha256':s.sha(artifact/'SEALED.json')}
  s.verify_artifact(item,{'synthetic_producer.py':producer})
  try:s.verify_artifact(item,{'different_producer.py':'f'*64})
  except ValueError:pass
  else:raise AssertionError('Unbound string producer accepted')
  checks['generic_two_key_artifact_schema_and_explicit_string_producer']=True
 return {'state':'passed_R14_Quad_synthetic_scoring_and_both_branch_gate_checks',
    'checks':checks,'actual_target_GT_taskIoU_sensor_model_GPU_accessed':False,
    'program_sha256':{str(p.relative_to(s.ROOT)):s.sha(p) for p in [Path(s.__file__),Path(__file__)]},
    'geometry_dependency_sha256':s.DEPENDENCIES}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check()
 if a.output.exists():raise FileExistsError(a.output)
 s.dump(a.output,r);print(json.dumps({'state':r['state'],'n_checks':len(r['checks'])}))
