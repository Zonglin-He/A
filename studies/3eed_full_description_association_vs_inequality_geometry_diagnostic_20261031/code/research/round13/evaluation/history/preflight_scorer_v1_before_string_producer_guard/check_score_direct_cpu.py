"""Synthetic integrity/geometry fixtures only; never actual index/GT/data IO."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,tempfile,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_direct as score


def check():
 checks={};detail={}
 with tempfile.TemporaryDirectory(prefix='scorer_synthetic_',dir=Path(__file__).resolve().parent) as td:
  base=Path(td);src=base/'source';pred=base/'pred';helper=base/'helper'
  (src/'source_cache').mkdir(parents=True);pred.mkdir();helper.mkdir()
  checkpoint=base/'SYNTHETIC_NOT_MODEL.bin';checkpoint.write_bytes(b'synthetic integrity certificate not model weights')
  csha=score.sha(checkpoint)
  manifest=[];cases=[]
  roles=[('synthetic_new','new_content','new_content108','drone'),
         ('synthetic_old','exposed_R12_all_changed_regression','regression8','quad')]
  data_by_id={}
  for sid,role,case_role,platform in roles:
   pool=np.tile(np.array([20,0,0,2,2,2],np.float32),(256,1))
   pool[0]=[0,0,0,2,2,2];pool[1]=[.1,0,0,2,2,2]
   scores=np.linspace(1,0,256,dtype=np.float32);q=0
   raw=np.tile(pool[q],(9,1));armq=np.zeros(9,np.int64)
   raw[1]=pool[1];armq[1]=1
   raw[5,0]=np.float32(.1);raw[6,0]=np.float32(.05)
   raw[7]=raw[0]+np.float32(.5)*(raw[5]-raw[0]);raw[8,0]=np.float32(-.1)
   d={'sample_id':np.asarray(sid),'arm_names':np.asarray(score.ARMS),'arm_query_index':armq,
      'arm_boxes_raw6':raw,'arm_boxes_eval6':score.eval_boxes(raw),'arm_status':np.array(['synthetic']*9),
      'source_boxes_raw6':pool,'source_text_scores256':scores,'q_source':np.asarray(q),
      'case_role':np.asarray(case_role)}
   np.savez_compressed(pred/(sid+'.npz'),**d);data_by_id[sid]=d
   np.savez_compressed(src/'source_cache'/(sid+'.npz'),sample_id=np.asarray(sid),
       boxes_raw6=pool[None],text_scores=scores[None],query_index=np.asarray(q))
   manifest.append({'id':sid,'platform':platform,'scene':'synthetic_scene_'+platform,
       'pointpath':'unused','metapath':'unused','imagepath':'unused','caption':'not read by scoring','suffix':None,'seed':0})
   cases.append({'id':sid,'platform':platform,'scene_PRIVATE':'synthetic_scene_'+platform,
       'role':role,'source_dir':str(src)})
  score.dump(base/'manifest.json',manifest);score.dump(base/'cases.json',cases)
  # If unlabelled_gate attempts to parse label payload it must fail. Hashing its
  # bytes is allowed before label release and is not parsing an annotation.
  (base/'INDEX_SYNTHETIC.json').write_text('NOT JSON: unlabelled gate must never parse this payload')
  (helper/'receipt.json').write_text('{"synthetic_only":true}\n')
  prompt=base/'prompt.json';prompt.write_text('{"synthetic_prompt":true}\n')
  score.dump(src/'SEALED.json',{'files':{str(p.relative_to(src)):score.sha(p) for p in (src/'source_cache').glob('*.npz')},
      'source_checkpoint_sha256':csha,'source_unchanged':True,'source_repeat_bitexact':True})
  score.dump(helper/'SEALED.json',{'files':{'receipt.json':score.sha(helper/'receipt.json')}})
  score.dump(pred/'SEALED.json',{'files':{p.name:score.sha(p) for p in pred.glob('*.npz')},
      'n_samples':2,'n_arms':9,'arm_names':list(score.ARMS),'source_checkpoint_sha256':csha,
      'manifest_sha256':score.sha(base/'manifest.json')})
  release={'CPU_SCORE_RELEASE':True,'stage':'R13_association_vs_inequality_geometry_direct',
    'all_A_B_predictions_and_helper_outputs_sealed_before_GT':True,
    'prediction_seal_sha256':score.sha(pred/'SEALED.json'),'manifest_sha256':score.sha(base/'manifest.json'),
    'cases_sha256':score.sha(base/'cases.json'),'scoring_index_sha256':score.sha(base/'INDEX_SYNTHETIC.json'),
    'scorer_sha256':score.sha(Path(score.__file__)),'source_checkpoint_path':str(checkpoint),
    'source_checkpoint_sha256':csha,'geometry_dependency_sha256':score.DEPENDENCIES,
    'program_bindings':{str(Path(score.__file__)):score.sha(Path(score.__file__))},
    'input_bindings':{str(base/'manifest.json'):score.sha(base/'manifest.json')},
    'prompt_bindings':{str(prompt):score.sha(prompt)},
    'expected_case_role_counts':{'new_content108':1,'regression8':1},
    'required_seal_roles':['source','composite','helper'],
    'sealed_artifacts':[{'role':name,'directory':str(directory),'seal_sha256':score.sha(directory/'SEALED.json')}
       for name,directory in [('source',src),('composite',pred),('helper',helper)]]}
  score.dump(base/'release.json',release)
  def args(out='out'):
   return SimpleNamespace(predictions=pred,manifest=base/'manifest.json',cases=base/'cases.json',
       scoring_index=base/'INDEX_SYNTHETIC.json',release=base/'release.json',
       release_sha256=score.sha(base/'release.json'),output=base/out)
  rr,mm,dd=score.unlabelled_gate(args())
  assert (base/'out/SCORER_LOCK.json').exists() and len(mm)==2
  assert [r['case_role'] for r in mm]==['new_content108','regression8']
  checks['all_hash_contracts_before_index_payload_parse']=True
  checks['ninekey_manifest_plus_separate_role_cases_bridge']=True
  checks['original_source_cache_pool_scores_q_bitexact']=True
  checks['cohort_count_dynamic_not_old146']=True
  (helper/'receipt.json').write_text('corrupt after seal')
  try:score.unlabelled_gate(args('must_not_exist'))
  except ValueError:checks['helper_corruption_fails_before_GT_lock']=True
  else:raise AssertionError('Corrupt helper accepted')
  assert not (base/'must_not_exist').exists()
  (helper/'receipt.json').write_text('{"synthetic_only":true}\n')
  # Direct geometry scoring uses supplied synthetic arrays only.
  synthetic_target=np.array([0,0,0,2,2,2,0],np.float32)
  evaluated=[]
  for row in mm:
   out,cost=score.evaluate_row(row,dd[row['id']],synthetic_target)
   evaluated.append(out)
  ss,cc,oo,tt=score.summarize(evaluated,list(release['expected_case_role_counts']))
  assert len(ss)==2*2*9 and len(oo)==4
  assert all(x['n']==1 for x in ss)
  checks['new_and_regression_cohorts_never_pooled']=True
  assert evaluated[0]['arms']['source']['iou']==1.
  assert evaluated[0]['arms']['A_full_consensus']['iou']<1.
  assert evaluated[0]['arms']['B_inequality_half']['coordinate_errors']['abs_error6_m'][0]<evaluated[0]['arms']['B_inequality_full']['coordinate_errors']['abs_error6_m'][0]
  checks['selected_yaw0_IoU_center_size_abs6_and_signed_tails']=True
  assert evaluated[0]['offline_source_pool_oracle']['max_iou_all256']==1.
  assert 'not method' in evaluated[0]['offline_source_pool_oracle']['scope']
  checks['all256_oracle_label_assisted_separate_from_methods']=True
  # Exact strict threshold counterexample at 0.25 / 0.5.
  pool=np.tile([100,0,0,2,2,2],(256,1)).astype(np.float32)
  pool[0]=[0,0,0,1,1,2];pool[1]=[0,0,0,1,2,2];pool[6]=synthetic_target[:6]
  sc=np.linspace(1,0,256,dtype=np.float32)
  oracle,_=score.pool_oracle(pool,sc,synthetic_target,0)
  assert oracle['source_q0_iou']==.25 and oracle['rank_recoverable_25']
  assert oracle['score_retrieval_5']['Acc25'] and not oracle['score_retrieval_5']['Acc50']
  assert oracle['score_retrieval_10']['Acc50']
  checks['strict_IoU25_50_and_retrieval_upper_bounds']=True
  tied=np.ones(256,np.float32);tie,_=score.pool_oracle(pool,tied,synthetic_target,0)
  assert tie['first_threshold_candidate_source_score_rank_interval_50']==[1,256]
  assert not tie['score_retrieval_5']['native_GPU_rank_claimed']
  checks['oracle_source_rank_tie_interval_not_native_GPU_claim']=True
  detail={'synthetic_summary_count':len(ss),'synthetic_contrast_count':len(cc),
      'synthetic_oracle_summary_count':len(oo),'synthetic_tail_group_count':len(tt)}
 return {'state':'passed_R13_synthetic_scoring_and_beforeGT_integrity_checks',
    'checks':checks,'details':detail,'real_GT_index_payload_sensor_model_GPU_access':False,
    'program_sha256':{'research/round13/evaluation/score_direct.py':score.sha(Path(score.__file__)),
      'research/round13/evaluation/check_score_direct_cpu.py':score.sha(Path(__file__))},
    'geometry_dependency_sha256':score.DEPENDENCIES}


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=check()
 if a.output.exists():raise FileExistsError(a.output)
 score.dump(a.output,result)
 print(json.dumps({'state':result['state'],'n_checks':len(result['checks'])}))
