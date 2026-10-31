"""Pure synthetic export/NA/privacy tests, never builds an actual bundle."""
from pathlib import Path
import copy,gzip,hashlib,importlib.util,json,sys,tempfile
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'research/round15/evaluation')]
import build_export as b
from export_projection import *
import score_direct as s

def main():
 out=HERE/'EXPORT_CPU_FIXTURE.json'
 if out.exists():raise FileExistsError(out)
 checks={};rows=[];cases=[]
 for role,prefix,nd,nq in [(ROLES[0],'fresh',32,32),(ROLES[1],'reg',64,76)]:
  for platform,n in [('drone',nd),('quad',nq)]:
   for i in range(n):
    sid=f'SYNTHETIC_PRIVATE_ID_{role}_{platform}_{i}';scene='SYNTHETIC_PRIVATE_SCENE_'+platform+str(i%2)
    case={'id':sid,'case_alias':f'{prefix}_{platform}_{i+1:05d}','role':role,'platform':platform,'scene_PRIVATE':scene};cases.append(case)
    arms={}
    for j,name in enumerate(ARMS):
     value=.5 if j==0 else .5+(.01 if i%3==0 else -.01 if i%3==1 else 0.)
     arms[name]={'iou':value,'Acc25':value>.25,'Acc50':value>.5,'coordinate_errors':{'center_L2_m':1.,'size_L2_m':2.,'abs_error6_m':[1.,0.,0.,0.,2.,0.],'size_parameter_not_global_yaw0_enclosure':True},'query_index':255,'query_changed':j>0,'actual_changed':j>0,'actual_changed_axes':[j>0,False,False,False,False,False],'raw_nonpositive_sizes':0,'status':'SYNTHETIC_source_fallback' if i%2 else 'SYNTHETIC_changed','center_shift_L2_m':float(j>0),'size_shift_L2_m':0.}
    oracle={'max_iou_all256':.9,'all256_Acc25':True,'all256_Acc50':True,'rank_recoverable_25':False,'rank_recoverable_50':True,'first_threshold_candidate_source_score_rank_interval_25':[1,1],'first_threshold_candidate_source_score_rank_interval_50':[1,1],**{'score_retrieval_'+str(k):{'best_iou_GTassisted':.9,'Acc25':True,'Acc50':True} for k in (1,5,10,256)}}
    rows.append({'id':sid,'platform':platform,'scene':scene,'case_role':role,'q_source':255,'arms':arms,'offline_source_pool_oracle':oracle})
 su,co,oo,ta=s.summarize(rows,ROLES)
 result={'state':'R15_eight_direct_arms_scored_only_after_complete_seal','n':204,'case_role_counts':{ROLES[0]:64,ROLES[1]:140},'rows_PRIVATE':rows,'summary':su,'paired_contrasts':co,'offline_source_pool_oracle':oo,'signed_tails_PRIVATE':ta,'seconds':0.,'source_checkpoint_sha256':'a'*64}
 aliases={(r['platform'],r['scene']):'SYNTHETIC_PUBLIC_'+r['platform']+str(i) for i,r in enumerate(rows)}
 projection,summary,tails=project_result(result,cases,aliases)
 assert len(projection)==1632 and len(summary['summary'])==48 and len(summary['paired_contrasts'])==84
 checks['complete204x8_roles64_140_not_mixed']=True
 text=json.dumps([projection,summary,tails]);assert 'SYNTHETIC_PRIVATE_ID' not in text and 'SYNTHETIC_PRIVATE_SCENE' not in text and 'query_index' not in text and 'q_source' not in text
 checks['ID_scene_query_indices_not_in_scalar_projection']=True
 for r in projection:
  orig=next(x for x in rows if x['id']==next(c['id'] for c in cases if c['case_alias']==r['case_alias']))['arms'][r['arm']]
  assert r['iou']==orig['iou'] and r['center_L2_error_m']==orig['coordinate_errors']['center_L2_m'] and r['status']==orig['status'] and r['actual_changed_axis_bitmask']==int(orig['actual_changed_axes'][0])
 checks['every_projected_synthetic_IoU_error_status_and_bitmask_exact']=True
 assert any(r['actual_changed'] is False and r['raw_nonpositive_sizes']==0 for r in projection)
 assert json.loads(json.dumps({'defined_zero':0.,'undefined':None,'defined':False}))=={'defined_zero':0.,'undefined':None,'defined':False}
 checks['zero_false_null_not_conflated']=True
 for key in ('A_selected_mask_index','qUeRy--indices','nested_query_index','pointpath','raw_answer_PRIVATE'):
  try:ensure_public({key:0})
  except ValueError:pass
  else:raise AssertionError('Forbidden payload key admitted '+key)
 checks['normalized_index_and_payload_keys_rejected']=True
 try:contrast({**co[0],'raw_boxes':[[1,2,3,4,5,6]]})
 except ValueError:pass
 else:raise AssertionError('Unregistered schema admitted')
 checks['future_raw_aggregate_fields_not_silently_copied']=True
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp);p=d/'readiness.json';p.write_text(json.dumps({'schema':'R15_finite_export_readiness_v1','build_authorized':False}))
  try:b.main(SimpleNamespace(readiness=p,readiness_sha256=b.sha(p),output=d/'not_created'))
  except ValueError:pass
  else:raise AssertionError('Preparation readiness built bundle')
  assert not (d/'not_created').exists();checks['unready_build_rejects_before_any_actual_result_read_or_output']=True
  for ext in ('.json','.md','.py'):
   p=d/('synthetic'+ext);p.write_text('x = "SYNTHETIC_FORBIDDEN_CAPTION"' if ext=='.py' else json.dumps({'state':'SYNTHETIC_FORBIDDEN_CAPTION'}) if ext=='.json' else 'SYNTHETIC_FORBIDDEN_CAPTION')
   try:b.privacy_scan(p,['SYNTHETIC_FORBIDDEN_CAPTION'])
   except ValueError:pass
   else:raise AssertionError('Known private marker admitted')
  checks['known_registry_marker_scan_checks_all_text_file_classes']=True
  prefix='/'+'home'+'/'
  p=d/'generic.py';p.write_text('prefix='+repr(prefix)+'\nfixture='+repr(prefix+'private/model')+'\n');b.privacy_scan(p,[])
  p.write_text('actual='+repr(prefix+'real_person/actual/data')+'\n')
  try:b.privacy_scan(p,[])
  except ValueError:pass
  else:raise AssertionError('Actual code hostpath admitted')
  checks['only_two_exact_generic_code_literals_exempt_not_real_hostpath']=True
 receipt={'state':'PASS_R15_FINITE_EXPORT_PREPARATION_SYNTHETIC_ONLY_NOT_ACTUAL_EXPORT_AUDIT','checks':checks,'check_count':len(checks),'n_synthetic_case_arm_rows':len(projection),'source_sha256':{p.name:b.sha(p) for p in [HERE/'build_export.py',HERE/'export_projection.py',Path(__file__)]},'actual_scored_results_GT_privatecaptions_media_rawsensor_model_GPU_read_or_actual_bundle_built':False,'actual_export_requires_separate_independent_review':True}
 out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'state':receipt['state'],'checks':len(checks),'sha256':b.sha(out)}))
if __name__=='__main__':main()
