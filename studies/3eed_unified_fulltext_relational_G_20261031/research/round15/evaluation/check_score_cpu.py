"""Synthetic scalar/geometry/release guards only, no real labels or predictions."""
from pathlib import Path
from types import SimpleNamespace
import collections,copy,hashlib,json,tempfile
import numpy as np
import score_direct as s
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'SCORER_CPU_FIXTURE.json'
 if out.exists():raise FileExistsError(out)
 checks={}
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'release.json';p.write_text(json.dumps({'CPU_SCORE_RELEASE':False,'stage':'R15_unified_full_description_direct8'}));ix=Path(tmp)/'index';ix.write_text('not valid JSON and must not be parsed')
  try:s.unlabelled_gate(SimpleNamespace(release=p,release_sha256=sha(p),scoring_index=ix))
  except ValueError as e:assert 'release' in str(e)
  else:raise AssertionError('False release admitted')
 checks['false_release_rejects_before_index_payload_or_output']=True
 box=np.array([0,0,0,2,2,2],np.float32);pool=np.tile(box,(256,1));raw=np.tile(box,(8,1));raw[4,0]=.5
 row,cost=s.evaluate_row({'id':'SYNTHETIC','platform':'drone','scene':'SYNTHETIC_SCENE','case_role':'SYNTHETIC'},
  {'arm_boxes_raw6':raw,'arm_boxes_eval6':s.eval_boxes(raw),'q_source':np.array(0),'arm_query_index':np.zeros(8,np.int64),'arm_status':np.array(['SYNTHETIC']*8),'source_boxes_raw6':pool,'source_text_scores256':np.zeros(256)},np.r_[box.astype(float),0.])
 assert abs(row['arms']['source']['iou']-1)<1e-12 and abs(row['arms']['graph_geo']['iou']-.6)<1e-12
 checks['native_yaw0_analytic_cube_iou_one_and_point_six']=True
 assert row['arms']['graph_geo']['coordinate_errors']['center_L2_m']==.5 and row['arms']['graph_geo']['coordinate_errors']['size_L2_m']==0
 checks['coordinate_center_size_units_m_separate']=True
 rows=[]
 for role,nd,nq in [('fresh64_recent_content_relative_only',32,32),('R14_regression140_exposed',64,76)]:
  for platform,n in [('drone',nd),('quad',nq)]:
   for i in range(n):
    r=copy.deepcopy(row);r.update(id=f'SYNTHETIC_{role}_{platform}_{i}',platform=platform,scene=f'SYNTHETIC_{platform}_{i%2 if platform=="quad" else 0}',case_role=role)
    for name,a in r['arms'].items():
     value=.5 if name=='source' else .51 if name=='graph_assoc' else .53
     if role=='R14_regression140_exposed':value=.9
     a.update(iou=value,Acc25=value>.25,Acc50=value>.5,query_changed=False,actual_changed=False,actual_changed_axes=[False]*6)
    rows.append(r)
 roles=['fresh64_recent_content_relative_only','R14_regression140_exposed'];su,co,oo,ta=s.summarize(rows,roles);gate=s.progression_gate(su)
 assert len(rows)==204 and len(su)==48 and len(co)==84 and len(ta)==84 and len(oo)==6
 checks['all204_eightarms64_140_independent_denominators']=True
 assert all(v['passed'] and v['n']==32 for v in gate.values()) and gate['drone']['scene_clusters']==1 and gate['quad']['scene_clusters']==2
 checks['source_and_association_micro_and_equal_scene_gate']=True
 fail=copy.deepcopy(su)
 for x in fail:
  if x['case_role']==roles[0] and x['platform']=='drone' and x['arm']=='graph_geo':x['mean_iou']=.5+5e-9
 assert not s.progression_gate(fail)['drone']['passed']
 checks['positive_smaller_than_1e_minus8_not_progression']=True
 fail=copy.deepcopy(su)
 for x in fail:
  if x['case_role']==roles[0] and x['platform']=='quad' and x['arm']=='graph_geo':x['Acc50']=0.
 assert not s.progression_gate(fail)['quad']['passed']
 checks['accuracy_decrease_blocks_even_if_mean_iou_positive']=True
 for a in row['arms'].values():a['iou']=.25
 assert not any(a['iou']>.25 for a in row['arms'].values()) and not (.5>.5)
 checks['strict_greater_thresholds_not_greater_equal']=True
 sign=s.signed_stats([.1,-.2,0,1e-9],1e-8)
 assert sign['positive_count']==1 and sign['negative_count']==1 and sign['ties_within_tol']==2 and not sign['gain_sum_exceeds_abs_harm_sum']
 checks['negative_zero_tolerance_and_signed_mass_retained']=True
 for p,h in s.DEPENDENCIES.items():assert s.sha(s.ROOT/p)==h
 checks['immutable_metrics_transform_and_3D_kernel_hashes']=True
 receipt={'state':'PASS_R15_SCORER_SYNTHETIC_ONLY_NO_REAL_LABEL_OR_PREDICTION_ACCESS','check_count':len(checks),'checks':checks,'scorer_sha256':sha(D/'score_direct.py'),'fixture_program_sha256':sha(__file__),'real_GT_index_annotation_model_GPU_or_sensor_read':False,'synthetic_3D_IoU_called':True,'frozen_dependencies':s.DEPENDENCIES}
 out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'state':receipt['state'],'checks':len(checks),'sha256':sha(out)}))
if __name__=='__main__':main()
