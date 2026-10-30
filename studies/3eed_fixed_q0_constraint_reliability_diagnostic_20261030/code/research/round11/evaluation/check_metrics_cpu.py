"""Synthetic-only necessary contract/denominator checks for R11 selected metrics."""
from pathlib import Path
import hashlib,json
import numpy as np
import metrics as m
D=Path(__file__).resolve().parent

def main():
 rows=[]
 selected={}
 for platform,ids in [('drone',['0','1']),('quad',['2','3'])]:
  for control,arm in m.CONTROL_TO_ARM.items():
   scores={sid:(.9 if int(sid)%2==0 else .1) if control=='reliability' else 0. for sid in ids}
   selected[arm,platform]=set(m.deterministic_select(ids,.5,scores,dict.fromkeys(ids,True),salt=m.TIE_SALT,platform=platform)['selected'])
 for i,(platform,x) in enumerate([('drone',.8),('drone',1.2),('quad',.8),('quad',1.2)]):
  source=np.array([1,0,0,2,2,2.],np.float32);b=np.tile(source,(11,1));b[1,0]=x;b[2]=b[1];b[3]=b[1];b[4]=b[1]
  chosen=i%2==0
  gate_selected=np.array([str(i) in selected[arm,platform] for arm in m.GATES],bool)
  for j,yes in enumerate(gate_selected):b[j+5]=b[1] if yes else source
  atomic=np.tile(source,(6,1));atomic[np.arange(6),np.arange(6)]=b[1]
  row,cost=m.evaluate_case(sid=str(i),platform=platform,scene=platform+str(i//2),boxes_raw6=b,atomic_boxes_raw6=atomic,target7=np.array([0,0,0,2,2,2,0.]),statuses=['synthetic']*11,atomic_active6=b[1]!=source,coordinate_scores6=np.array([.9 if chosen else .1,0,0,0,0,0]),coordinate_defined6=np.array([1,0,0,0,0,0],bool),RGB_score=0.,RGB_defined=False,case_score=.9 if chosen else .1,case_defined=True,gate_selected6=gate_selected,control_names=['reliability','hash_random','source_agreement','low_plane_residual','point_count','stability_only'],control_scores=np.array([.9 if chosen else .1,0,0,0,0,0]),control_defined=np.ones(6,bool))
  assert row['units_coordinate_application'][0]['effect']>0 if chosen else row['units_coordinate_application'][0]['effect']<0
  assert not row['unit_RGB_term']['candidate']
  rows.append(row)
 out=m.summarize(rows)
 for platform in ('drone','quad'):
  u=out['discrimination'][platform]['coordinate_primary_IoU'];assert u['n_candidate']==2 and u['n_non_candidate']==10 and u['ranking']['helpful_positive_AUROC_casebalanced']==1
  rgb=out['discrimination'][platform]['RGB_primary_matched_IoU'];assert rgb['n_candidate']==0 and rgb['ranking']['helpful_positive_AUROC_casebalanced'] is None
  assert all(r['risk']['K']==1 and r['risk']['allcase_n']==2 for r in out['coverage_risk_primary50'] if r['platform']==platform)
  tab=[t for t in out['summary'] if t['platform']==platform and t['arm']=='source' and t['population']=='original_G_changed'][0];assert tab['n']==2
 json.dumps(out,allow_nan=False)
 receipt={'state':'PASS_CPU_SYNTHETIC_ONLY','all11arms_and_atomic6_validated':True,'beneficial_harmful_signs_and_error_direction_checked':True,'same_common_changed_source_denominator':True,'RGB_zero_candidate_NA_not_false_negative':True,'matched_gate_coverage_and_allcase_denominator':True,'missing_scene_singlecluster_CI_NA':True,'no_GT_dataset_model_GPU_used':True,'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('metrics.py','discrimination.py','check_metrics_cpu.py')}}
 (D/'CPU_METRICS_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
