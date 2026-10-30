"""Synthetic scalars only: independent auditor cross-check, no dataset/GT IO."""
from pathlib import Path
from types import SimpleNamespace
import copy,tempfile,json,hashlib
import metrics as m
import audit_scored as auditor
D=Path(__file__).resolve().parent

def main():
 rows=[]
 for platform,n in [('drone',128),('quad',192)]:
  for i in range(n):
   sid=platform+str(i);active=i<4;effect=(.01 if i%2==0 else -.01) if active else 0.;score=.9 if i%2==0 else .1
   arms={}
   for name in m.ARMS:
    delta=effect if name in ('unconditional_full','no_rgb_coordinate','weighted_constraints','uniform_weighted_constraints') else 0.
    arms[name]={'iou':.3+delta,'coordinate_errors':{'center_L2_m':.1-delta,'size_L2_m':.2,'abs_error6_m':[.1-delta,0,0,.2,0,0]},'actual_changed':bool(delta),'actual_changed_axes':[bool(delta),False,False,False,False,False],'status':'synthetic'}
   units=[{'case_id':sid,'coordinate':j,'candidate':active and j==0,'score':score if j==0 else 0.,'score_defined':active and j==0,'effect':effect if j==0 else 0.,'coordinate_error_reduction_m':effect if j==0 else 0.} for j in range(6)]
   row={'id':sid,'platform':platform,'scene':platform+str(i% (2 if platform=='drone' else 3)),'arms':arms,'units_coordinate_application':units,'unit_RGB_term':{'case_id':sid,'candidate':False,'score':0.,'score_defined':False,'effect':0.,'center_error_reduction_m':0.,'size_error_reduction_m':0.},'case_unit':{'case_id':sid,'candidate':active,'score':score if active else 0.,'score_defined':active,'effect':effect},'case_controls':{k:{'value':score if k=='reliability' else 0.,'defined':active} for k in m.CONTROL_TO_ARM},'gate_selected':dict.fromkeys(m.GATES,False)};rows.append(row)
 for platform in ('drone','quad'):
  pp=[r for r in rows if r['platform']==platform and r['case_unit']['candidate']];ids=[r['id'] for r in pp]
  for control,arm in m.CONTROL_TO_ARM.items():
   scores={r['id']:0. if control=='hash_random' else r['case_controls'][control]['value'] for r in pp};sel=m.deterministic_select(ids,.5,scores,dict.fromkeys(ids,True),salt=m.TIE_SALT,platform=platform)['selected']
   for r in rows:
    if r['platform']==platform:
     chosen=r['id'] in sel;r['gate_selected'][arm]=chosen;r['arms'][arm]=copy.deepcopy(r['arms']['unconditional_full' if chosen else 'source'])
 out={'state':'R11_reliability_scored_after_all_arm_seal','n':320,**m.summarize(rows),'rows_PRIVATE':rows}
 with tempfile.TemporaryDirectory(prefix='R11_synthetic_reduce_',dir=D) as temp:
  temp=Path(temp)
  for name,data in [('RESULTS_PRIVATE.json',out),('SCALAR_SUMMARY.json',{k:v for k,v in out.items() if k!='rows_PRIVATE'})]:(temp/name).write_text(json.dumps(data,allow_nan=False)+'\n')
  seal={'files':{p.name:auditor.sha(p) for p in temp.iterdir()},'prediction_seal_sha256':'synthetic','release_sha256':'synthetic'};(temp/'SEALED.json').write_text(json.dumps(seal))
  auditor.main(SimpleNamespace(scored=temp,output=temp/'audit.json'));receipt=json.loads((temp/'audit.json').read_text());assert receipt['state'].startswith('PASS')
 fixture={'state':'PASS_CPU_SYNTHETIC_REDUCTION','checks':receipt['numeric_or_exact_checks'],'max_absolute_difference':receipt['max_absolute_numeric_difference'],'all11arm_means_strictthresholds_scene_pairs_discrimination_verified':True,'no_dataset_annotations_GT_model_GPU':True,'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('audit_scored.py','metrics.py','discrimination.py','check_reduction_cpu.py')}}
 (D/'CPU_REDUCTION_FIXTURE.json').write_text(json.dumps(fixture,indent=2)+'\n');print(json.dumps(fixture))
if __name__=='__main__':main()
