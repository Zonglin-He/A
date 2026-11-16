"""Independent exact scene-frequency weights and fixed-method reductions."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round119';O=H/'analysis_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 rows=rd(O/'R118_POSTSTRATIFIED_METHOD_ROWS.json');new=rd(O/'R118_POSTSTRATIFIED_AGGREGATE_ROWS.json');n=0
 for x in rows:
  name=x['source']+'_to_'+x['target'];full=rd(O/(name+'_ALL_ROWS_PRIVATE.json'));rr=[]
  for co in (['DEV','EXP'] if x['cohort']=='UNION' else [x['cohort']]):rr+=rd(R/f'research/round118/scores_attempt001/{co}_{name}_ALL_ROWS_PRIVATE.json')
  available={r['scene_PRIVATE'] for r in rr};supported=[r for r in full if r['scene_PRIVATE'] in available];mass=len(supported)/len(full);assert abs(mass-x['covered_full_mass'])<1e-14;v=[0.,0.,0.]
  for sc in available:
   sr=[r for r in rr if r['scene_PRIVATE']==sc];w=sum(r['scene_PRIVATE']==sc for r in supported)/len(supported);yy=[r['arms'][x['arm']]['IoU'] for r in sr];v[0]+=w*sum(yy)/len(yy)*100;v[1]+=w*sum(y>.25 for y in yy)/len(yy)*100;v[2]+=w*sum(y>.5 for y in yy)/len(yy)*100
  assert all(abs(vv-x[k])<1e-9 for k,vv in zip(['mIoU_percent','Acc25_percent','Acc50_percent'],v));n+=1
 for x in new:
  mm=[r for r in rows if r['cohort']==x['cohort'] and r['arm']==x['arm'] and (x['group']=='all_six_equal' or r['target']==x['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(r[k] for r in mm)/len(mm)-x[k])<1e-9
  assert min(r['covered_full_mass'] for r in mm)==x['min_covered_full_mass']
 out={'state':'PASS_FIXED_R118_ALL25_METHODS_COUNT_ONLY_SCENE_POSTSTRATIFICATION','method_rows':n,'aggregate_rows':len(new),'weight_uses_task_labels':False,'no_method_selection':True,'new_model_or_GT_scoring':0,'FINAL_access':False};(H/'POSTSTRATIFICATION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
