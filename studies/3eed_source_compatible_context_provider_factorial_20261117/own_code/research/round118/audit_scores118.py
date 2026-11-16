"""Independent scalar aggregation from all rows, including aliases and negative tails."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round118';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 mat=rd(R/'crossplatform/round118/MATRIX_SEALED.json');rows=[x for p in O.glob('*_ALL_ROWS_PRIVATE.json') for x in rd(p)];assert len(rows)==2304 and sum(x['protected_alias_fallback'] for x in rows)==154
 def m(rr,k):
  yy=[x['arms'][k]['IoU'] for x in rr];n=len(yy);return {'n':n,'mIoU_percent':100*sum(yy)/n,'Acc25_percent':100*sum(y>.25 for y in yy)/n,'Acc50_percent':100*sum(y>.5 for y in yy)/n}
 for fn in ['ALL_METHOD_ROWS','ALL_SCENE_ROWS_PRIVATE','AGGREGATE_ROWS']:
  for x in rd(O/(fn+'.json')):
   rr=[r for r in rows if r['cohort']==x['cohort']]
   if 'group' in x:
    if x['group']!='all_six_equal':rr=[r for r in rr if r['target']==x['group']]
   else:rr=[r for r in rr if r['source']==x['source'] and r['target']==x['target']]
   if 'scene_PRIVATE' in x:rr=[r for r in rr if r['scene_PRIVATE']==x['scene_PRIVATE']]
   mm=m(rr,x['arm']);assert all(abs(x[k]-v)<1e-9 for k,v in mm.items() if k in x)
 for x in rd(O/'ALL_PAIRED_ROWS.json'):
  dd=[r['arms'][x['method']]['IoU']-r['arms'][x['control']]['IoU'] for r in rows if r['cohort']==x['cohort'] and r['source']==x['source'] and r['target']==x['target']];assert len(dd)==x['n'] and abs(100*sum(dd)/len(dd)-x['gain_pp'])<1e-9 and sum(y>1e-12 for y in dd)==x['better'] and sum(y< -1e-12 for y in dd)==x['worse']
 out={'state':'PASS_INDEPENDENT_ALL_TASK_COHORT_PLATFORM_SCENE_PAIRED_AGGREGATES','n':2304,'arms':25,'readouts':57600,'new_readouts':43776,'reused_readouts':13824,'aliases':154,'FINAL_access':False};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
