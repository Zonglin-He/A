"""Independent reductions of all scalar, scene and count-only poststratified results."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round120';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 rr=[r for p in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)];assert len(rr)==2304 and sum(r['protected_alias_fallback'] for r in rr)==154
 def m(rows,k):
  y=[r['arms'][k]['IoU'] for r in rows];return {'n':len(y),'mIoU_percent':sum(y)/len(y)*100,'Acc25_percent':sum(v>.25 for v in y)/len(y)*100,'Acc50_percent':sum(v>.5 for v in y)/len(y)*100}
 for fn in ['ALL_METHOD_ROWS','ALL_SCENE_ROWS_PRIVATE','AGGREGATE_ROWS']:
  for x in rd(O/(fn+'.json')):
   cr=[r for r in rr if r['cohort']==x['cohort']]
   if 'group' in x:
    if x['group']!='all_six_equal':cr=[r for r in cr if r['target']==x['group']]
   else:cr=[r for r in cr if r['source']==x['source'] and r['target']==x['target']]
   if 'scene_PRIVATE' in x:cr=[r for r in cr if r['scene_PRIVATE']==x['scene_PRIVATE']]
   mm=m(cr,x['arm']);assert all(abs(x[k]-v)<1e-9 for k,v in mm.items() if k in x)
 for x in rd(O/'ALL_PAIRED_ROWS.json'):
  cr=[r for r in rr if r['cohort']==x['cohort'] and r['source']==x['source'] and r['target']==x['target']];dd=[r['arms'][x['method']]['IoU']-r['arms'][x['control']]['IoU'] for r in cr];assert len(dd)==x['n'] and abs(sum(dd)/len(dd)*100-x['gain_pp'])<1e-9 and sum(y>1e-12 for y in dd)==x['better'] and sum(y< -1e-12 for y in dd)==x['worse']
 post=rd(O/'POSTSTRATIFIED_METHOD_ROWS.json')
 for x in post:
  cr=[r for r in rr if r['cohort']==x['cohort'] and r['source']==x['source'] and r['target']==x['target']];full=rd(R/f"research/round119/analysis_attempt001/{x['source']}_to_{x['target']}_ALL_ROWS_PRIVATE.json");scs={r['scene_PRIVATE'] for r in cr};support=[r for r in full if r['scene_PRIVATE'] in scs];assert len(support)/len(full)==x['covered_full_mass'];v={k:0. for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}
  for sc in scs:
   w=sum(r['scene_PRIVATE']==sc for r in support)/len(support);a=m([r for r in cr if r['scene_PRIVATE']==sc],x['arm'])
   for k in v:v[k]+=w*a[k]
  assert all(abs(v[k]-x[k])<1e-9 for k in v)
 for x in rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json'):
  p=[r for r in post if r['cohort']==x['cohort'] and r['arm']==x['arm'] and (x['group']=='all_six_equal' or r['target']==x['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(r[k] for r in p)/len(p)-x[k])<1e-9
 out={'state':'PASS_ALL_TASK_PLATFORM_SCENE_PAIRS_POSTSTRATIFICATION','n':2304,'arms':36,'task_values':82944,'new_task_values':62208,'reused_task_values':20736,'aliases':154,'FINAL_access':False};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
