"""Independent Python-only scalar/scene/poststratification and decision reductions."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round127';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 rr=[r for p in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)];assert len(rr)==2304 and sum(r['protected_alias_fallback'] for r in rr)==154
 def met(rows,k):
  y=[r['arms'][k]['IoU'] for r in rows];return {'n':len(y),'mIoU_percent':sum(y)/len(y)*100,'Acc25_percent':sum(v>.25 for v in y)/len(y)*100,'Acc50_percent':sum(v>.5 for v in y)/len(y)*100}
 for fn in ['ALL_METHOD_ROWS','ALL_SCENE_ROWS_PRIVATE','AGGREGATE_ROWS']:
  for x in rd(O/(fn+'.json')):
   cr=[r for r in rr if r['cohort']==x['cohort']]
   if 'group' in x:
    if x['group']!='all_six_equal':cr=[r for r in cr if r['target']==x['group']]
   else:cr=[r for r in cr if r['source']==x['source'] and r['target']==x['target']]
   if 'scene_PRIVATE' in x:cr=[r for r in cr if r['scene_PRIVATE']==x['scene_PRIVATE']]
   assert all(abs(x[k]-v)<1e-9 for k,v in met(cr,x['arm']).items() if k in x)
 for x in rd(O/'ALL_PAIRED_ROWS.json'):
  cr=[r for r in rr if r['cohort']==x['cohort'] and r['source']==x['source'] and r['target']==x['target']];d=[r['arms'][x['method']]['IoU']-r['arms'][x['control']]['IoU'] for r in cr];assert len(d)==x['n'] and abs(sum(d)/len(d)*100-x['gain_pp'])<1e-9 and sum(v>1e-12 for v in d)==x['better'] and sum(v< -1e-12 for v in d)==x['worse']
 post=rd(O/'POSTSTRATIFIED_METHOD_ROWS.json')
 for x in post:
  cr=[r for r in rr if r['cohort']==x['cohort'] and r['source']==x['source'] and r['target']==x['target']];full=rd(R/f"research/round119/analysis_attempt001/{x['source']}_to_{x['target']}_ALL_ROWS_PRIVATE.json");scs={r['scene_PRIVATE'] for r in cr};support=[r for r in full if r['scene_PRIVATE'] in scs];assert len(support)/len(full)==x['covered_full_mass'];values={k:0. for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}
  for sc in scs:
   w=sum(r['scene_PRIVATE']==sc for r in support)/len(support);m=met([r for r in cr if r['scene_PRIVATE']==sc],x['arm'])
   for k in values:values[k]+=w*m[k]
  assert all(abs(values[k]-x[k])<1e-9 for k in values)
 for x in rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json'):
  a=[r for r in post if r['cohort']==x['cohort'] and r['arm']==x['arm'] and (x['group']=='all_six_equal' or r['target']==x['group'])]
  assert all(abs(sum(r[k] for r in a)/len(a)-x[k])<1e-9 for k in ['mIoU_percent','Acc25_percent','Acc50_percent'])
 held=rd(H/'HELDOUT_C_TASK_ROWS.json');dec=rd(H/'DECISION.json');assert len(held)==12 and dec['C_is_descriptive_not_configuration_gate']
 for x in held:
  a=[r for r in rr if r['cohort']==x['cohort'] and r['source']==x['source'] and r['target']==x['target'] and r['C_available'] and r['B_active']];d=[r['arms']['fab_full_C']['IoU']-r['arms']['zero_C']['IoU'] for r in a];g=sum(d)/len(d)*100 if d else None;assert len(d)==x['n'] and (g is None and x['gain_pp'] is None or abs(g-x['gain_pp'])<1e-9)
 controls=['zero_F','direct_full_F','direct_native_bridge_F','wrong_fab_full_F','random_F'];aa=rd(O/'AGGREGATE_ROWS.json');pp=rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json');flags=[]
 for co in ['DEV','EXP']:
  a={x['arm']:x['mIoU_percent'] for x in aa if x['cohort']==co and x['group']=='all_six_equal'};p={x['arm']:x['mIoU_percent'] for x in pp if x['cohort']==co and x['group']=='all_six_equal'};passed=all(a['fab_full_F']>a[c] and p['fab_full_F']>p[c] for c in controls);assert dec['cohorts'][co]['passes_all_prelocked_controls']==passed;flags.append(passed)
 assert dec['development_gate_pass']==all(flags) and not dec['posthoc_primary_switch']
 out={'state':'PASS_ALL_TASK_SCENE_PAIRED_POSTSTRATIFIED_GATES','n':2304,'aliases':154,'arms':21,'task_values':48384,'new_task_values':41472,'reused_task_values':6912,'FINAL_access':False};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
