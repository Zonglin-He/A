"""Separate scalar aggregation from all per-input scalars, not imported score code."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round116';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 plan=rd(R/'crossplatform/round116/MATRIX_SEALED.json');rows=[]
 for p in O.glob('*_ALL_ROWS_PRIVATE.json'):rows+=rd(p)
 assert len(rows)==768 and all(set(r['arms'])==set(plan['arms']) for r in rows)
 def metric(rr,k):
  yy=[x['arms'][k]['IoU'] for x in rr];n=len(yy);return {'n':n,'mIoU_percent':100*sum(yy)/n,'Acc25_percent':100*sum(y>.25 for y in yy)/n,'Acc50_percent':100*sum(y>.5 for y in yy)/n}
 for x in rd(O/'ALL_METHOD_ROWS.json'):
  rr=[r for r in rows if r['source']==x['source'] and r['target']==x['target']];m=metric(rr,x['arm']);assert all(abs(x[k]-v)<1e-9 for k,v in m.items())
 for x in rd(O/'ALL_SCENE_ROWS_PRIVATE.json'):
  rr=[r for r in rows if r['source']==x['source'] and r['target']==x['target'] and r['scene_PRIVATE']==x['scene_PRIVATE']];m=metric(rr,x['arm']);assert all(abs(x[k]-v)<1e-9 for k,v in m.items())
 for x in rd(O/'AGGREGATE_ROWS.json'):
  rr=rows if x['group']=='all_six_equal' else [r for r in rows if r['target']==x['group']];m=metric(rr,x['arm']);assert all(abs(x[k]-m[k])<1e-9 for k in ['mIoU_percent','Acc25_percent','Acc50_percent'])
 for x in rd(O/'ALL_PAIRED_ROWS.json'):
  dd=[r['arms'][x['method']]['IoU']-r['arms'][x['control']]['IoU'] for r in rows if r['source']==x['source'] and r['target']==x['target']];assert len(dd)==x['n'] and abs(100*sum(dd)/len(dd)-x['gain_pp'])<1e-9 and sum(y>1e-12 for y in dd)==x['better'] and sum(y< -1e-12 for y in dd)==x['worse']
 result={'state':'PASS_INDEPENDENT_ALL_TASK_PLATFORM_SCENE_PAIRED_AGGREGATES','n':768,'arms':31,'readouts':23808,'new_readouts':16128,'reused_readouts':7680,'FINAL_access':False};(H/'SCALAR_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print('SCALAR_AUDIT116_PASS',flush=True)
if __name__=='__main__':main()
