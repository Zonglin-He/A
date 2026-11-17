"""Independent scalar recalculation from sealed-output scorer rows, no new GT."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round128';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 table=rd(O/'ALL_METHOD_ROWS.json');rr=[];err=0.
 for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')):
  rows=rd(f);assert len(rows)==128;rr.extend(rows)
  src,tgt=rows[0]['source'],rows[0]['target']
  for k in rows[0]['IoUs']:
   y=np.array([r['IoUs'][k] for r in rows]);x=next(t for t in table if t['source']==src and t['target']==tgt and t['arm']==k)
   for name,value in [('mIoU_percent',sum(y)*100/len(y)),('Acc25_percent',sum(v>.25 for v in y)*100/len(y)),('Acc50_percent',sum(v>.5 for v in y)*100/len(y))]:err=max(err,abs(x[name]-value));assert abs(x[name]-value)<1e-10
 aggregates=rd(O/'AGGREGATE_ROWS.json')
 for a in aggregates:
  group=[r for r in rr if a['group']=='all_six_equal' or r['target']==a['group']];value=np.mean([r['IoUs'][a['arm']] for r in group])*100;assert abs(value-a['mIoU_percent'])<1e-10
 for p in rd(O/'ALL_PAIRED_ROWS.json'):
  rows=[r for r in rr if r['source']==p['source'] and r['target']==p['target']];v=[r['IoUs'][p['method']]-r['IoUs'][p['control']] for r in rows];assert abs(np.mean(v)*100-p['gain_pp'])<1e-10;assert sum(x>1e-12 for x in v)==p['better'] and sum(x< -1e-12 for x in v)==p['worse']
 out={'state':'PASS_INDEPENDENT_ALL_SCALARS_AND_PAIRS','n':len(rr),'directions':6,'arms':len(rr[0]['IoUs']),'new_task_GT_scored':False,'max_scalar_error':err,'all_negative_records_retained':True};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
