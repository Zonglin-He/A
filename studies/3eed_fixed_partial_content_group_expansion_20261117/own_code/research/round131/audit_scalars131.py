"""Independent scalar/weight/source-aggregation audit, no new task GT."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round131';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 rows=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];assert len(rows)==1798;methods=rd(O/'ALL_METHOD_ROWS.json');maximum=0.;strata={(s['target'],s['scene_PRIVATE']):s for s in rd(H/'STRATA_PRIVATE.json')}
 for r in rows:
  s=strata[(r['target'],r['scene_PRIVATE'])];assert abs(r['sampling_weight']-s['population_content_groups']/s['selected_content_groups'])<1e-12
 for a in methods+rd(O/'ALL_SCENE_ROWS_PRIVATE.json'):
  rr=[r for r in rows if r['source']==a['source'] and r['target']==a['target'] and ('scene_PRIVATE' not in a or r['scene_PRIVATE']==a['scene_PRIVATE'])];assert len(rr)==a['n'];w=[r['sampling_weight'] if a['aggregation']=='design_weighted' else 1. for r in rr];y=[r['arms'][a['arm']]['IoU'] for r in rr]
  for k,vals in [('mIoU_percent',y),('Acc25_percent',[v>.25 for v in y]),('Acc50_percent',[v>.5 for v in y])]:v=sum(x*q for x,q in zip(vals,w))*100/sum(w);maximum=max(maximum,abs(v-a[k]));assert abs(v-a[k])<1e-10
 for a in rd(O/'AGGREGATE_ROWS.json'):
  rr=[r for r in methods if r['aggregation']==a['aggregation'] and r['arm']==a['arm'] and (a['group']=='all_six_equal' or r['target']==a['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(r[k] for r in rr)/len(rr)-a[k])<1e-10
 for p in rd(O/'ALL_PAIRED_ROWS.json'):
  rr=[r for r in rows if r['source']==p['source'] and r['target']==p['target']];d=[r['arms'][p['method']]['IoU']-r['arms'][p['control']]['IoU'] for r in rr];w=[r['sampling_weight'] if p['aggregation']=='design_weighted' else 1. for r in rr];assert abs(sum(x*q for x,q in zip(d,w))*100/sum(w)-p['gain_pp'])<1e-10;assert sum(x>1e-12 for x in d)==p['better'] and sum(x< -1e-12 for x in d)==p['worse'] and sum(abs(x)<=1e-12 for x in d)==p['same']
 out={'status':'PASS_INDEPENDENT_ALL_SCALARS_INPUT_DESIGN_WEIGHTS_AND_PAIRS','n':len(rows),'arms':4,'max_scalar_error':maximum,'new_GT_reads':False,'no_scene_or_endpoint_selection':True};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
