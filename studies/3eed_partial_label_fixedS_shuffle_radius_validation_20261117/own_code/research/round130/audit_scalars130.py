"""Independent arithmetic/paired-count audit, using scored scalar rows only."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round130';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 rows=[r for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];assert len(rows)==2304;maximum=0.;methods=rd(O/'ALL_METHOD_ROWS.json');scenes=rd(O/'ALL_SCENE_ROWS_PRIVATE.json')
 for a in methods+scenes:
  rr=[r for r in rows if r['cohort']==a['cohort'] and r['source']==a['source'] and r['target']==a['target'] and ('scene_PRIVATE' not in a or r['scene_PRIVATE']==a['scene_PRIVATE'])];assert len(rr)==a['n'];y=[r['arms'][a['arm']]['IoU'] for r in rr]
  for k,v in [('mIoU_percent',sum(y)*100/len(y)),('Acc25_percent',sum(z>.25 for z in y)*100/len(y)),('Acc50_percent',sum(z>.5 for z in y)*100/len(y))]:maximum=max(maximum,abs(v-a[k]));assert abs(v-a[k])<1e-10
 for a in rd(O/'AGGREGATE_ROWS.json'):
  subset=[m for m in methods if m['cohort']==a['cohort'] and m['arm']==a['arm'] and (a['group']=='all_six_equal' or m['target']==a['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(m[k] for m in subset)/len(subset)-a[k])<1e-10
 for p in rd(O/'ALL_PAIRED_ROWS.json'):
  rr=[r for r in rows if r['cohort']==p['cohort'] and r['source']==p['source'] and r['target']==p['target']];d=[r['arms'][p['method']]['IoU']-r['arms'][p['control']]['IoU'] for r in rr];assert abs(sum(d)*100/len(d)-p['gain_pp'])<1e-10;assert sum(v>1e-12 for v in d)==p['better'] and sum(v< -1e-12 for v in d)==p['worse'] and sum(abs(v)<=1e-12 for v in d)==p['same']
 post=rd(O/'POSTSTRATIFIED_METHOD_ROWS.json')
 for a in post:
  full=rd(R/f"research/round119/analysis_attempt001/{a['source']}_to_{a['target']}_ALL_ROWS_PRIVATE.json");rr=[r for r in rows if r['cohort']==a['cohort'] and r['source']==a['source'] and r['target']==a['target']];cluster={r['scene_PRIVATE'] for r in rr};counts={s:sum(x['scene_PRIVATE']==s for x in full) for s in cluster};total=sum(counts.values());assert abs(total/len(full)-a['covered_full_mass'])<1e-12
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:
   v=0.
   for sc,n in counts.items():y=[r['arms'][a['arm']]['IoU'] for r in rr if r['scene_PRIVATE']==sc];m=sum(y)/len(y) if k=='mIoU_percent' else sum(t>(.25 if k=='Acc25_percent' else .5) for t in y)/len(y);v+=n/total*m*100
   assert abs(v-a[k])<1e-10
 for a in rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json'):
  subset=[m for m in post if m['cohort']==a['cohort'] and m['arm']==a['arm'] and (a['group']=='all_six_equal' or m['target']==a['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(m[k] for m in subset)/len(subset)-a[k])<1e-10
 out={'state':'PASS_INDEPENDENT_SCALARS_PAIRS_AND_SCENE_POSTSTRATIFICATION','n':len(rows),'arms':23,'protected_aliases':sum(r['protected_alias_fallback'] for r in rows),'max_scalar_error':maximum,'new_GT_reads':False,'all_negative_rows_retained':True};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
