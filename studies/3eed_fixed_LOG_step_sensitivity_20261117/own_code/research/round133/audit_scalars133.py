"""Independent all-endpoint scalars, cluster weights, matched controls and aggregates."""
from common133 import *
def main():
 O=H133/'scores_attempt001';rows=[r for p in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)];assert len(rows)==1798;methods=rd(O/'ALL_METHOD_ROWS.json');maximum=0.;strata={(s['target'],s['scene_PRIVATE']):s for s in rd(R133/'research/round131/STRATA_PRIVATE.json')}
 for r in rows:s=strata[(r['target'],r['scene_PRIVATE'])];assert abs(r['sampling_weight']-s['population_content_groups']/s['selected_content_groups'])<1e-12
 for a in methods+rd(O/'ALL_SCENE_ROWS_PRIVATE.json'):
  rr=[r for r in rows if r['source']==a['source'] and r['target']==a['target'] and ('scene_PRIVATE' not in a or r['scene_PRIVATE']==a['scene_PRIVATE'])];assert len(rr)==a['n'];w=[r['sampling_weight'] if a['aggregation']=='design_weighted' else 1. for r in rr];y=[r['IoUs'][a['arm']] for r in rr]
  for k,v in [('mIoU_percent',y),('Acc25_percent',[q>.25 for q in y]),('Acc50_percent',[q>.5 for q in y])]:val=sum(x*b for x,b in zip(v,w))*100/sum(w);maximum=max(maximum,abs(val-a[k]));assert abs(val-a[k])<1e-10
 for a in rd(O/'AGGREGATE_ROWS.json'):
  rr=[r for r in methods if r['aggregation']==a['aggregation'] and r['arm']==a['arm'] and (a['group']=='all_six_equal' or r['target']==a['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(r[k] for r in rr)/len(rr)-a[k])<1e-10
 for a in rd(O/'ALL_PAIRED_ROWS.json'):
  rr=[r for r in rows if r['source']==a['source'] and r['target']==a['target']];d=[r['IoUs'][a['method']]-r['IoUs'][a['control']] for r in rr];w=[r['sampling_weight'] if a['aggregation']=='design_weighted' else 1. for r in rr];assert abs(sum(v*b for v,b in zip(d,w))*100/sum(w)-a['gain_pp'])<1e-10;assert sum(v>1e-12 for v in d)==a['better'] and sum(v< -1e-12 for v in d)==a['worse'] and sum(abs(v)<=1e-12 for v in d)==a['same']
 out={'status':'PASS_ALL9ARMS_SCALARS_PAIRS_AND_ORIGINAL_DESIGN_WEIGHTS','n':len(rows),'task_values':len(rows)*9,'max_scalar_error':maximum,'new_GT_read':False,'step_selection_from_target_GT':False};dump(H133/'SCALAR_AUDIT.json',out);print(out)
if __name__=='__main__':main()
