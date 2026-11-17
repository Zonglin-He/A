"""Independent complete scalar/weight/paired audit, cached outputs only."""
from common132 import *
def main():
 O=H132/'scores_attempt001';rows=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];assert len(rows)==1798;methods=rd(O/'ALL_METHOD_ROWS.json');maximum=0.;strata={(s['target'],s['scene_PRIVATE']):s for s in rd(R132/'research/round131/STRATA_PRIVATE.json')}
 for r in rows:s=strata[(r['target'],r['scene_PRIVATE'])];assert abs(r['sampling_weight']-s['population_content_groups']/s['selected_content_groups'])<1e-12
 for a in methods+rd(O/'ALL_SCENE_ROWS_PRIVATE.json'):
  rr=[r for r in rows if r['source']==a['source'] and r['target']==a['target'] and ('scene_PRIVATE' not in a or r['scene_PRIVATE']==a['scene_PRIVATE'])];assert len(rr)==a['n'];w=[r['sampling_weight'] if a['aggregation']=='design_weighted' else 1. for r in rr];y=[r['IoUs'][a['arm']] for r in rr]
  for k,v in [('mIoU_percent',y),('Acc25_percent',[q>.25 for q in y]),('Acc50_percent',[q>.5 for q in y])]:result=sum(x*b for x,b in zip(v,w))*100/sum(w);maximum=max(maximum,abs(result-a[k]));assert abs(result-a[k])<1e-10
 for a in rd(O/'AGGREGATE_ROWS.json'):
  rr=[r for r in methods if r['aggregation']==a['aggregation'] and r['arm']==a['arm'] and (a['group']=='all_six_equal' or r['target']==a['group'])]
  for k in ['mIoU_percent','Acc25_percent','Acc50_percent']:assert abs(sum(r[k] for r in rr)/len(rr)-a[k])<1e-10
 for p in rd(O/'ALL_PAIRED_ROWS.json'):
  rr=[r for r in rows if r['source']==p['source'] and r['target']==p['target']];dd=[r['IoUs'][p['method']]-r['IoUs'][p['control']] for r in rr];w=[r['sampling_weight'] if p['aggregation']=='design_weighted' else 1. for r in rr];assert abs(sum(d*b for d,b in zip(dd,w))*100/sum(w)-p['gain_pp'])<1e-10;assert sum(d>1e-12 for d in dd)==p['better'] and sum(d< -1e-12 for d in dd)==p['worse'] and sum(abs(d)<=1e-12 for d in dd)==p['same']
 dump(H132/'SCALAR_AUDIT.json',{'status':'PASS_ALL12ARMS_PAIRED_DESIGN_WEIGHTS_AND_RECORDINGS','n':len(rows),'task_values':len(rows)*12,'maximum_scalar_error':maximum,'new_GT_read':False});print('SCALAR132',len(rows),maximum)
if __name__=='__main__':main()
