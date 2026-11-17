"""Second scalar evaluation from sealed argmaxes and the original GT cache."""
from common135 import *
import numpy as np,collections
def main():
 Q=H135/'scores_attempt001';n=0;check=0;rows=[]
 for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  name=p.name.removesuffix('_ALL_ROWS_PRIVATE.json')
  for r in rd(p):
   a=np.load(B135/'formal'/name/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'));y=np.load(R135/'research/round131/scores_attempt001'/(name+'_'+r['id_PRIVATE']+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];assert np.array_equal(a['queries'],np.argmax(a['scores'],axis=1))
   for k,q in zip(a['arm_names'],a['queries']):assert abs(float(y[q])-r['IoUs'][k])<1e-15;check+=1
   n+=1;rows.append(r)
 aggregate=rd(Q/'AGGREGATE_ROWS.json');maximum=0.
 for a in aggregate:
  total=[]
  for source,target in sorted({(r['source'],r['target']) for r in rows if a['group']=='all_six_equal' or r['target']==a['group']}):
   rr=[r for r in rows if r['source']==source and r['target']==target];den=sum(r['sampling_weight'] if a['aggregation']=='design_weighted' else 1 for r in rr);v=[]
   for metric in ['mIoU_percent','Acc25_percent','Acc50_percent']:
    vals=[]
    for r in rr:
     y=r['IoUs'][a['arm']];x=y if metric=='mIoU_percent' else float(y>(.25 if metric=='Acc25_percent' else .5));vals.append((r['sampling_weight'] if a['aggregation']=='design_weighted' else 1)*x)
    v.append(sum(vals)/den*100)
   total.append(v)
  truth=np.mean(total,axis=0);reported=np.array([a[k] for k in ['mIoU_percent','Acc25_percent','Acc50_percent']]);err=float(abs(truth-reported).max());assert err<1e-10;maximum=max(maximum,err)
 assert n==1798;dump(H135/'SCALAR_AUDIT.json',{'status':'PASS','n':n,'task_values':check,'aggregate_values':len(aggregate)*3,'maximum_error':maximum,'all_negative_rows_retained':True,'FINAL_access':False});print({'n':n,'task_values':check,'maximum_error':maximum})
if __name__=='__main__':main()
