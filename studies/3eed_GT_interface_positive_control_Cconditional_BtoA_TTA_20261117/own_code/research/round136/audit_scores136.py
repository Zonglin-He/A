from common136 import *
import numpy as np
def verify_aggregates(rows,agg):
 maxerr=0.
 for a in agg:
  dirs=[]
  for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
   rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)];weights=[r['sampling_weight'] if a['aggregation']=='design_weighted' else 1. for r in rr];den=sum(weights);y=[r['IoUs'][a['arm']] for r in rr];dirs.append([sum(w*v for w,v in zip(weights,y))/den*100,sum(w*(v>.25) for w,v in zip(weights,y))/den*100,sum(w*(v>.5) for w,v in zip(weights,y))/den*100])
  err=float(abs(np.mean(dirs,axis=0)-np.array([a[k] for k in ['mIoU','Acc25','Acc50']])).max());assert err<1e-10;maxerr=max(maxerr,err)
 return maxerr
def main():
 check();oracle=rd(H136/'ORACLE_ALL_ROWS_PRIVATE.json');rows=rd(H136/'METHOD_ALL_ROWS_PRIVATE.json');assert len(oracle)==len(rows)==1798;n=0;values=0
 for stage,rr in [('oracle',oracle),('method',rows)]:
  for r in rr:
   name=r['source']+'_to_'+r['target'];sid=r['id_PRIVATE'];a=np.load(B136/stage/name/'samples_PRIVATE'/(sid+'.npz'));y=np.load(R136/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];assert np.array_equal(a['queries'],a['scores'].argmax(1))
   for arm,q in zip(a['arm_names'],a['queries']):assert abs(float(y[q])-r['IoUs'][arm])<1e-15;values+=1
   if stage=='method':
    b=np.load(R136/'crossplatform/round135/formal'/name/'evidence_PRIVATE'/(sid+'.npz'))['pB'];assert abs(float(y[b.argmax()])-r['IoUs']['Frozen_B'])<1e-15;values+=1
   n+=1
 mx=max(verify_aggregates(oracle,rd(H136/'ORACLE_AGGREGATE.json')),verify_aggregates(rows,rd(H136/'METHOD_AGGREGATE_ROWS.json')));v={r['arm']:r for r in rd(H136/'ORACLE_AGGREGATE.json') if r['aggregation']=='design_weighted'};fraction=(v['GT3_diagnostic']['mIoU']-v['Frozen_A']['mIoU'])/(v['GT_direct_diagnostic']['mIoU']-v['Frozen_A']['mIoU']);out={'status':'PASS','n_each_stage':1798,'checked_task_scalars':values,'max_scalar_error':mx,'GT3_fraction_of_available_GT_direct_gain':fraction,'GT_is_diagnostic_only':True,'FINAL_access':False};dump(H136/'SCALAR_AUDIT.json',out);print(out,flush=True)
if __name__=='__main__':main()
