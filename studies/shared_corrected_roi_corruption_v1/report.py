"""Whole384 views/domain seal, independent official scoring, separate cold audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from panel_common import *
import argparse,subprocess,traceback
def score(ds):
 import numpy as np,torch
 from full_score import metric
 torch.set_num_threads(1);validate();seal=read(D/f'{ds}_sealed.json');assert seal['views']==384 and not seal['GT_read'];truth=labels(ds);clean={r['id']:r for r in read(selected.D/'components'/ds/'metrics.json')['private_rows']};out=[];checks=0;maxerr=0.
 for row in rows(ds):
  g=truth[row['id']]
  for view in VIEWS:
   e=seal['files'][row['id']+'.'+view]
   for k,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(D/e[k])==e[h]
   values={}
   with np.load(D/e['path'])as z:
    inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
    for arm in ARMS:values[arm],err=metric(z[arm],inp,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
   c=clean[row['id']];assert c['parent']==str(g['parent']);values.update(CleanSource=c['values']['Source'],CleanFullMAP=c['values']['FullMAP']);out.append(dict(id=row['id']+'.'+view,parent=str(g['parent']),duration_group=c['duration_group'],view=view,values=values))
 folder=P/ds
 for view in VIEWS:
  rr=[r for r in out if r['view']==view];suite().parent_report(rr,ARMS+['CleanSource','CleanFullMAP'],PAIRS,folder/view);suite().signed_tails(rr,PAIRS,folder/view)
 atomic(D/ds/'metrics.json',dict(private_rows=out));rec=dict(status='scored',views=len(out),checks=checks,max_error=maxerr,seal_sha256=sha(D/f'{ds}_sealed.json'),all384sealed_before_GT=True,matched_clean_audit_sha256=sha(selected.D/f'components_{ds}_reporting_receipt.json'));atomic(D/f'{ds}_score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)
def audit(ds):
 import numpy as np,torch
 from panel_core import replay
 torch.set_num_threads(1);validate();seal=read(D/f'{ds}_sealed.json');sp=D/f'{ds}_score_receipt.json';assert read(sp)['seal_sha256']==sha(D/f'{ds}_sealed.json');checks=0;segments={}
 for row in rows(ds):
  for view in VIEWS:
   e=seal['files'][row['id']+'.'+view]
   for k,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(D/e[k])==e[h]
   c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert c['proof']==e['proof']and not c['GT_read'];old,oe=checked_parent(ds,row['id'],view);assert c['proof']['old_compact_sha256']==oe['compact_sha256'];te=e['teacher'];assert sha(te['path'])==te['sha256']and sha(te['receipt'])==te['receipt_sha256'];a=replay(c)
   with np.load(D/e['path'])as z:assert all(np.array_equal(z[k],a[k])for k in a)
   assert np.array_equal(c['Source'],old['Source'])and np.array_equal(c['q0'],old['native_prior']);hs=fixed(parent.D/f'corruption_{ds}_sealed.json')['files'][row['id']+'.'+view];assert sha(parent.D/hs['path'])==hs['sha256']and sha(parent.D/hs['receipt_path'])==hs['receipt_sha256']
   with np.load(parent.D/hs['path'])as z:assert np.array_equal(a['HistoricalTwoDP'],z['FullMAP'])and np.array_equal(a['HistoricalTwoDP_interval'],z['FullMAP_interval'])
   checks+=3;w=e['worker'];key=f"{w['pid']}:{w['startticks']}";s=segments.setdefault(key,dict(views=0,appearance_prefix_calls=0,new_full_Source=0,new_expert=0,prefix_seconds=0.,pipeline_seconds=0.,compatibility_seconds=0.,selected_finish_seconds=0.));s['views']+=1;s['appearance_prefix_calls']+=1
   for k in ['prefix_seconds','pipeline_seconds','compatibility_seconds','selected_finish_seconds']:s[k]+=e[k]
 rr=read(D/ds/'metrics.json')['private_rows']
 for view in VIEWS:
  report=read(P/ds/view/'report.json');vr=[r for r in rr if r['view']==view]
  for arm,mm in report['means'].items():
   for k,v in mm.items():
    vals=[r['values'][arm][k]for r in vr if r['values'][arm][k]is not None];assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12
 rec=dict(status='verified_whole_domain',queries=128,views=384,SHA_checks=checks,all_arrays_cold_rederived_exact=True,all_means_recomputed=True,score_sha256=sha(sp),worker_segments=segments,actual_counts=seal['counts'],model_load=seal['model_load'],new_GT_scoring=0,scope='New appearance-prefix reconstruction only; Source/teacher prior acquisition nonzero. Prefix/compatibility/selected finish/pipeline clocks are elapsed walls with nesting and decode/transform/outer-loop in whole-worker wall; not deployment latency. Three views share the same128parents, not384independent parents. Own new prefix raw disposed after committed cold replay; no arbitrary later repooling.');atomic(D/f'{ds}_reporting_receipt.json',rec);atomic(P/ds/'reporting_receipt.json',rec)
def domain(ds):
 validate();atomic(D/f'{ds}_CPU_worker.json',identity())
 try:
  for action in ['score','audit']:
   atomic(D/f'{ds}_CPU_state.json',dict(status=action));log=(D/f'{ds}_{action}.log').open('ab');c=subprocess.Popen([sys.executable,'-B',__file__,action,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'{ds}_{action}_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/f'{ds}_{action}_exit.json',dict(code=rc));assert rc==0,'Failure preserved; no retry'
  atomic(D/f'{ds}_CPU_state.json',dict(status='complete_scored_audited_pending_root_analysis'))
 except BaseException:atomic(D/f'{ds}_CPU_state.json',dict(status='failed_preserved',error=traceback.format_exc()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['score','audit','domain']);p.add_argument('--dataset',choices=COUNTS,required=True);z=p.parse_args();globals()[z.command](z.dataset)
