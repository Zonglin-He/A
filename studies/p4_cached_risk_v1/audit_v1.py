"""Separate scalar-only auditor: independent counts, quantile, parent aggregation."""
from run import *
def audit(ds):
 import numpy as np
 start=time.perf_counter();plan=validate();receipt=read(D/ds/'derived_receipt.json');assert receipt['report_sha256']==sha(P/ds/'report.json')and receipt['private_scalar_sha256']==sha(D/ds/'private_scalar_rows.json')and receipt['signed_sha256']==sha(P/ds/'all_signed_rows.tsv')and receipt['table_sha256']==sha(P/ds/'risk_table.tsv')
 rows=load_inputs(ds);assert rows==read(D/ds/'private_scalar_rows.json')['rows'];published=read(P/ds/'report.json')['groups'];checks=0
 for name,rr in groups(rows,ds).items():
  v=published[name];d=[r['delta_pp']for r in rr];n=len(d);assert v['queries']==n and v['parents']==len({r['parent']for r in rr})
  negative=sum(x< -EPS for x in d);equal=sum(abs(x)<=EPS for x in d);positive=sum(x>EPS for x in d);severe=sum(x<=-10 for x in d);assert negative+equal+positive==n
  for k,z in dict(negative_count=negative,equal_count=equal,positive_count=positive,severe_le_minus10_count=severe,negative_not_severe_count=negative-severe).items():assert v[k]==z;checks+=1
  ordered=sorted(d);p=.05*(n-1);lo=math.floor(p);hi=math.ceil(p);quantile=ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
  for k,z in dict(mean_delta_pp=math.fsum(d)/n,negative_percent=100*negative/n,severe_le_minus10_percent=100*severe/n,delta_q05_pp=quantile,source_vIoU_percent=100*math.fsum(r['source_v']for r in rr)/n,current_vIoU_percent=100*math.fsum(r['current_v']for r in rr)/n).items():assert abs(v[k]-z)<1e-12,(name,k,v[k],z);checks+=1
  parent=collections.defaultdict(list)
  for r in rr:parent[r['parent']].append(r['delta_pp'])
  keys=sorted(parent);nn=np.array([len(parent[k])for k in keys]);ss=np.array([[math.fsum(parent[k]),sum(x< -EPS for x in parent[k]),sum(x<=-10 for x in parent[k])]for k in keys]);draw=np.random.default_rng(20261022).multinomial(len(keys),[1/len(keys)]*len(keys),size=10000);b=(draw@ss)/(draw@nn)[:,None];q=np.quantile(b,[.025,.975],axis=0)
  for k,z in [('mean_delta_CI95_pp',q[:,0]),('negative_percent_CI95',100*q[:,1]),('severe_percent_CI95',100*q[:,2])]:assert np.allclose(v[k],z,rtol=0,atol=1e-12),(name,k);checks+=2
 with(P/ds/'all_signed_rows.tsv').open()as f:
  table=list(csv.DictReader(f,delimiter='\t'));assert len(table)==COUNTS[ds]
 for r,z in zip(rows,table):assert float(z['delta_pp'])==r['delta_pp']and(z['negative']=='True')==(r['delta_pp']< -EPS)and(z['severe_le_minus10']=='True')==(r['delta_pp']<=-10)
 assert sum(v['queries']for k,v in published.items()if k.startswith('expert_quality/'))==COUNTS[ds]
 if ds=='vidstg':assert published['query_form/declarative']['queries']==4610 and published['query_form/interrogative']['queries']==5693
 assert 'torch'not in sys.modules
 rec=dict(status='independent_P4_scalar_input_SHA_all_rows_groups_counts_linear_quantile_parent_bootstrap_exact',dataset=ds,queries=len(rows),groups=len(published),scalar_checks=checks,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,plan_fingerprint=plan['fingerprint'],derive_receipt_sha256=sha(D/ds/'derived_receipt.json'),report_sha256=sha(P/ds/'report.json'),all_signed_rows_exact=True,all_group_statistics_independently_recomputed=True,query_forms_exact_original_catalog_join=True,quality_missing_not_zero=True,new_GT_scoring=0,new_model=0,GPU=False)
 atomic(D/ds/'audit_receipt.json',rec);atomic(P/ds/'audit_receipt.json',rec);print(json.dumps(rec))
if __name__=='__main__':
 sys.addaudithook(guard);p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);audit(p.parse_args().dataset)
