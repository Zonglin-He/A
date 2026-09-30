"""Verify sealed P1 evidence without new model execution or GT scoring."""
import sys,json,time,hashlib,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from scripts.tastvg_paper48_common_v1 import verify
from scripts.decota_matrix_common_v1 import read,sha,load,write
b=root/'artifacts/tastvg_paper48_v1/P1';plan=verify('P1');bar=read(b/'PREDICTION_BARRIER.json');completion=read(b/'COMPLETION.json');exposure=read(b/'GT_EXPOSURE.json')
assert completion['audit_sha256']==sha(b/'AUDIT.json') and read(b/'AUDIT.json')['status']=='pass'
assert exposure['prediction_barrier_sha256']==sha(b/'PREDICTION_BARRIER.json') and bar['time']<exposure['time']<completion['time']
assert bar['cells']==len(bar['files'])==8040 and bar['GT_read'] is False and bar['model_restored'] is True
rows=read(b/'ROWS.json');index={(r['condition'],r['order'],r['arrival']):r for r in rows}
def check(item):
 rel,h=item;f=b/rel;assert sha(f)==h;rr=read(f);pred=f.with_suffix('.pt');assert sha(pred)==rr['sha256']
 _,cond,order,name=Path(rel).parts;arrival=int(Path(name).stem);parent=plan['orders'][order][arrival]
 assert rr['parent']==parent==index[cond,order,arrival]['parent']
 assert rr['updated']==index[cond,order,arrival]['updated']
 return pred if arrival in [0,len(plan['orders'][order])-1] else None
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:ends=[x for x in pool.map(check,bar['files'].items()) if x]
assert len(ends)==24
for f in ends:
 x=load(f);assert x['reinsertion']['full_pipeline_exact'] and x['GT_read'] is False
sec=sum(read(f)['seconds'] for f in (b/'allocations').glob('*.json'))
write(b/'ROOT_READBACK.json',dict(status='pass',sealed_receipts_verified=8040,prediction_payload_hashes_verified=8040,scalar_identity_matches=8040,full_spatial_reinsertion_endpoints=24,GT_read_by_readback=False,new_inference=False,completed_before_scoring=True,prediction_barrier_sha256=sha(b/'PREDICTION_BARRIER.json'),audit_sha256=sha(b/'AUDIT.json'),scalar_rows_sha256=sha(b/'ROWS.json'),online_all_allocations_seconds=sec,time=time.time()))
print('verified 8040 receipts/payload hashes/scalar identities and 24 reinsertion endpoints; seconds',sec)
