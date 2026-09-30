"""Verify sealed Paper48 panel evidence without new model execution or GT scoring."""
import sys,json,time,hashlib,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from scripts.tastvg_paper48_common_v1 import verify
from scripts.decota_matrix_common_v1 import read,sha,load,write
panel=sys.argv[1];assert panel in ['P1','P2','P3_b0','P3_b25','P3_b100','P5']
if panel=='P5':
 from scripts.tastvg_paper48_p5_common_v1 import verify as verify_hc2
 plan=verify_hc2()
else:plan=verify(panel)
b=root/'artifacts/tastvg_paper48_v1'/panel;bar=read(b/'PREDICTION_BARRIER.json');completion=read(b/'COMPLETION.json');exposure=read(b/'GT_EXPOSURE.json')
assert completion['audit_sha256']==sha(b/'AUDIT.json') and read(b/'AUDIT.json')['status']=='pass'
assert exposure['prediction_barrier_sha256']==sha(b/'PREDICTION_BARRIER.json') and bar['time']<exposure['time']<completion['time']
assert bar['cells']==len(bar['files'])==plan['total'] and bar['GT_read'] is False and bar['model_restored'] is True
rows=read(b/'ROWS.json');index={(r['condition'],r['order'],r['arrival']):r for r in rows}
def check(item):
 rel,h=item;f=b/rel;assert sha(f)==h;rr=read(f);pred=f.with_suffix('.pt');assert sha(pred)==rr['sha256']
 _,cond,order,name=Path(rel).parts;arrival=int(Path(name).stem);parent=plan['orders'][order][arrival]
 assert rr['parent']==parent==index[cond,order,arrival]['parent']
 assert rr['updated']==index[cond,order,arrival]['updated']
 return pred if arrival in [0,len(plan['orders'][order])-1] else None
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:ends=[x for x in pool.map(check,bar['files'].items()) if x]
assert len(ends)==len(plan['conditions'])*len(plan['orders'])*2
for f in ends:
 x=load(f);assert x['reinsertion']['full_pipeline_exact'] and x['GT_read'] is False
sec=sum(read(f)['seconds'] for f in (b/'allocations').glob('*.json'))
write(b/'ROOT_READBACK.json',dict(status='pass',sealed_receipts_verified=plan['total'],prediction_payload_hashes_verified=plan['total'],scalar_identity_matches=plan['total'],full_spatial_reinsertion_endpoints=len(ends),GT_read_by_readback=False,new_inference=False,completed_before_scoring=True,prediction_barrier_sha256=sha(b/'PREDICTION_BARRIER.json'),audit_sha256=sha(b/'AUDIT.json'),scalar_rows_sha256=sha(b/'ROWS.json'),online_all_allocations_seconds=sec,time=time.time()))
print('verified',panel,plan['total'],'receipts and payloads;',len(ends),'reinsertion endpoints; seconds',sec)
