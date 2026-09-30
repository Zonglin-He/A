"""Independent public timing aggregation for P4; no models or private data."""
import csv,json,math,sys
from pathlib import Path
import numpy as np

def run(base):
    base=Path(base);summary=json.loads((base/'SUMMARY.json').read_text())
    with (base/'TIMINGS.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k not in ['mode']:r[k]=float(r[k])
    assert len(rows)==200
    assert {(r['mode'],int(r['index'])) for r in rows}=={(m,i) for m in ['Frozen','Ours'] for i in range(100)}
    assert sum(r['logical_expert_calls'] for r in rows)==50
    checks=0
    fields=['end_to_end_cold_seconds','native_process_seconds','load_seconds','decode_seconds','native_seconds','initialization_seconds','adapter_seconds','spatial_process_seconds','temporal_process_seconds','logical_expert_calls']
    for r in rows:
        scheduled=r['mode']=='Ours' and int(r['index'])%4==0
        assert bool(r['expert_scheduled'])==scheduled
        assert r['logical_expert_calls']==2*scheduled
        assert r['end_to_end_cold_seconds']>=sum(r[k] for k in ['native_process_seconds','spatial_process_seconds','temporal_process_seconds'])
        assert r['native_process_seconds']>=sum(r[k] for k in ['load_seconds','decode_seconds','native_seconds','initialization_seconds','adapter_seconds'])
        assert all(math.isfinite(r[k]) and r[k]>=0 for k in fields)
        checks+=5
    for mode in ['Frozen','Ours']:
        for group,saved in summary['results'][mode].items():
            seq=[r for r in rows if r['mode']==mode and (group=='all' or bool(r['expert_scheduled'])==(group=='expert'))]
            assert len(seq)==saved['queries'];checks+=1
            for k in fields:
                a=[r[k] for r in seq];calc={'mean':np.mean(a),'p50':np.median(a),'p95':np.quantile(a,.95)}
                for metric,v in calc.items():assert math.isclose(v,saved[k][metric],rel_tol=1e-12,abs_tol=1e-10),(mode,group,k,metric);checks+=1
            assert max(r['peak_vram_including_experts'] for r in seq)==saved['peak_vram_bytes'];checks+=1
    return dict(status='pass',rows=200,queries=100,logical_specialist_calls=50,checks=checks,scope='Independent cold-start timing/source-order/schedule aggregation; not a warm-service estimate or private prediction audit')

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
