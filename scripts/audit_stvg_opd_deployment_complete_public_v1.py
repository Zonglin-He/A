"""Independent scalar-only verification of all 128 complete measured paths."""
import json,math,sys
from pathlib import Path
import numpy as np
REL='results/stvg_opd_claim_strengthening/2026-10-13/deployment_latency_complete'
KEYS=['row_decode_preprocess_seconds','native_student_seconds','fresh_expert_seconds',
      'adaptation_seconds','final_readout_seconds','writeback_output_dispatch_seconds']
def read(p):return json.loads(p.read_text())
def summary(v):return dict(median=float(np.median(v)),p95=float(np.quantile(v,.95)),minimum=float(min(v)),maximum=float(max(v)))
def compare(a,b):
    assert set(a)==set(b)
    for k in a:assert abs(a[k]-b[k])<1e-9,(k,a[k],b[k])
    return len(a)
def run(root):
    out=root/REL;saved=read(out/'PAIRED_DEPLOYMENT_SUMMARY.json');count=checks=dino=0
    assert saved['status']=='actual_complete_paired_profiling' and saved['actual_timed_query_paths']==128
    for ds in ['hc2','vidstg']:
        cells={};hardware={};all_rows=[]
        for method in ['Frozen','Full']:
            rr=[]
            for repeat in [0,1]:
                obj=read(out/ds/method/f'repeat{repeat}.json');assert obj['status']=='complete_actual_timing' and obj['GT_read'] is False
                assert obj['resident_models']==(['student','expert'] if method=='Full' else ['student'])
                hardware[method]=obj['hardware'];assert len(obj['rows'])==16
                for r in obj['rows']:
                    assert r['dataset']==ds and r['method']==method and r['repeat']==repeat
                    assert not r['independent_audit_inside_timer'] and not r['full_trace_inside_timer']
                    assert abs(math.fsum(r[k] for k in KEYS)-r['total_seconds'])<1e-7
                    assert r['total_seconds']>0 and all(r[k]>=0 for k in KEYS)
                    assert 'query_ordinal' not in r and 'source_id' not in r
                    key=method,repeat,r['profile_id'];assert key not in cells;cells[key]=r;rr.append(r);count+=1;checks+=8
                    dino+=r['actual_new_DINO_calls']
                    if method=='Frozen':assert r['actual_new_DINO_calls']==r['admitted_references']==r['backward_steps']==0
            assert {r['profile_id'] for r in rr}==set(range(16))
            m=saved['datasets'][ds]['methods'][method]
            assert m['actual_query_paths']==32 and m['unique_queries']==16
            for repeat in [0,1]:checks+=compare(m['per_repeat'][str(repeat)],summary([r['total_seconds'] for r in rr if r['repeat']==repeat]))
            checks+=compare(m['all_repeated_paths'],summary([r['total_seconds'] for r in rr]))
            checks+=compare(m['per_query_repeat_mean'],summary([math.fsum(cells[method,j,i]['total_seconds'] for j in [0,1])/2 for i in range(16)]))
            for k in KEYS:checks+=compare(m['stage_timings'][k],summary([r[k] for r in rr]))
            assert m['peak_allocated_bytes']==max(r['peak_allocated_bytes'] for r in rr)
            assert m['peak_reserved_bytes']==max(r['peak_reserved_bytes'] for r in rr)
            assert m['actual_new_DINO_calls']==sum(r['actual_new_DINO_calls'] for r in rr)
            all_rows+=rr
        assert saved['datasets'][ds]['hardware']==hardware
        extra=[];ratios=[]
        for i in range(16):
            full=math.fsum(cells['Full',j,i]['total_seconds'] for j in [0,1])/2
            frozen=math.fsum(cells['Frozen',j,i]['total_seconds'] for j in [0,1])/2
            extra.append(full-frozen);ratios.append(full/frozen)
        z=saved['datasets'][ds];checks+=compare(z['paired_extra_seconds'],summary(extra));checks+=compare(z['paired_full_over_frozen'],summary(ratios))
        assert max(abs(a-b) for a,b in zip(z['paired_extra_seconds_all'],extra))<1e-9
        c=read(out/'MEASUREMENT_CONTEXT.json')['datasets'][ds]
        full=[r for r in all_rows if r['method']=='Full']
        assert c['Full_no_DINO_paths']==sum(r['actual_new_DINO_calls']==0 for r in full)
        assert c['Full_no_admitted_paths']==sum(r['admitted_references']==0 for r in full)
        assert c['Full_no_backward_paths']==sum(r['backward_steps']==0 for r in full)
    assert count==128 and dino==224 and saved['GT_read'] is False
    return dict(status='pass',scope='all128 actual measured scalar paths/stage sums/16 matched queries each/two repeats/all median-p95 and negative paired costs/memory/fresh expert/no-information counts',
        actual_timed_query_paths=128,unique_queries=32,scalar_comparisons=checks,actual_timed_DINO_calls=dino,
        new_GPU_fits=0,new_DINO_calls=0,new_GT_scoring_calls=0,CPU_only=True,GT_read=False)
if __name__=='__main__':print(json.dumps(run(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])))
