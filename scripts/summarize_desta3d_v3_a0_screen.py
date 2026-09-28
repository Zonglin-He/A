"""Finish only the authorized A0 branch, preserving all negative results."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a0_fast_screen import D,OUT
from vg_tta.desta3d_v3_oracle_io import read,write,sha,total_prior

def main():
    assert not (D/'ROOT_COMPLETION_SUMMARY.json').exists()
    cache=read(D/'ROOT_CACHE_READBACK.json');direction=read(D/'ROOT_DIRECTION_READBACK.json');assert cache['status']==direction['status']=='passed'
    native=None
    if direction['decision']=='native_dev64':native=read(D/'native/independent_readback_v1/REPORT.json')
    decision=native['decision'] if native else direction['decision']
    receipts=[]
    for p in sorted(OUT.glob('a0_*/RECEIPT.json')):
        r=read(p);receipts.append(dict(path=str(p),sha256=sha(p),**r))
    assert all(r['status']=='completed' for r in receipts),'Failures need explicit recovery addendum, not this normal completion'
    assert len(receipts)==(6 if native else 4)
    seconds=sum(x['seconds'] for x in receipts);prior=read(D/'CONFIG.json')['prior_seconds'];assert abs(total_prior()-prior-seconds)<1e-7
    results=dict(status='completed_independently_audited',decision=decision,train_queries=128,train_parents=95,dev_queries=64,dev_parents=16,
        steps=200,seed=20260928,parameters=107648,loss='1-global_coefficient_cosine',direction=direction['summary'],
        native_evaluated=native is not None,native_comparisons=native['comparisons'] if native else None,
        native_query_tails=native['query_tails'] if native else None,native_retention=native['retention'] if native else None,
        source_GT_privilege_and_gradient_supervision=True,exposed_development=True,fresh_read=False,target_read=False,full_expansion_started=False,
        GPU_and_wrapper_seconds=seconds,cumulative_GPU_seconds=prior+seconds,cap=None,cache_errors=cache['errors'],direction_cosine_max_error=direction['max_cosine_error'])
    write(D/'PUBLIC_REPORT.json',results);write(D/'ROOT_COMPLETION_SUMMARY.json',{**results,'receipts':receipts})
    texts={'capacity_or_optimization':'Dev direction median is below0.1 and train below0.3. The fixed A0 does not fit the oracle directions sufficiently at this budget; capacity/optimization is the next screening hypothesis, not an established cause. No native inference or expansion was run.',
      'conditioning_or_generalization':'Train direction median reaches0.3 but Dev is below0.1. The next screening hypothesis is conditioning/generalization. Native inference and expansion were not run.',
      'trust_or_noop_gate_next':'Direction gate passed, but Dev native vIoU does not improve. The only proposed next branch is trust/no-op gating; no further direction-mixer changes or full expansion were run.',
      'branch_collapse_stop':'Dev vIoU improved but a branch met the predeclared systematic-collapse criterion. Full expansion was not run.',
      'eligible_later_full_expansion':'Both direction and native development gates passed. This permits a separately registered larger experiment; the current run did not expand.'}
    lines=['# A0 fast-screen completed','',texts.get(decision,decision),'',
      '|Set|Queries / parents|Defined directions|Mean cosine|Median cosine|','|---|---:|---:|---:|---:|']
    for split,n,p in [('train',128,95),('dev',64,16)]:
        x=direction['summary'][split];lines.append(f"|{split}|{n} / {p}|{x['defined']}|{x['mean']:.9f}|{x['median']:.9f}|")
    lines+=['','Existing StateAwareDirectionMixer hidden128/state33/union256 and fixed radius. One seed,200 actual AdamW steps,800 query occurrences, no best-step/CE/KL/gate. Source GT supplies privilege and analytic oracle targets. Dev is exposed diagnosis; fresh31/388 and target remain untouched. CPU controls do not establish native utility.']
    if native:
        lines+=['','|Native parent macro|tIoU %|sIoU %|vIoU %|','|---|---:|---:|---:|']
        for arm in ['B1','A0']:
            z=native['arms'][arm]['summary']['parent_macro'];lines.append('|'+arm+'|'+'|'.join(f"{100*z[k]:.6f}" for k in ['tIoU','sIoU','vIoU'])+'|')
        lines+=['',json.dumps({k:native[k] for k in ['comparisons','query_tails','retention','systematic_branch_collapse']},indent=2)]
    lines+=['',f'Measured GPU allocation and wrapper seconds: {seconds:.9f}; cumulative {prior+seconds:.9f}, cap=null. Prior failures remain counted.','',
      'Neither screening classification nor a development pass proves a general mechanism, fresh generalization, teacher qualification or target TTA. All missing/invalid cases and negative outcomes remain retained.']
    (D/'REPORT.md').write_text('\n'.join(lines)+'\n');write(D/'COMPLETE.json',dict(time=time.time(),summary_sha=sha(D/'ROOT_COMPLETION_SUMMARY.json'),report_sha=sha(D/'REPORT.md'),decision=decision))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
