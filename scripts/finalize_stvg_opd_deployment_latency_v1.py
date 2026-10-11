"""Root readback of every matched measured output, paired timings and memory."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_opd_deployment_latency_v1 import BASE,PUB,read,write,status,sha,load,verify,original


def summary(values):
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),
                minimum=float(min(values)),maximum=float(max(values)))


def run():
    verify();assert read(BASE/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    import torch
    torch.set_num_threads(2)
    results={};count=0;scalar_checks=0;coords=0
    keys=['row_decode_preprocess_seconds','native_student_seconds','fresh_expert_seconds',
          'adaptation_seconds','final_readout_seconds','writeback_output_dispatch_seconds']
    for ds,s in read(BASE/'DESIGN_LOCK.json')['stages'].items():
        methods={};cells={};hardware={}
        for method in ['Frozen','Full']:
            rr=[]
            for repeat in [0,1]:
                done=read(BASE/'timing'/ds/method/str(repeat)/'COMPLETION.json')
                p=PUB/ds/method/f'repeat{repeat}.json';assert sha(p)==done['timing_rows_sha256']
                z=read(p);hardware[method]=z['hardware'];assert z['status']=='complete_actual_timing'
                for row in z['rows']:
                    assert abs(math.fsum(row[k] for k in keys)-row['total_seconds'])<1e-7
                    assert row['method']==method and row['repeat']==repeat
                    assert not row['independent_audit_inside_timer'] and not row['full_trace_inside_timer']
                    b=s['bindings'][row['profile_id']];_,old,_,inp=original(b)
                    proof=BASE/'timing'/ds/method/str(repeat)/'outputs'/f"{row['profile_id']:02}.pt"
                    assert sha(proof)==row['output_sha256'];rc=read(proof.with_suffix('.json'))
                    assert rc['status']=='pass' and rc['sha256']==sha(proof)
                    value=load(proof);output=value['output'];assert not value['GT_read']
                    reference=old['fit']['final'] if method=='Full' else inp['native_boxes']
                    actual=torch.tensor(output['boxes'],dtype=reference.dtype)
                    assert torch.equal(actual,reference);coords+=reference.numel()
                    assert output['physical_interval']==inp['interval'] and output['frame_ids']==inp['frame_ids']
                    if method=='Full':
                        assert value['config']==old['fit']['config']
                        for n,v in value['state'].items():assert torch.equal(v,old['fit']['state'][n]);coords+=v.numel()
                        expected={n:(torch.zeros_like(v) if n=='spatial.query_residual' else
                            v+(old['fit']['state'][n]-v)*value['config']['writeback']) for n,v in old['fit']['initial'].items()}
                        for n,v in expected.items():assert torch.equal(v,value['committed'][n]);coords+=v.numel()
                        for n,v in expected.items():assert torch.equal(v,value['actual_writeback'][n].cpu());coords+=v.numel()
                    cells[method,repeat,row['profile_id']]=row;rr.append(row);count+=1;scalar_checks+=7
            assert len(rr)==32
            methods[method]=dict(actual_query_paths=32,unique_queries=16,
                per_repeat={str(i):summary([r['total_seconds'] for r in rr if r['repeat']==i]) for i in [0,1]},
                all_repeated_paths=summary([r['total_seconds'] for r in rr]),
                per_query_repeat_mean=summary([(cells[method,0,i]['total_seconds']+cells[method,1,i]['total_seconds'])/2 for i in range(16)]),
                stage_timings={k:summary([r[k] for r in rr]) for k in keys},
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rr),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rr),
                actual_new_DINO_calls=sum(r['actual_new_DINO_calls'] for r in rr))
        extra=[];ratio=[]
        for i in range(16):
            full=math.fsum(cells['Full',j,i]['total_seconds'] for j in [0,1])/2
            frozen=math.fsum(cells['Frozen',j,i]['total_seconds'] for j in [0,1])/2
            extra.append(full-frozen);ratio.append(full/frozen)
        results[ds]=dict(methods=methods,hardware=hardware,paired_extra_seconds=summary(extra),
            paired_full_over_frozen=summary(ratio),paired_extra_seconds_all=extra,
            relative_cost_is_per_matched_query_not_difference_of_medians=True)
    assert count==128
    write(PUB/'PAIRED_DEPLOYMENT_SUMMARY.json',dict(status='actual_complete_paired_profiling',datasets=results,
        unique_queries=32,actual_timed_query_paths=128,paired_repeats=2,
        parser_cache_prepared=True,model_startup_reported_separately=True,OS_page_cache_cold=False,
        independent_math_and_full_evidence_trace_outside_timer=True,GT_read=False))
    lines=['# Paired Full/Frozen deployment profiling','','Fixed 16 original queries per target, two paired repeats. Each application frame cache is empty; OS pages are not evicted. Resident weights per process and saved deterministic subject parses. Source and Full parameters are unchanged. Full always calls the real frozen expert; Frozen keeps only its student model resident. Full complete-round qualification and timed trace-off state/output checks passed.','','| Target | Path | Median (s) | p95 (s) | Peak allocated (GiB) |','|---|---|---:|---:|---:|']
    for ds,z in results.items():
        for method,v in z['methods'].items():
            t=v['per_query_repeat_mean'];lines.append(f"| {ds} | {method} | {t['median']:.3f} | {t['p95']:.3f} | {v['peak_allocated_bytes']/2**30:.3f} |")
        e=z['paired_extra_seconds'];r=z['paired_full_over_frozen']
        lines.append(f"\n{ds}: paired additional cost median {e['median']:.3f}s, p95 {e['p95']:.3f}s; paired Full/Frozen ratio median {r['median']:.3f}, p95 {r['p95']:.3f}.")
    lines+=['','The complete measured path includes row/cache loading, decoding, student, expert, adaptation, final readout, consolidation and CPU output materialization. Audit-only CPU mathematics, full round serialization and diagnostic extra decoder outputs are excluded and separately qualified. These are local measurements under the stated cache contract; they do not include network access, fresh external language parsing or process startup. No low-latency conclusion is asserted before these actual measurements.']
    (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.4))
    for ax,(ds,z) in zip(axes,results.items()):
        x=np.arange(2);med=[z['methods'][m]['per_query_repeat_mean']['median'] for m in ['Frozen','Full']]
        p95=[z['methods'][m]['per_query_repeat_mean']['p95'] for m in ['Frozen','Full']]
        ax.bar(x-.18,med,.36,label='Median',color='#237ca3');ax.bar(x+.18,p95,.36,label='p95',color='#df7959')
        ax.set_xticks(x,['Frozen','Full']);ax.set_ylabel('Complete query latency (s)')
        ax.set_title('HC-STVG-v2' if ds=='hc2' else 'VidSTG');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False);fig.tight_layout()
    out=ROOT/'output/pdf/stvg_opd_claim_strengthening';out.mkdir(parents=True,exist_ok=True)
    pdf=out/'paired_deployment_latency.pdf';fig.savefig(pdf);fig.savefig(pdf.with_suffix('.png'),dpi=180);plt.close(fig)
    write(BASE/'ROOT_COMPLETE_READBACK.json',dict(status='pass',actual_timed_query_paths=count,
        unique_queries=32,paired_repeats=2,independent_stage_sum_checks=scalar_checks,
        exact_old_prediction_state_and_writeback_coordinates=coords,GT_read=False,
        statistics_sha256=sha(PUB/'PAIRED_DEPLOYMENT_SUMMARY.json'),time=time.time()))
    status(BASE/'STATUS.json',dict(status='pending_actual_root_PDF_pixel_view_and_publication',
        planned_unique_queries=32,completed_unique_queries=32,actual_timed_query_paths=128,
        experiment_complete=False,GT_read=False,time=time.time()))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_root_PDF_pixel_view_and_publication',
        experiment_complete=False,root_readback_sha256=sha(BASE/'ROOT_COMPLETE_READBACK.json'),
        pdf_sha256=sha(pdf),GT_read=False,time=time.time()))


if __name__=='__main__':run()
