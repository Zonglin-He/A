"""Root-level aggregation and parameter influence readback from sealed scores."""
import argparse
import collections
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.desta_native_common import D,OUT,read,write,sha,check_pins,total_prior


def main():
    import numpy as np
    start=time.monotonic()
    reports={phase:read(D/'scores'/phase/'REPORT.json') for phase in ('dev16','dev64','ablations')}
    error=0.;parent_values={}
    for phase,report in reports.items():
        check_pins(read(D/'scores'/phase/'COMPLETE.json')['pins'])
        parent_values[phase]={}
        for arm,rows in report['rows'].items():
            grouped=collections.defaultdict(list)
            for row in rows:grouped[row['source']].append(row['metrics'])
            parent_values[phase][arm]={}
            for metric in ('tIoU','sIoU','vIoU'):
                values={p:statistics.fmean(r[metric] for r in rs) for p,rs in grouped.items()}
                measured=statistics.fmean(values.values())*100
                error=max(error,abs(measured-report['summary_percent'][arm][metric]));assert error<1e-9
                parent_values[phase][arm][metric]=values
                if arm!='B1':
                    base=parent_values[phase]['B1'][metric]
                    d=np.array([(values[p]-base[p])*100 for p in sorted(values)])
                    rng=np.random.default_rng(20260927);samples=rng.integers(0,len(d),size=(10000,len(d)))
                    ci=np.quantile(d[samples].mean(1),[.025,.975])
                    expected=report['comparisons'][arm][metric]
                    assert np.max(abs(ci-expected['bootstrap_ci95_pp']))<1e-8
                    assert abs(float(d.mean())-expected['mean_delta_pp'])<1e-9
    grid=read(D/'CONFIG.json')['grid'];dev=reports['dev16'];factor={}
    ys=np.array([dev['summary_percent'][k]['vIoU'] for k in grid]);grand=float(ys.mean());ss=float(((ys-grand)**2).sum())
    for name in ('steps','radius','temporal_weight'):
        levels=sorted({c[name] for c in grid.values()});means={}
        for value in levels:means[str(value)]=statistics.fmean(dev['summary_percent'][k]['vIoU'] for k,c in grid.items() if c[name]==value)
        lo=levels[0];hi=levels[-1];d=[]
        for parent in sorted(parent_values['dev16']['B1']['vIoU']):
            low=statistics.fmean(parent_values['dev16'][k]['vIoU'][parent] for k,c in grid.items() if c[name]==lo)
            high=statistics.fmean(parent_values['dev16'][k]['vIoU'][parent] for k,c in grid.items() if c[name]==hi)
            d.append((high-low)*100)
        d=np.array(d);rng=np.random.default_rng(20260927);samples=rng.integers(0,16,size=(10000,16))
        component=sum(9*(v-grand)**2 for v in means.values())
        factor[name]=dict(marginal_vIoU_percent=means,range_pp=max(means.values())-min(means.values()),
            high_minus_low_pp=float(d.mean()),high_minus_low_ci95_pp=np.quantile(d[samples].mean(1),[.025,.975]).tolist(),
            main_effect_fraction_config_mean_variance=component/ss if ss else 0.)
    ranking=sorted(factor,key=lambda x:factor[x]['range_pp'],reverse=True)
    selected=reports['dev64']['selected'][0]
    receipts={str(p):read(p) for p in OUT.glob('desta_native_*/RECEIPT.json')}
    actual=sum(r['seconds'] for r in receipts.values())
    report=dict(status='completed_independently_audited',selected=selected,config=grid[selected],
        dev64=reports['dev64']['summary_percent'][selected],baseline=reports['dev64']['summary_percent']['B1'],
        delta=reports['dev64']['comparisons'][selected],factorial_influence=factor,
        influence_rank_by_dev16_marginal_v_span=ranking,root_aggregation_max_abs=error,
        main_effect_residual_interaction_fraction=1-sum(v['main_effect_fraction_config_mean_variance'] for v in factor.values()),
        new_GPU_allocation_and_wrapper_seconds=actual,cumulative_GPU_seconds=total_prior(),cap=None,
        CPU_seconds=time.monotonic()-start,
        limitation='27 configurations tuned on exposed Dev16; top6 selected on exposed Dev64. Parameter influence is Dev16-only, not fresh generalization. Independent TVG plus detector/tracker spatial fallback. No guarantee of positive utility.')
    write(D/'ROOT_COMPLETION_SUMMARY.json',report)
    lines=['# DESTA dual-expert native adaptation: completed results','',
           f'Selected configuration: **{selected}**. All three grid stages completed and independently checked.','',
           '| Dev64 arm | tIoU % | sIoU % | vIoU % | >5pp v harm |','|---|---:|---:|---:|---:|']
    arm_reports=[('B1',reports['dev64']),('GT-R16',reports['dev64']),(selected,reports['dev64']),('T-only',reports['ablations']),('S-only',reports['ablations'])]
    for arm,r in arm_reports:
        s=r['summary_percent'][arm];harm=r['query_tails'].get(arm,{}).get('vIoU',{}).get('harm_gt5pp',0)
        lines.append(f'| {arm} | {s["tIoU"]:.6f} | {s["sIoU"]:.6f} | {s["vIoU"]:.6f} | {harm} |')
    lines+=['','## Parameter influence: complete Dev16 factorial','',
            '| Parameter | Marginal vIoU span (pp) | High minus low (pp) | Paired parent CI |','|---|---:|---:|---|']
    for key in ranking:
        f=factor[key];lines.append(f'| {key} | {f["range_pp"]:.6f} | {f["high_minus_low_pp"]:.6f} | {f["high_minus_low_ci95_pp"]} |')
    lines+=['',report['limitation'],'',
            'GT-R16 is an existing source-GT diagnostic at its historical radius; it is not a fair label-free competitor or a theoretical upper bound.',
            'All final native predictions, missing support, harms and positive examples remain in the local sealed artifacts.',
            f'Actual new GPU allocation plus wrapper wall time: {actual:.3f}s; cumulative {report["cumulative_GPU_seconds"]:.3f}s, cap=null.']
    (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(D/'COMPLETE.json',dict(status=report['status'],pins={str(D/p):sha(D/p) for p in ('ROOT_COMPLETION_SUMMARY.json','REPORT.md')}))
    print('DESTA_FINAL',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['final']);p.parse_args();main()
