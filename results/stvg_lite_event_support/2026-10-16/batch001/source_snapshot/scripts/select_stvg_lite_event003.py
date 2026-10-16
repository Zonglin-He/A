"""One predeclared structural selection from completely sealed development arms."""
import math
import statistics
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,sha,verify


def means(path,arm):
    doc=read(path);assert doc['status']=='complete' and doc['count']==128
    rows=doc['rows'];out={}
    for ds in ('hc2','vidstg'):
        vals=[r['metrics']['m_vIoU'] for r in rows if r['dataset']==ds and r['arm']==arm]
        assert len(vals)==64 and all(v is not None and math.isfinite(v) for v in vals)
        out[ds]=100*math.fsum(vals)/64
    return out


def efficiency(path):
    doc=read(path);rows=doc['records'];out={}
    assert doc['status']=='complete' and doc['count']==128 and not doc['GT_read']
    for ds in ('hc2','vidstg'):
        cases=[r for r in rows if r['dataset']==ds];assert len(cases)==64
        actual=[r['fit_cost'] for r in cases if r['has_reference']]
        counts=[sum(v['decode_counts'].values()) for v in actual]
        times=[v['fit_wall_cuda_synchronized_seconds'] for v in actual]
        out[ds]=dict(actual_fit_count=len(actual),decoder_calls=sum(counts),median_fit_seconds=statistics.median(times),accepted=sum(r['accepted'] for r in cases),proposed=sum(r['proposed'] for r in cases))
    return out


def main():
    from scripts.stvg_lite_recovery003_common import verify_recovery003
    verify_recovery003();verify();budget=read(BASE/'DESIGN_LOCK.json')['structural_selection_max_per_domain_loss_pp']
    trials={};bindings={}
    for tid in ('L000','L001','L002'):
        b=BASE/'trials'/tid
        for name,status in [('ROOT_READBACK.json','pass'),('CPU_COMPLETION.json','complete')]:
            d=read(b/name);assert d['status']==status and d['count']==128
            bindings[str((b/name).relative_to(ROOT))]=sha(b/name)
        score=b/'POSTSEAL_ROWS.json';cost=b/'MATCHED_EFFICIENCY_READBACK.json'
        trials[tid]=dict(means_pp=means(score,'TS_M2'),efficiency=efficiency(cost),recipe=read(b/'DESIGN_LOCK.json')['recipe'])
        bindings[str(score.relative_to(ROOT))]=sha(score);bindings[str(cost.relative_to(ROOT))]=sha(cost)
    baseline=trials['L000']['means_pp'];eligible=[]
    for tid in ('L001','L002'):
        row=trials[tid];row['delta_vs_full_pp']={ds:row['means_pp'][ds]-baseline[ds] for ds in baseline}
        row['within_budget']=all(v>=-budget-1e-12 for v in row['delta_vs_full_pp'].values())
        if row['within_budget']:eligible.append(tid)
    # Calls are deterministic method work, wall time is measured corroboration.
    def costkey(tid):
        c=trials[tid]['efficiency']
        return (sum(c[ds]['decoder_calls'] for ds in c),sum(c[ds]['median_fit_seconds'] for ds in c),0 if tid=='L002' else 1)
    selected=min(eligible,key=costkey) if eligible else 'L000'
    t=BASE/'temporal/revision003';complete=read(t/'CPU_COMPLETION.json');assert complete['status']=='complete' and complete['count']==128
    temporal_score=t/'POSTSEAL_ROWS.json'
    old=means(temporal_score,'TS_offset_trust');lite=means(temporal_score,'TS_temporal_lite')
    delta={ds:lite[ds]-old[ds] for ds in old};keep_lite=all(v>=-budget-1e-12 for v in delta.values())
    bindings[str(temporal_score.relative_to(ROOT))]=sha(temporal_score)
    bindings[str((t/'CPU_COMPLETION.json').relative_to(ROOT))]=sha(t/'CPU_COMPLETION.json')
    write(BASE/'STRUCTURAL_SELECTION_LOCK.json',dict(status='locked',scope='development structural decision, not independent confirmation',count=128,selected_spatial_trial=selected,selected_spatial_recipe=trials[selected]['recipe'],selected_temporal='temporal_lite' if keep_lite else 'offset_trust',temporal_delta_pp=delta,temporal_means_pp={'offset_trust':old,'temporal_lite':lite},spatial_trials=trials,structural_loss_budget_pp=budget,bindings=bindings,GT_read=True,GT_used_for_online_reference_or_step=False,CURRENT_promoted=False,bootstrap_executed=False))
    print('STRUCTURAL_SELECTION_LOCKED',selected,'temporal_lite' if keep_lite else 'offset_trust',flush=True)


if __name__=='__main__':main()
