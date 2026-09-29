"""Post-seal development scoring; GT is absent from the corruption generator."""
import sys,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_c05_v2 import OUT,OLD,verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats,METRICS

def run():
    p=verify();bar=read(OUT/'C05_BARRIER.json');assert len(bar['files'])==480
    for f,h in bar['files'].items():assert sha(OUT/f)==h
    truth=read(OLD/'GT_SUBSET.json');rows=[];calls=0
    for r in p['rows']:
        gt=truth[r['key']];clean=load(OLD/'capture/clean'/f"{r['ordinal']:03}.pt")['native']
        cm,_=checked_score(clean['boxes'],gt,r['frame_ids'],clean['indices']);calls+=1
        for family in p['families']:
            for pct in p['levels']:
                condition=f'{family}_{pct}';x=load(OUT/'c05'/condition/f"{r['ordinal']:03}.pt");pred=x['native']
                m,_=checked_score(pred['boxes'],gt,r['frame_ids'],pred['indices']);calls+=1
                z=x['generation'];a,b=z['physical_start'],z['physical_end'];s,e=gt['interval']
                overlap=max(0,min(b,e)-max(a,s))/max(1,e-s)
                rows.append(dict(parent=r['ordinal'],condition=condition,family=family,severity=pct,
                    clean={k:cm[k] for k in METRICS},native={k:m[k] for k in METRICS},delta={k:m[k]-cm[k] for k in METRICS},
                    observed_burst_frames=len(z['positions']),actually_changed_frames=len(x['actual_changed_positions']),observations=len(r['frame_ids']),
                    physical_fraction=z['actual_physical_fraction'],observed_fraction=z['actual_observed_fraction'],GT_event_overlap=overlap,
                    overlap_bin='zero' if overlap==0 else 'low_0_to_10pct' if overlap<=.1 else 'high_gt10pct',reused=x['reused'] is not None))
    summary={}
    groups={'panel':rows}
    for family in p['families']:groups[family]=[r for r in rows if r['family']==family]
    for c in sorted({r['condition'] for r in rows}):groups[c]=[r for r in rows if r['condition']==c]
    for name,rr in groups.items():
        d=collections.defaultdict(list)
        for r in rr:d[r['parent']].append(r)
        summary[name]=dict(cells=len(rr),parents=len(d),delta={m:stats([np.mean([r['delta'][m] for r in d[k]]) for k in sorted(d)]) for m in METRICS},
            harms_gt5pp={m:sum(r['delta'][m]<-.05 for r in rr) for m in METRICS},zero_observed_hit=sum(not r['observed_burst_frames'] for r in rr),
            zero_actual_change=sum(not r['actually_changed_frames'] for r in rr),mean_observed_fraction=float(np.mean([r['observed_fraction'] for r in rr])))
    overlap={}
    for name in ['zero','low_0_to_10pct','high_gt10pct']:
        rr=[r for r in rows if r['overlap_bin']==name];d=collections.defaultdict(list)
        for r in rr:d[r['parent']].append(r)
        overlap[name]=dict(cells=len(rr),parents=len(d),delta={m:stats([np.mean([r['delta'][m] for r in d[k]]) for k in sorted(d)]) for m in METRICS},scope='post-hoc association; different parents/composition, not randomized causal comparison')
    qualified=all(summary['panel']['delta'][m]['ci95'][1]<0 for m in ['tIoU','vIoU_corrected'])
    write(OUT/'analysis/C05_ROWS.json',rows);write(OUT/'analysis/C05_SUMMARY.json',summary);write(OUT/'analysis/C05_OVERLAP.json',overlap)
    write(OUT/'C05_DECISION.json',dict(status='completed',measurement='dual_implementation_checked',benchmark_development_gate=qualified,scope='32 previously exposed development parents; no confirmation claim',C2_T='next_authorized_fixed_old_panel',adaptation=False))
    write(OUT/'C05_AUDIT.json',dict(status='pass',cells=480,parents=32,metric_calls=calls,barrier_sha256=sha(OUT/'C05_BARRIER.json'),GT_read_after_prediction_seal=True,GT_generation=False,time=time.time()))
    print('C05',summary['panel'],'QUALIFIED',qualified)

if __name__=='__main__':run()
