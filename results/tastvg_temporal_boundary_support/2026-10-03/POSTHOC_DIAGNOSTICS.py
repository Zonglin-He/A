"""Descriptive analysis of sealed anonymous results; never selects a method."""
import json,sys,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(f):return json.loads(f.read_text())
def compute():
    out=dict(scope='posthoc descriptive analysis only, no formula/width/selection changes',panels={})
    for split in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            allrows=read(ROOT/split/ds/'ROWS.json');s=read(ROOT/split/ds/'SUMMARY.json')
            rr=[r for r in allrows if r['condition']!='clean' and r['expert_scheduled']]
            cap=s['corruption']['expert']['metrics']['capacity_gain']['source_values']
            z=dict(corrupt_expert_cells=len(rr),expert_sources=len(cap),
                sources_with_capacity_gain=sum(v>1e-12 for v in cap.values()),
                per_source_capacity_gain=cap,selectors={})
            for arm in ['A8','A32','B32','D32']:
                ll=[r['intervals_normalized'][r['choices'][arm]][1]-r['intervals_normalized'][r['choices'][arm]][0] for r in rr]
                sec=np.array(ll)*np.array([r['duration_seconds'] for r in rr])
                z['selectors'][arm]=dict(mean_window_fraction=float(np.mean(ll)),median_seconds=float(np.median(sec)),
                    duration_under_two_boundary_windows=sum(float(v)<2.0 for v in sec),
                    new_appended_candidate_chosen=sum(r['choices'][arm]>=8 for r in rr))
            pos=[r for r in rr if min(r['details']['D32'][r['choices']['D32']]['start_transition'],
                r['details']['D32'][r['choices']['D32']]['end_transition'])>1e-12]
            z['D32_both_transitions_gt_numerical_epsilon']=dict(cells=len(pos),
                harmed_vs_A8=sum(r['D32_gain']<-1e-12 for r in pos),
                severe_harm_vs_A8_gt5pp=sum(r['D32_gain']<-.05 for r in pos),
                harmed_vs_A32=sum(r['D32_vs_A32_v']<-1e-12 for r in pos))
            z['top1_warning']='pairwise accuracy across different supports is not a matched top1 metric'
            out['panels'][ds+'/'+split]=z
    return out

if __name__=='__main__':
    z=compute();f=ROOT/'POSTHOC_DIAGNOSTICS.json'
    if '--verify' in sys.argv:
        assert read(f)==z;print('POSTHOC_RECOMPUTATION_PASS')
    else:
        assert not f.exists();f.write_text(json.dumps(z,indent=2)+'\n')
        (ROOT/'POSTHOC_PROVENANCE.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_type='sealed anonymous ROWS and independently audited SUMMARY only',
            no_new_GT_coordinates_features_models_or_selection=True),indent=2)+'\n')
        print('POSTHOC_RECORDED')
