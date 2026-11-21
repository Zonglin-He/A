"""Recount the published, de-identified rows and apply the preset feasibility rule."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PUBLIC = HERE/'results'
PLATFORMS = ('waymo','drone','quad')


def parse(s):
    if s=='': return None
    if s=='True': return True
    if s=='False': return False
    try:
        return float(s)
    except ValueError:
        return s


def median(rows,key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else None


def pct(n,d):
    return 100*n/d if d else None


def by_scope(rows):
    scopes = dict(all=rows, validation=[r for r in rows if r['panel']=='validation'],
                  recording_separated=[r for r in rows if r['separation']=='new_recording'],
                  train_extension=[r for r in rows if r['panel']!='validation'])
    for name,h in scopes.items():
        for platform in ('all',*PLATFORMS):
            p=h if platform=='all' else [r for r in h if r['target']==platform]
            for selection in ('all','selected','fallback'):
                g=p if selection=='all' else [r for r in p if r['selected']==(selection=='selected')]
                if g: yield dict(scope=name,platform=platform,branch=selection),g


def summary(g):
    v=[r for r in g if r['frozen_pass'] is False]
    ok=[r for r in g if r['frozen_pass'] is True]
    both=[r for r in g if r['frozen_pass'] is not None and r['truth_pass'] is not None]
    out=dict(n=len(g),valid_window=sum(r['valid_window'] for r in g),recordings=len({r['recording'] for r in g}),
             frozen_flag=len(v),frozen_pass=len(ok),frozen_unknown=len(g)-len(v)-len(ok),joint_observable=len(both))
    for tag in ('truth','native','frozen'):
        sub=[r for r in g if r[tag+'_pass'] is not None]
        bad=[r for r in sub if not r[tag+'_pass']]
        out.update({tag+'_observable':len(sub),tag+'_pass4':len(sub)-len(bad),
                    tag+'_pass0':sum(r[tag+'_pass_zero'] for r in sub),
                    tag+'_pass4_pct':pct(len(sub)-len(bad),len(sub)),
                    tag+'_median_area_covered':median(sub,tag+'_coverage'),
                    tag+'_median_bad_shortfall_px':median(bad,tag+'_worst_px'),
                    tag+'_median_bad_shortfall_fraction':median(bad,tag+'_worst_fraction'),
                    tag+'_bad_zero_overlap':sum(r[tag+'_coverage']==0 for r in bad),
                    tag+'_bad_area_90pct_covered':sum(r[tag+'_coverage']>=.9 for r in bad),
                    tag+'_dense_flag_changes':sum(r[tag+'_pass']!=r[tag+'_dense_pass'] for r in sub)})
    for f,grp in [('flag',v),('pass',ok)]:
        out[f+'_truth_pass']=sum(r['truth_pass'] is True for r in grp)
        out[f+'_truth_fail']=sum(r['truth_pass'] is False for r in grp)
        for cut in (25,50):
            failures=[r for r in grp if r['iou']<=cut/100]
            out[f+'_failed'+str(cut)]=len(failures)
            out[f+'_failure'+str(cut)+'_pct']=pct(len(failures),len(grp))
            out[f+'_fail'+str(cut)+'_truth_pass']=sum(r['truth_pass'] is True for r in failures)
        out[f+'_native_failed50']=sum(r['native_iou']<=.5 for r in grp)
        for measure in ('center_m','dim_relative','yaw_error_deg','iou'):
            out[f+'_median_'+measure]=median(grp,measure)
    for key in ('annot_vs_native_iou2','annot_vs_truth_iou2','annot_vs_window_iou2'):
        out['median_'+key]=median(g,key)
    out['annot_observable']=sum(r['annot_encloses_window'] is not None for r in g)
    out['annot_pass']=sum(r['annot_encloses_window'] is True for r in g)
    out['flagged_sides']=int(sum(r['flagged_sides'] or 0 for r in both))
    out['flagged_sides_covered_by_gt']=int(sum(r['flagged_sides_covered_by_gt'] or 0 for r in both))
    return out


def rates(rec):
    # fixed columns are n GT passes / observed, then n failures / flagged,
    # then n failures / passing for each platform and bootstrap replicate.
    with np.errstate(invalid='ignore', divide='ignore'):
        q=rec[:,::2]/rec[:,1::2]
    return np.column_stack([q[:,0],q[:,1],q[:,2],q[:,1]-q[:,2]])


def cluster(rows):
    rng=np.random.default_rng(20261121)
    main=[]; boot=[]
    for t in PLATFORMS:
        frames=[r for r in rows if r['target']==t]
        buckets=[]
        for recording in sorted({r['recording'] for r in frames}):
            counts=np.zeros(6)
            for r in (a for a in frames if a['recording']==recording):
                if r['truth_pass'] is not None:
                    counts[:2]+=[int(r['truth_pass']),1]
                if r['selected'] and r['frozen_pass'] is not None:
                    idx=4 if r['frozen_pass'] else 2
                    counts[idx:idx+2]+=[int(r['iou']<=.5),1]
            buckets.append(counts)
        buckets=np.asarray(buckets)
        main.append(rates(buckets.sum(0,keepdims=True))[0])
        pick=rng.integers(len(buckets),size=(5000,len(buckets)))
        boot.append(rates(buckets[pick].sum(1)))
    samples=100*np.mean(boot,axis=0)
    result={}
    for i,k in enumerate(('true_window_coverage_pct','selected_flag_failure_pct',
                          'selected_pass_failure_pct','selected_failure_difference_pp')):
        valid=samples[:,i][np.isfinite(samples[:,i])]
        result[k]=dict(estimate=float(100*np.mean(main,axis=0)[i]),
                       cluster95=np.percentile(valid,[2.5,97.5]).tolist(),finite_replicates=len(valid))
    return result


def main():
    with (PUBLIC/'rows.csv').open() as h:
        rows=[{k:parse(v) for k,v in row.items()} for row in csv.DictReader(h)]
    metrics=[key|summary(g) for key,g in by_scope(rows)]
    with (PUBLIC/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(metrics)
    intervals=cluster(rows)
    s=metrics[0]
    gate_truth=s['truth_pass4'] > s['truth_observable']/2
    gate_failure=intervals['selected_failure_difference_pp']['estimate'] > 0
    decision=dict(cohort='ordinary exposed',physical_frames=len(rows),recordings=s['recordings'],
                  majority_of_true_rectangles_cover=gate_truth,
                  higher_within_platform_failure_rate_among_flagged_regions=gate_failure,
                  run_correction=gate_truth and gate_failure,
                  primary_model='C_E_Qsize',platform_stratified_recording_bootstrap=intervals,
                  new_qwen_calls=0,target_updates=0,FINAL_access=False)
    (HERE/'DECISION.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':
    main()
