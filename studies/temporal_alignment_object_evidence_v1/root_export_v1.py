"""Independent root analysis of whole-domain audited scalars; no GT rescoring."""
from run import *
import numpy as np

def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n=0,mean=None,CI95=None)
    rng=np.random.default_rng(20260920)
    means=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
    return dict(n=len(a),mean=float(a.mean()),CI95=np.quantile(means,[.025,.975]).tolist())

def run(ds):
    validate();start=time.perf_counter();f=D/ds;p=P/ds;audit=read(f/'reporting_receipt.json');score=read(f/'score_receipt.json');seal=read(f/'sealed.json');rr=read(f/'metrics.json')['private_rows']
    assert audit['status']=='verified_whole128_all_arrays_inputs_parent_SHA_means' and audit['all_arrays_cold_rederived_exact']
    assert audit['score_sha256']==sha(f/'score_receipt.json') and audit['metrics_sha256']==sha(f/'metrics.json') and audit['seal_sha256']==sha(f/'sealed.json')
    assert score['seal_sha256']==sha(f/'sealed.json') and score['metrics_sha256']==sha(f/'metrics.json')
    assert len(rr)==128==len({x['parent']for x in rr}) and not seal['GT_read']
    fields=list(rr[0]['values'][ARMS[0]])
    for arm in ARMS:
        for k,v in audit['means'][arm].items():
            vals=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None]
            assert (v is None and not vals) or abs(v-float(np.mean(vals)))<1e-12
    def group(rs):
        out=dict(queries=len(rs),parents=len({r['parent']for r in rs}),means={},aligned_minus_permuted={})
        for arm in ARMS:
            out['means'][arm]={k:float(np.mean(v))if(v:=[r['values'][arm][k]for r in rs if r['values'][arm][k]is not None])else None for k in fields}
        for k in fields:
            # IoU/recall changes are percentage points; boundary/duration retain native units.
            scale=1 if ('seconds'in k or k=='boundary_MAE') else 100
            vv=[scale*(r['values'][ARMS[0]][k]-r['values'][ARMS[1]][k])for r in rs if all(r['values'][arm][k]is not None for arm in ARMS)]
            out['aligned_minus_permuted'][k]={**stats(vv),'unit':'native'if scale==1 else'pp'}
        dt=np.array([100*(r['values'][ARMS[1]]['t']-r['values'][ARMS[0]]['t'])for r in rs]);dv=np.array([100*(r['values'][ARMS[1]]['v']-r['values'][ARMS[0]]['v'])for r in rs])
        out['permutation_relative_to_aligned']=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()),v_improve=int((dv>0).sum()),v_harm=int((dv<0).sum()),v_tie=int((dv==0).sum()))
        return out
    report=dict(dataset=ds,whole=group(rr),duration={g:group([r for r in rr if r['duration_group']==g])for g in ['short_le5s','long_gt5s']},bootstrap=dict(replicates=10000,seed=20260920,unit='paired distinct parent',CI='pointwise 95%, not familywise'))
    interval_changes=box_changes=posterior_changes=0
    for row in rows(ds):
        e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256'] and sha(D/e['receipt_path'])==e['receipt_sha256']
        with np.load(D/e['path'])as z:
            interval_changes+=not np.array_equal(z[ARMS[0]+'_interval'],z[ARMS[1]+'_interval'])
            box_changes+=not np.array_equal(z[ARMS[0]],z[ARMS[1]])
        posterior_changes+=e['diagnostics']['posterior_sha256'][ARMS[0]]!=e['diagnostics']['posterior_sha256'][ARMS[1]]
    diag=[r['diagnostics']for r in rr]
    report['permutation_diagnostics']=dict(total_nodes=sum(x['nodes']for x in diag),fixed_points=sum(x['fixed_points']for x in diag),nonintervened_queries=sum(x['nonintervened']for x in diag),query_mean_index_displacement=float(np.mean([x['mean_absolute_index_displacement']for x in diag])),query_mean_seconds_displacement=float(np.mean([x['mean_absolute_seconds_displacement']for x in diag])),maximum_seconds_displacement=max(x['max_absolute_seconds_displacement']for x in diag),changed_intervals=int(interval_changes),changed_final_box_arrays=int(box_changes),changed_posteriors=int(posterior_changes),all_vector_multisets_inverse_exact=all(x['vector_multiset_inverse_exact']for x in diag),all_spatial_inputs_unchanged=all(x['fixed_spatial_inputs_exact']for x in diag),aligned_arrays_equal_current=all(x['aligned_current_arrays_exact']for x in diag))
    costs=dict(new_Source=0,new_expert=0,new_model=0,new_optimizer=0,new_acquisition=0,inherited_Source_expert_acquisition_nonzero=True,prediction_worker=seal['worker'],prediction_cached_elapsed_wall_seconds=seal['elapsed_wall_seconds'],conditional_query_elapsed_sum=sum(x['conditional_elapsed_seconds']for x in seal['files'].values()),score=score,audit_worker=audit['worker'],audit_elapsed_wall_seconds=audit['elapsed_wall_seconds'],actual_stage_exits={s:read(f/f'{s}_exit_v1.json')for s in ['predict','score','audit']},scope='Cached replay/input checks and official CPU evaluation; not process CPU time or deployment latency. Stage clocks include nested work and must not be added to inherited acquisition as a speed benchmark.')
    assert all(x['actual_subprocess_wait'] and x['exit_code']==0 for x in costs['actual_stage_exits'].values())
    atomic(p/'report.json',report);atomic(p/'cost_segments.json',costs)
    with(p/'all_signed_metrics.tsv').open('w')as h:
        w=csv.writer(h,delimiter='\t',lineterminator='\n');w.writerow(['anonymous_query','duration_group','arm',*fields])
        for i,r in enumerate(rr,1):
            for arm in ARMS:w.writerow([f'Q{i:03d}',r['duration_group'],arm,*[r['values'][arm][k]for k in fields]])
    with(p/'all_paired_deltas.tsv').open('w')as h:
        w=csv.writer(h,delimiter='\t',lineterminator='\n');w.writerow(['anonymous_query','duration_group',*[k+'_aligned_minus_permuted_native'for k in fields],'fixed_points','nodes','mean_seconds_displacement'])
        for i,r in enumerate(rr,1):w.writerow([f'Q{i:03d}',r['duration_group'],*[r['values'][ARMS[0]][k]-r['values'][ARMS[1]][k]if all(r['values'][a][k]is not None for a in ARMS)else None for k in fields],r['diagnostics']['fixed_points'],r['diagnostics']['nodes'],r['diagnostics']['mean_absolute_seconds_displacement']])
    def fmt(s):return f"{s['mean']:+.6f} [{s['CI95'][0]:.6f}, {s['CI95'][1]:.6f}]"
    text=f'# Temporal alignment of object evidence: {ds}\n\n128 queries / 128 distinct parents; complete prediction seal before independent official scoring ({score["official_checks"]} checks, max error {score["max_error"]:.17g}), independent exact cold-array/input/parent/SHA and mean audit. Root analysis only reads audited scalar metrics and saved prediction arrays; no new GT scoring or model/acquisition.\n\n|Method|tIoU %|vIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|\n'
    for arm,m in report['whole']['means'].items():text+=f'|{arm}|'+ '|'.join(f'{100*m[k]:.6f}'for k in ['t','v','vIoU_ge_03','vIoU_ge_05'])+'|\n'
    text+='\n|Subset|Queries|Aligned minus permuted t pp [95% CI]|Aligned minus permuted v pp [95% CI]|\n|---|---:|---:|---:|\n'
    for g,x in [('whole',report['whole']),*report['duration'].items()]:text+=f'|{g}|{x["queries"]}|{fmt(x["aligned_minus_permuted"]["t"])}|{fmt(x["aligned_minus_permuted"]["v"])}|\n'
    text+='\nPositive deltas favor the original temporal correspondence. All full E-ROI vectors are reassigned to fixed physical temporal nodes by a single predeclared query-specific SHA256 permutation; the vector multiset is exactly preserved, while temporally weighted prototypes can change by design. Source q0/boxes, eight anchor positions, spatial geometry/candidates/unary/link features and all parameters remain fixed. Each arm recomputes prototype/evidence/posterior/legal joint MAP interval and final trajectory DP in its own interval. CurrentAligned reproduces the audited current method bitwise. This intervention tests temporal correspondence of local object evidence conditional on this pipeline; it does not isolate action semantics, prove self-reinforcement causality or compare learned action recognition.\n\nShort means annotated event duration <=5s; long >5s. Bootstrap uses10000 paired parent resamples, seed20260920, pointwise95% intervals, not familywise. The panel was previously exposed; this is not unknown independent confirmation. No seed, parameters or sample counts were chosen from these scores. Null/negative findings and all signed metrics are retained. Fixed-support spatial metrics, boundary errors, recall, severe >5pp t/v union-once and rescues are fully reported in JSON/TSVs; a positive mean does not imply every-query improvement.\n\n## Permutation diagnostics\n\n'+json.dumps(report['permutation_diagnostics'],indent=2)+'\n\n## Tails: permutation relative to original alignment\n\n'+json.dumps({g:x['permutation_relative_to_aligned']for g,x in [('whole',report['whole']),*report['duration'].items()]},indent=2)+'\n\n## Actual costs\n\n'+json.dumps(costs,indent=2)+'\n'
    (p/'ANALYSIS.md').write_text(text)
    rec=dict(status='root_complete_all128_paired_signed_duration_permutation_tails_cost_analysis',dataset=ds,queries=128,parents=128,code_sha256=sha(Path(__file__)),plan_fingerprint=validate()['fingerprint'],audit_sha256=sha(f/'reporting_receipt.json'),score_sha256=sha(f/'score_receipt.json'),seal_sha256=sha(f/'sealed.json'),metrics_sha256=sha(f/'metrics.json'),report_sha256=sha(p/'report.json'),analysis_sha256=sha(p/'ANALYSIS.md'),root_worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_GT_scoring=0,new_model=0,new_Source=0,new_expert=0)
    atomic(f/'root_export_receipt_v1.json',rec);atomic(p/'root_export_receipt_v1.json',rec)
    print(json.dumps(dict(dataset=ds,paired=report['whole']['aligned_minus_permuted'],diagnostics=report['permutation_diagnostics'],tails=report['whole']['permutation_relative_to_aligned'])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['hc2','vidstg'],required=True);run(ap.parse_args().dataset)
