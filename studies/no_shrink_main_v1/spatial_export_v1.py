"""Root read-only interpretation/export of audited current-kappa0 structural controls."""
from ns_common import *
import argparse

def run(ds):
    import numpy as np
    validate();folder=P/'spatial'/ds;ap=D/f'spatial_{ds}_reporting_receipt.json';sp=D/f'spatial_{ds}_score_receipt.json'
    audit=read(ap);score=read(sp);seal=read(D/f'spatial_{ds}_sealed.json')
    assert audit['status']=='verified_whole_domain' and audit['score_sha256']==sha(sp)
    assert score['seal_sha256']==sha(D/f'spatial_{ds}_sealed.json') and seal['queries']==COUNTS[ds]
    rr=read(D/'spatial'/ds/'metrics.json')['private_rows'];report=read(folder/'report.json');interaction=read(folder/'interaction.json')
    for arm,mm in report['means'].items():
        for k,v in mm.items():
            values=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None]
            assert(v is None and not values)or abs(v-float(np.mean(values)))<1e-12
    tails={}
    for a,b in pairs('spatial'):
        dt=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr])
        tails[a+'-'+b]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()))
    atomic(folder/'all_paired_tails.json',tails)
    collector=PREV/'spatial'/f'{ds}_sealed.json';parent=read(collector);assert sha(collector)==seal['parent_collector_seal_sha256']
    costs=dict(current_cached_query_CPU_segments=audit['worker_CPU_seconds'],parent_collector=dict(seal_sha256=sha(collector),worker=parent['worker'],actual_counts=parent['counts'],whole_worker_seconds=parent['seconds'],model_load_seconds=parent['model_load_seconds'],Source_forward_seconds=sum(e['Source_seconds']for e in parent['files'].values()),query_pipeline_seconds=sum(e['query_seconds']for e in parent['files'].values())),inherited_original_Source_and_eight_teacher_cost_nonzero=True,new_expert=0,scope='Original raw-grid acquisition and cached CPU replay reported separately. Parent worker/query/Source/model-load intervals may be nested; do not add them or call replay deployment latency. Shared old raw-grid acquisitions retain original cost provenance.')
    atomic(folder/'cost_segments.json',costs)
    with(folder/'structural_table.tsv').open('w')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['configuration','tIoU_percent','vIoU_percent','R03_percent','R05_percent']);w.writerows([[a,*[100*m[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]]for a,m in report['means'].items()])
    def fmt(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:.6f}, {x['CI95'][1]:.6f}]"
    text=f'# Current no-shrinkage structural controls: {ds}\n\nAll {len(rr)} queries / {report["parents"]} parents sealed before independent official labels; {score["checks"]} official dense checks (max error {score["max_error"]:.17g}), {audit["prediction_receipt_SHA_checks"]} new prediction/receipt SHA checks plus all bound collector/capture/compact/teacher provenance. Every array cold rederived exactly and all scalar means checked. This export reads already-scored scalars only: zero model, optimizer, Source, expert or GT-scoring calls.\n\nCurrent kappa0 was selected by direct user after full-result exposure. Other method parameters remain frozen; these historically exposed full sets are not new independent confirmation. Native Source remains per-offset MAP plus physical hull, and SourceJointMAP isolates readout.\n\n|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Native-relative severe t/v/union|\n|---|---:|---:|---:|---:|---|\n'
    for arm,m in report['means'].items():
        z=report['severe_vs_Source'][arm];text+=f"|{arm}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{z['t']}/{z['v']}/{z['union']}|\n"
    text+='\n## Paired effects and tails\n\nPositive means first configuration exceeds second. All intervals are pointwise parent-cluster bootstrap CIs, not familywise. Severe is a loss greater than5pp in tIoU or vIoU; the union counts each query once. Rescue is a gain greater than5pp in either metric.\n\n|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|\n|---|---:|---:|---|---:|\n'
    for key,p in report['paired'].items():
        z=tails[key];text+=f"|{key}|{fmt(p['t'])}|{fmt(p['v'])}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
    text+='\n## Duration groups\n\nShort is native-GT duration <=5s; long is >5s.\n\n|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---:|---:|\n'
    for group,g in report['timegroups'].items():
        for key,p in g['paired'].items():text+=f"|{group} / {key}|{g['queries']}|{fmt(p['t'])}|{fmt(p['v'])}|\n"
    text+='\n## Controls and interpretation\n\nNoSharedCorrection sets E=Source and rebuilds all affected geometry, auxiliary trajectory, actual raw-grid ROI pooling, posterior, MAP interval and final DP. PastOnly removes future anchors; OneWayAffinity replaces the original bidirectional complete-box matching by the original forward term, keeping the other settings fixed. Their effects are assessed against Full in the paired table, not assumed beneficial.\n\nAuxIndependent fixes the auxiliary bank/validity/unary and removes only auxiliary path links, with every downstream step recomputed. AuxSourceROI and AuxE_ROI use Source or E directly as auxiliary ROI boxes. Their results do not establish an independent first-auxiliary-DP benefit when CIs cross0. This is distinct from the already completed final-DP comparison.\n\nNoSpatialTrajectory removes propagation candidates and both path DPs, using E directly for boxes and ROI; it is not interpolation, greedy replacement or expert overwrite. NoTemporalInference removes visual posterior updating and restores native per-offset MAP/hull, then resolves the retained spatial DP on that interval. Its comparison also changes temporal readout, so it is not solely a posterior-evidence attribution.\n\nThe E-fixed factorial uses S0T0=E/q0jointMAP, S1T0=spatial/q0jointMAP, S0T1=E/updated q1 from E ROIs, S1T1=full spatial trajectory/updated q1. Every changed interval resolves the relevant final DP. Interaction is (S11-S10)-(S01-S00), rather than a sum of progressive gains.\n\n'
    text+=f"Interaction delta t {fmt(interaction['paired']['t'])} pp; delta v {fmt(interaction['paired']['v'])} pp. A negative interaction rules out a positive superadditive claim under this defined control; it does not negate the separate positive component effects.\n\n"
    text+='## Costs and reproducibility\n\n'+json.dumps(costs,indent=2)+'\n\nAll observed/unobserved fixed-support metrics, boundary errors, signed positives and negatives, severe/rescue counts and full duration CIs remain in report.json, all_paired_tails.json and both signed TSVs. Media, captions, coordinates, raw features and weights stay private.\n'
    (folder/'ANALYSIS.md').write_text(text)
    receipt=dict(status='root_complete_structural_analysis',dataset=ds,code_sha256=sha(Path(__file__)),audit_sha256=sha(ap),score_sha256=sha(sp),report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),all15arms=True,all_paired_tails_and_timegroups=True,interaction_preserved=True,new_model=0,new_GT_scoring=0)
    atomic(D/f'spatial_{ds}_root_export_receipt.json',receipt);atomic(folder/'root_export_receipt.json',receipt);print(ds,'root full structural analysis exported',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);run(p.parse_args().dataset)
