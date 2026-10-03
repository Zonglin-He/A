"""Report the prespecified paired intervention; never pick or promote a probe."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUB=ROOT/'results/tastvg_query_swap_specificity/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def num(v):return 'undefined' if v is None else f'{v:.4f}'
def interval(d):return 'undefined' if d['ci95'] is None else f'[{num(d["ci95"][0])}, {num(d["ci95"][1])}]'
ENDPOINTS=[('Event AUROC','frame/Hidden/event/real','auc'),
           ('Full precision R²','candidate/Full/precision/real','r2'),
           ('Full recall R²','candidate/Full/recall/real','r2'),
           ('Full tIoU R²','candidate/Full/tiou/real','r2')]
def run():
    cfg=read(PUB/'CONFIG.json');audit=read(PUB/'ROOT_AUDIT.json');assert audit['status']=='passed'
    res=read(PUB/'RESOURCES.json');s={ds:read(PUB/ds/'SUMMARY.json') for ds in ['vidstg','hc2']}
    text=['# TA-STVG Query-swap Specificity Review','',
        'This completed audit tests the fixed source-trained readouts under a same-video query intervention. '
        'It does not train a new quality model, select a new output, or change A. The predecessor is '
        '[the frozen temporal information atlas](TA_TEMPORAL_INFORMATION_ATLAS_REVIEW.md).','',
        '## Primary result: corrupted expert arrivals','',
        '| Dataset | Endpoint | Original query | Swapped query | Original − swapped | Paired 95% CI |',
        '|---|---|---:|---:|---:|---|']
    conclusions={}
    for ds,label in [('vidstg','VidSTG'),('hc2','HC-STVG-v2')]:
        z=s[ds]['target_corrupt'];conclusions[ds]={}
        for title,name,field in ENDPOINTS:
            a=z['metrics']['true/'+name]['metrics'][field]['mean'];b=z['metrics']['swap/'+name]['metrics'][field]['mean']
            d=z['paired_true_minus_swap'][name][field]
            text.append(f'| {label} | {title} | {num(a)} | {num(b)} | {num(d["mean"])} | {interval(d)} |')
            ci=d['ci95'];tag=('positive_gap' if ci and ci[0]>0 else 'negative_gap' if ci and ci[1]<0 else 'inconclusive')
            conclusions[ds][name]=dict(true=a,swap=b,paired=d,scope=tag)
    text+=['','The differences are in raw R²/AUROC units, **not vIoU percentage-point gains**. '
        'A positive gap with an interval above zero supports dependence of this frozen readout on '
        'the original query under this donor mapping. It does not prove exclusively semantic coding. '
        'Intervals that cross zero do not establish equivalence or absence of query-conditioned information.','',
        '**Measured primary outcome:** all eight corrupted-primary paired intervals cross zero. '
        'The Full precision/recall/tIoU point estimates decline for both datasets, but this panel '
        'does not establish a stable Full query-specificity gap. The existing quality information '
        'also does not disappear under swapped queries. These two statements must be retained together.','',
        '## Prespecified block supplement: a narrower positive finding','',
        'HC2 Contrast has lower absolute predictability than Full, but stronger query dependence '
        'under this intervention. Its three corrupted-panel paired intervals are above zero. '
        'These are supplementary, correlated endpoints with unadjusted pointwise intervals, '
        'not three independent replications or a new selected readout.','',
        '| HC2 Contrast endpoint | Original R² | Swap R² | Paired gap | 95% CI |',
        '|---|---:|---:|---:|---|']
    for task in ['precision','recall','tiou']:
        name=f'candidate/Contrast/{task}/real';z=s['hc2']['target_corrupt']
        a=z['metrics']['true/'+name]['metrics']['r2']['mean'];b=z['metrics']['swap/'+name]['metrics']['r2']['mean'];d=z['paired_true_minus_swap'][name]['r2']
        text.append(f'| {task} | {num(a)} | {num(b)} | {num(d["mean"])} | {interval(d)} |')
    ct=s['hc2']['target_corrupt_confirm']['paired_true_minus_swap']['candidate/Contrast/tiou/real']['r2']
    text+=['',f'The HC2 confirmation Contrast tIoU gap is {num(ct["mean"])} {interval(ct)} '
        '(seven sources); confirmation precision/recall gaps still cross zero. Clean Contrast '
        'recall/tIoU gaps are also positive. Vid’s real-label candidate block gaps all remain '
        'uncertain. The distinction is **absolute accessible quality versus query dependence**. '
        'It does not license a Contrast-based selector, a nonlinear quality model or a claim '
        'that all precise event semantics are established.','',
        '## What was held fixed','',
        '288 existing expert cells: 240 corruption and 48 clean. VidSTG has 144 cells / 24 independent '
        'sources (16 search, 8 confirm); HC2 has 144 / 21 (14, 7). Each original panel contained 32 '
        'search and 16 confirm sources, but only the existing sparse expert latent cells are evaluated. '
        'The two orders and source-dependent sparse schedule are retained. All data have historical '
        'development exposure; confirmation here means the predecessor panel, not a fresh test.','',
        'For each recipient video, a fixed caption from a different source/video in the same 48-source '
        'dataset pool replaces the query. SHA source sorting and the first globally valid cyclic '
        'offset define a one-to-one mapping before inference. Normalized captions differ; no GT or '
        'score chooses or filters donors. The donor’s existing subject parse is supplied through '
        'the original TA-STVG query interface. Thus this intervenes on the full query interface '
        '(caption and subject), not a token-only patch. Different provenance does not certify that '
        'a caption is false on the recipient video. No target-label hard-negative filtering is used.','',
        'Pixels, frame grid, both offsets, official same-domain checkpoints, pre-update A states, '
        'and Old8⊂Expanded32 endpoint indices remain fixed. Candidate features for the swapped '
        'query are pooled over the **original** 32 intervals. Neither the swapped query’s native '
        'interval nor a changed candidate pool enters the metric. A’s spatial predictions and '
        'persistent parameter trajectory are not rerun, adapted or promoted.','',
        'All 136 atlas models and their source-fitted normalization/alpha remain byte-identical. '
        'The prespecified focused set evaluates 40 models per dataset plus Null controls: Hidden/position '
        'event and six candidate views × precision/recall/tIoU × real/shuffled-fit labels. No source '
        'training GT is reread and there is no refitting, MLP, layerwise capture or expert call.','',
        '## Clean and confirmation controls','',
        '| Dataset / panel | Endpoint | Original | Swap | Paired gap | 95% CI |',
        '|---|---|---:|---:|---:|---|']
    for ds,label in [('vidstg','VidSTG'),('hc2','HC-STVG-v2')]:
        for group,gname in [('target_clean','clean'),('target_corrupt_confirm','corrupt confirm')]:
            z=s[ds][group]
            for title,name,field in ENDPOINTS:
                a=z['metrics']['true/'+name]['metrics'][field]['mean'];b=z['metrics']['swap/'+name]['metrics'][field]['mean'];d=z['paired_true_minus_swap'][name][field]
                text.append(f'| {label} / {gname} | {title} | {num(a)} | {num(b)} | {num(d["mean"])} | {interval(d)} |')
    text+=['','The full anonymous ROWS and SUMMARY include both fixed orders, search/confirm, '
        'all latent blocks, geometry/position, shuffled-fit and Null controls, R²/MSE/MAE, '
        'event AP/logloss, and within-cell R². Geometry, position and Null inputs are identical '
        'for the two queries, and their paired changes are exactly zero. They are invariance '
        'controls, not evidence that all latent signal is a shortcut.','',
        '## Validation and resources','',
        'Four no-GT clean smokes recomputed the original query with the full encoder and reproduced '
        'the cached temporal hidden, pooled candidate features, A boxes and interval bitwise. '
        'The observed hidden exactly reproduced the native start/end head. After root acceptance, '
        'the swapped-query encoder was recomputed on all fixed inputs; an encoder cache was reused '
        'only across identical recipient/query/pixels/frame-grid inputs. Each arrival replay used '
        'its original A pre-state.','',
        f'All {audit["changed_hidden_cells"]}/288 target cells changed their hidden under query swap. '
        'This verifies that the intervention reaches the representation; hidden change by itself '
        'does not establish localization-quality dependence.','',
        'Both datasets’ hidden/features and true/swap probe outputs were sealed before joining '
        'the original query’s already-exposed cached GT span. There is no raw annotation, donor-GT '
        'or new source-label read. The genuine original-query readouts and all focused per-cell '
        'metrics reproduce the preceding atlas bitwise.','',
        f'Root readback checked {audit["scalar_checks"]:,} scalars/features plus all input/state/probe '
        f'bindings (maximum numerical error {audit["maximum_numeric_error"]:.3g}). The independent '
        f'anonymous audit checked {audit["independent_public_audit"]["checks"]:,} items and recomputed '
        'all source aggregations and 10,000-draw paired intervals. Five synthetic CPU tests pass.','',
        '| Stage | Worker wall seconds | New encoder inputs | Backbone offset forwards |',
        '|---|---:|---:|---:|']
    for ds,label in [('vidstg','VidSTG'),('hc2','HC-STVG-v2')]:
        for stage,sname in [('SMOKE','smoke'),('CAPTURE_BARRIER','swap capture')]:
            r=res['workers'][ds][stage];text.append(f'| {label} {sname} | {r["worker_wall_seconds"]:.2f} | {r["new_encoder_inputs"]} | {r["new_backbone_offset_forwards"]} |')
    text+=['',f'CPU frozen readout: {res["CPU_readout_seconds"]:.2f}s; metric/interval generation: '
        f'{res["CPU_diagnosis_seconds"]:.2f}s. Total 278 encoder inputs and 556 two-offset backbone '
        'forwards, including eight true/swap smoke inputs; 288 production swap arrival replays. '
        'Zero new experts, training, backward calls, candidates or parameter updates. Worker wall '
        'time includes loading, decoding, hashing and I/O; it is not pure GPU kernel time.','',
        '## Interpretation and decision','',
        'This audit isolates readout dependence on a fixed wrong-source query while preserving '
        'labels/support/state. Performance can be retained by video dynamics, position, shared '
        'caption concepts and dataset priors; it can fall because of semantic dependence or '
        'off-distribution caption/subject effects. These alternatives are not fully separated '
        'by one donor per video. Source bootstrap is conditional on the fixed donor mapping.','',
        '**Correction to the incoming review:** within-cell R² measures absolute prediction '
        'accuracy and can be inflated negatively by low within-cell label variance. It does '
        'not directly measure candidate ranking. Neither pooled R² nor a true−swap gap proves '
        'safe top-1 choice, a deployable quality head, a realized tIoU/vIoU gain, or that a '
        'nonlinear model must succeed/fail. No probe is chosen on this result.','',
        'A and production CURRENT remain unchanged. This round ends with the audit and publication. '
        'No structured P/R quality head, layerwise/appearance-motion analysis, MLP, new TTA, '
        'external expert or paused full-query queue is automatically started.','',
        '## Figures','',
        '![Original and swapped candidate readouts](../results/tastvg_query_swap_specificity/2026-10-03/figures/candidate_true_swap.png)','',
        '![Paired query gaps](../results/tastvg_query_swap_specificity/2026-10-03/figures/paired_query_gap.png)','',
        '![Block-specific query gaps](../results/tastvg_query_swap_specificity/2026-10-03/figures/block_query_gaps.png)','',
        'Reproduction: [protocol](../protocols/tastvg_query_swap_specificity_v1.md), '
        '[execution](tastvg_query_swap_specificity_v1/EXECUTION.md), '
        '[full anonymous results](../results/tastvg_query_swap_specificity/2026-10-03).','']
    (ROOT/'docs/TA_QUERY_SWAP_SPECIFICITY_REVIEW.md').write_text('\n'.join(text))
    (PUB/'DECISION.json').write_text(json.dumps(dict(status='completed_audit_no_method_promotion',
        primary_endpoint_results=conclusions,A_unchanged=True,production_method_unchanged=True,
        supplementary_hc_contrast={task:dict(
            corrupt=s['hc2']['target_corrupt']['paired_true_minus_swap'][f'candidate/Contrast/{task}/real']['r2'],
            confirm=s['hc2']['target_corrupt_confirm']['paired_true_minus_swap'][f'candidate/Contrast/{task}/real']['r2'])
            for task in ['precision','recall','tiou']},
        supplementary_intervals_pointwise_unadjusted=True,selected_probe=None,
        automatic_next_experiment=False,query_specificity_scope='fixed donor mapping and frozen source-supervised readouts',
        evidence_not_absence_or_semantics_proof=True),indent=2,allow_nan=False)+'\n')
    print('QUERY_SWAP_REPORT_WRITTEN')
if __name__=='__main__':run()
