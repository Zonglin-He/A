"""Report generator: actual sealed matched comparison, cases and limitations."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_anchor_quality/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def ci(m,scale=100,digits=4):
    a,b=m['ci95'];return f"{m['mean']*scale:+.{digits}f} [{a*scale:+.{digits}f}, {b*scale:+.{digits}f}]"
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
summ={ds:{sp:read(OUT/sp/ds/'SUMMARY.json') for sp in ['search','confirm']} for ds in ['vidstg','hc2']}
loso={ds:read(OUT/ds/'LOSO_SUMMARY.json') for ds in ['vidstg','hc2']}
baseline=read(OUT/'BASELINE_DIAGNOSIS.json');decision=read(OUT/'DECISION.json');audit=read(OUT/'ROOT_AUDIT.json')
labels=dict(vidstg='VidSTG',hc2='HC-STVG-v2');parts=[]
parts.append('''# Anchor-score conditional delta: a matched CPU review

**The locked M1 feature does not establish useful joint confirmation behavior.** HC source LOSO error increases and HC target choices are identical to M0/L32. Vid confirmation rejects just one additional severe harmful correction; that gain is confined to one source and the complete-flow CI still crosses zero. Keep A; stop this scalar recipe. No structured model or historical GPU queue is started.

This experiment checks the user's proposed anchor-score factor with matched linear controls. It does not prove that true anchor quality is unimportant, that scalar evidence never works, or that source-scale/native-anchor explanations have been causally eliminated.

## Fixed data and execution

Official same-domain TA-STVG checkpoints Vid fbb1ed88 / HC ee72f0d9, original two-offset pixels, all 32 hypotheses including Old8, frozen L scores, Uniform expert arrival schedule and A's 1792-parameter persistent spatial trajectory (Vid K1 / HC K8) are unchanged. Each dataset uses 32 search + 16 confirmation sources, one query/source, two orders, clean and five 5% corruptions, 25% expert positions. All 1152 arrivals are evaluated: 288 expert (240 corrupt/48 clean), 864 nonexpert identically A. These are historically exposed development panels, not fresh tests.

The 31/16 official source-validation sources already selected the original ridge alpha and earlier calibrations. Only these source labels fit the new delta predictors. The original ranking and alpha are not retrained. Baseline GT case diagnosis precedes fitting and is explicitly recorded; separate guarded fit/selection stages cannot parse target metric rows or that diagnostic. Input bytes are read solely for SHA256 verification before each guard is installed; seal flags concerning label reads mean no parsing of target label values into fitting or selection, not absence of hash-only file I/O. Both source models and all new choices seal before the new-arm target metric join. This controls this run's decisions, not earlier GT exposure.

## Good/failure cases and the measured gap

The attachment's GT-quality quartiles reproduce. These are **cell means on corruption expert arrivals**, including no-op cells, and cannot be used as an online gate. Source-balanced estimates and intervals are also released. Improve/harm here means tIoU increase/decrease.
''')
qr=[]
for ds in labels:
    for q in baseline['panels'][ds+'/confirm']['quartiles']:
        z=q['arms']['L32'];qr.append([labels[ds],q['quartile'],f"{q['cell_mean_A8_t']:.4f}",f"{z['cell_means']['dt']*100:+.4f}",f"{z['cell_means']['dv']*100:+.4f}",f"{z['counts']['benefit_t']} / {z['harmful_t']} / {z['neutral_t']}"])
parts.append(table(['Confirm expert','GT quartile','Mean A8 tIoU','L32 ΔtIoU pp','L32 ΔvIoU pp','Improve / harm / neutral'],qr))
bad=next(r for r in audit['cited_case_readback'] if r['source_id']==37)
good=next(r for r in baseline['panels']['vidstg/confirm']['examples']['cited_source34'] if r['condition']=='frame_drop_5')
parts.append(f"""
The cited Vid source37/exposure/order2 case is verified: A8 tIoU {bad['A8_t']:.6f} → L32 {bad['arms']['L32']['t']:.6f}, radius {bad['top_geometry']['radius']:.6f}, vIoU {bad['arms']['L32']['dv']*100:+.4f} pp. Prior Pair-Norm accepts it. **M1 also accepts it**, predicting +{bad['evidence']['M1']['predicted_delta_t']:.6f} tIoU. The single newly rejected Vid confirmation case is source37/**frame_drop**, not this exposure example.

The helpful source34/frame_drop/order1 case is also verified: {good['A8_t']:.6f} → {good['arms']['L32']['t']:.6f}, vIoU {good['arms']['L32']['dv']*100:+.4f} pp. Both are small corrections. Radius alone therefore does not separate these two cases, but two cases do not identify a universal cause. The baseline convenience named-case filter expected `exposure` while the real condition is `exposure_5`; full rows/quartiles were intact and independent case readback uses the real condition.

Two limits matter before fitting. First, Δt=t_winner−t_anchor contains minus t_anchor. The HC confirmation correlation is {baseline['panels']['hc2/confirm']['correlations']['A8_t_vs_L32_dt']:.4f}; the exact average-over-candidates delta has an even stronger negative correlation ({baseline['panels']['hc2/confirm']['correlations']['A8_t_vs_mean_candidate_delta']:.4f}). The latter is a post-hoc arithmetic control, not a new deployed scorer. These correlations do not prove anchor-quality blindness.

Second, the proposed proxy is a relative **L score**, not GT correctness. Its cell-level correlation with A8 GT tIoU on confirmation is {baseline['panels']['vidstg/confirm']['correlations']['anchor_score_vs_A8_t']:.4f} Vid and {baseline['panels']['hc2/confirm']['correlations']['anchor_score_vs_A8_t']:.4f} HC. These descriptive associations, with repeated cells and few sources, are not calibration guarantees.

## Matched single-feature test

Each source's unique highest L candidate is compared with all 32 pseudo-anchors, including its self pair with delta=0. Retain negative/neutral labels and duplicate intervals; total fit weight one/source. This produces 992 Vid and 512 HC pseudo-anchor examples, **31/16 independent sources**, not 1504 independent observations.

With fixed IQR+1e−8, m=(s_top−s_anchor)/denominator and a=(s_anchor−median(s))/denominator:

    M0: predicted delta = beta0 + beta1*m
    M1: predicted delta = beta0 + beta1*m + beta2*a

Both use weighted unregularized SVD least squares with rcond=1e−12. No interaction, MLP, threshold/q/regularization search, extra latent or expert. Both target rules accept the unchanged unique top strictly above A8 only if predicted mean delta>1e−12. This point-mean rule is **not** a lower safety certificate. Marginal source-range extrapolation is recorded, never another gate. M0 differs from the previous isotonic Pair-Norm; that previous arm is descriptive.

Exactly m+a=c_pool=(s_top−median(s))/denominator. Therefore M1 is equivalently beta0+(beta1−beta2)m+beta2*c_pool. It adds across-pool top-score context, **not independent within-pool anchor evidence**. Full design ranks are 2/3 for M0/M1 in both datasets; rank sufficiency across sources does not remove that identity or identify a causal anchor mechanism.

## Source leave-one-source-out

Every source is predicted from a fit to all other validation sources. All 32 pseudo-anchors have equal within-source weight; the eligible non-self analysis is also released. MSE uses 0–1 tIoU units squared. Positive error reduction favors M1. Bootstrap resamples 10000 whole sources, not pairs.
""")
lr=[]
for ds in labels:
    m=loso[ds]['all_pseudo_anchors']['metrics'];lr.append([labels[ds],f"{m['M0_MSE']['mean']:.6f}",f"{m['M1_MSE']['mean']:.6f}",ci(m['MSE_improvement'],1,6),f"{loso[ds]['all_pseudo_anchors']['positive_sources']}/{loso[ds]['source_counts']}"])
parts.append(table(['Dataset','M0 MSE','M1 MSE','M0−M1 MSE [95% CI]','Improved source folds'],lr))
parts.append('Vid improvement is uncertain; HC M1 has higher error and its paired interval is below zero. All folds, coefficients, rank/condition diagnostics and predictions are exported; no fold outcome selects a different rule. The top pseudo-anchor self pair is excluded from actual replacement but retained as explicitly specified in training.')
rr=[]
for ds in labels:
    for sp in ['search','confirm']:
        s=summ[ds][sp]['corruption'];rr.append([labels[ds]+'/'+sp]+[ci(s['all'][a]['metrics']['dv']) for a in ['L32','Pair-Norm','M0','M1']]+[ci(s['paired']['metrics']['M1_vs_M0_v'])])
parts.append('## Complete corruption flow\n\nAll differences below are source-macro vIoU pp against unchanged A8. Source→order→condition averages and paired 10000-source bootstrap are used; per-order and leave-one-source-out values are released.\n\n'+table(['Panel','L32−A8','Prior Pair-Norm−A8','M0−A8','M1−A8','M1−M0'],rr))
rows=[]
for ds in labels:
    for a in ['L32','Pair-Norm','M0','M1']:
        s=summ[ds]['confirm']['corruption']['all'][a];m=s['metrics'];c=s['raw_counts']
        rows.append([labels[ds],a,f"{c['accepted']}/40",s['accepting_sources'],f"{c['benefit_t']}/{c['accepted']-c['benefit_t']}",f"{m['gross_gain']['mean']*100:.4f}/{m['gross_loss']['mean']*100:.4f}",c['severe']])
parts.append('## Confirmed coverage, gross utility and negative tail\n\nAcceptance counts use the 40 corrupt expert cells/dataset; gross gain/loss use the complete corrupt flow. Severe means ΔvIoU<−5 pp.\n\n'+table(['Dataset','Arm','Accepted','Accepting sources','t-benefit / other accepted','Gross gain/loss pp','Severe harms'],rows))
parts.append('Vid M0 accepts every L32 replacement. M1 changes one confirmation decision, lowers severe harms 6→5 and leaves all 27 helpful proposals accepted. M1−M0 +0.0543 pp comes entirely from source37; leaving it out gives zero. The overall M1−A8 interval crosses zero. HC M1 accepts exactly the same 34 proposals as M0/L32, including all 25 harmful proposals and 13 severe harms. This is not near-zero coverage safety; it is absent rejection utility.')
strrows=[]
for ds in labels:
    qs=read(OUT/'confirm'/ds/'ANCHOR_STRATA.json')['quartiles']
    for qi in [0,3]:
        q=qs[qi]
        for a in ['M0','M1']:
            z=q['proposal_retention'][a];strrows.append([labels[ds],f"Q{qi+1}",a,f"{z['helpful_accepted']}/{z['helpful_total']}",f"{z['harmful_rejected']}/{z['harmful_total']}"])
parts.append('## Does the conditional mechanism appear?\n\nGT strata are diagnostic only. Zero denominators mean no proposals of that kind, not a safety rate.\n\n'+table(['Dataset','GT anchor quartile','Arm','Helpful proposals kept','Harmful proposals rejected'],strrows))
parts.append('M1 keeps the lowest-quartile helpful proposals on both datasets, but it rejects only 1/6 top-quartile harmful proposals on Vid and 0/7 on HC. It has not learned the claimed pattern of selectively protecting reliable anchors. The frozen relative score proxy is insufficient in this locked fit; failure does not refute a mechanism using an independent, valid anchor-quality measurement.')
clean=[]
for ds in labels:
    for sp in ['search','confirm']:clean.append([labels[ds]+'/'+sp]+[ci(summ[ds][sp]['clean']['all'][a]['metrics']['dv']) for a in ['M0','M1']])
parts.append('## Clean control and scope\n\n'+table(['Clean complete flow','M0−A8 pp','M1−A8 pp'],clean)+'\n\nAll 864 nonexpert cells remain A. These arms change current expert temporal readout only, not future adaptation or persistent spatial learning. Clean, search and confirmation are not stitched into dataset-specific winners.')
fails=[]
for ds in labels:fails.append(labels[ds]+': '+', '.join(k for k,v in decision['datasets'][ds]['tests'].items() if not v))
parts.append('## Decision and verification\n\n**'+decision['status']+'**. Failed prelocked development checks:\n\n'+ '\n'.join('- '+x for x in fails))
parts.append(f"""
Keep A/CURRENT. Stop more variants of this margin/relative-anchor-score certification recipe. A structured candidate-relative representation is a possible future variable; no concrete new model is trained here. Scale shift, selected-top conditioning and source→target mismatch remain possible; two normalized-score failures do not causally eliminate them. The one added feature comparison changes neither source pairs nor fit/rule, but M0 vs prior Pair-Norm changes the pair population and regressor and cannot be read as a one-factor causal comparison.

Independent SciPy pivoted QR reconstructs both full fits and every LOSO fold against NumPy SVD. Every input hash, choice/state/pixel hash, cached metric, quartile, paired/source/order/bootstrap summary and decision check passes: **{audit['scalar_checks']:,} scalar checks**, maximum difference {audit['maximum_numeric_error']:.3g}. Seven meaningful CPU controls pass. Baseline case audit {read(OUT/'GT_EXPOSURE.json')['CPU_wall_seconds']:.3f}s, source fitting/LOSO {read(OUT/'CALIBRATION_SEAL.json')['CPU_wall_seconds']:.3f}s, new choice sealing {read(OUT/'GLOBAL_DECISION_SEAL.json')['CPU_wall_seconds']:.3f}s, metric aggregation {read(OUT/'RESOURCES.json')['CPU_evaluation_wall_seconds']:.3f}s, independent audit {audit['CPU_wall_seconds']:.3f}s. These are CPU wall times excluding rendering/publication, not GPU timing. New GPU forwards, expert calls, replays, backwards and updates are all zero.

## Reproduction and evidence

- [Protocol](../protocols/tastvg_anchor_quality_v1.md), [execution](tastvg_anchor_quality_v1/EXECUTION.md), [actual configuration](../results/tastvg_anchor_quality/2026-10-03/CONFIG.json).
- [All source/target scores, labels, fits, LOSO, seals, positive/negative cases and figures](../results/tastvg_anchor_quality/2026-10-03), [decision](../results/tastvg_anchor_quality/2026-10-03/DECISION.json), [independent audit](../results/tastvg_anchor_quality/2026-10-03/ROOT_AUDIT.json).
- [Runner](../scripts/run_tastvg_anchor_quality_v1.py), [math](../scripts/tastvg_anchor_quality_math_v1.py), [tests](../scripts/test_tastvg_anchor_quality_v1.py), [auditor](../scripts/audit_tastvg_anchor_quality_v1.py), [drawing](../scripts/draw_tastvg_anchor_quality_v1.py), [report generator](../scripts/report_tastvg_anchor_quality_v1.py).

```bash
python -B -m unittest scripts.test_tastvg_anchor_quality_v1
python -B scripts/audit_tastvg_anchor_quality_v1.py
```

The public audit uses anonymous saved scores and scalar labels. Preparation also verifies private predecessor/current receipts; it is not a public model inference launcher. No media, weights, raw annotations, hidden caches or personal conversations are released.

![Confirmation readout](../results/tastvg_anchor_quality/2026-10-03/figures/confirmation_readout.png)

![GT anchor strata](../results/tastvg_anchor_quality/2026-10-03/figures/GT_anchor_strata.png)

![Source LOSO](../results/tastvg_anchor_quality/2026-10-03/figures/source_LOSO.png)
""")
path=ROOT/'docs/TA_ANCHOR_QUALITY_REVIEW.md';path.write_text('\n\n'.join(parts).rstrip()+'\n')
print(path)
