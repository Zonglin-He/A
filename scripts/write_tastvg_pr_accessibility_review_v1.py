"""Write a reproducible review from audited anonymous scalar outputs."""
from pathlib import Path
import json
import math
import csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/tastvg_pr_accessibility/2026-10-03'
FITS = ['source_fit', 'target_search_fit']
ROLES = ['P_A', 'R_A', 'P_W', 'R_W']
ARMS = ['predicted', 'GT_anchor', 'GT_winner', 'GT_precision', 'GT_recall', 'GT_all']
NAMES = {'vidstg': 'VidSTG', 'hc2': 'HC-STVG-v2'}


def read(path):
    return json.loads(path.read_text())


def num(x, scale=1):
    return 'undefined' if x is None or not math.isfinite(x) else f'{x*scale:.4f}'


def packed(v, scale=1):
    if v['mean'] is None:
        return 'undefined'
    bounds = v['ci95']
    return num(v['mean'], scale) + (f" [{num(bounds[0],scale)}, {num(bounds[1],scale)}]" if bounds else '')


def table(lines, headers, rows):
    lines += ['', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]
    lines.append('')


def main():
    audit = read(OUT / 'ROOT_AUDIT.json')
    assert audit['status'] == 'pass'
    config = read(OUT / 'CONFIG.json')
    data = {ds: read(OUT / ds / 'SUMMARY.json') for ds in NAMES}
    rows = read(OUT / 'ROWS.json')
    resources = read(OUT / 'RESOURCES.json')
    lines = ['# Target-domain P/R readout accessibility audit', '',
        '**The target-search refit improves in-sample P/R fitting, but does not establish held-out recovery of the four A/W readouts. HC retains a narrower positive result: the fixed-zero decision rejects more harmful replacements.**', '',
        'This finite CPU audit is complete. It uses target-search GT for four supervised diagnostic ridge fits, then evaluates source-disjoint confirmation features and labels. It is not unsupervised test-time adaptation, a new gate, a new temporal selector or a method promotion. A, L32 winners, checkpoint states, input pixels and production CURRENT_METHOD remain unchanged.', '',
        'The two tests requested after [f5e3f2d](https://github.com/Zonglin-He/A/blob/f5e3f2d4758f50379ac94baa2b3e16c293d652dd/docs/TA_STRUCTURED_SEPARABILITY_REVIEW.md) are both implemented: matched source-fit versus target-search-fit P/R regression, followed by predicted/oracle P/R composition on fixed A8 and L32 winner.', '',
        '## Actual setting and data use', '',
        'Both original panels contain 32 search +16 confirmation sources/dataset, one query/source, two fixed orders, clean +five 5% transient corruptions, with the existing 25% expert schedule. The regression audit uses only expert arrivals for which the cached feature interface exists; the other 864 arrivals are not added. There are 288 expert cells total (240 corrupt +48 clean), 30 independent expert training sources and 15 expert confirmation sources across the two datasets. All target sources have project-level historical exposure; source-disjoint confirmation is not a fresh test.']
    table(lines, ['Dataset', 'Search expert sources / cells', 'Confirm expert sources / cells', 'Training candidate rows', 'Fixed alpha P / R', 'Original source train / alpha-validation sources'],
          [[NAMES[ds], f"{config['actual_expert_sources'][ds]['search']} /96", f"{config['actual_expert_sources'][ds]['confirm']} /48", '3072',
            f"{config['source_fit_alpha'][ds]['P']:g} / {config['source_fit_alpha'][ds]['R']:g}",
            f"{config['source_probe_train_sources'][ds]} / {config['source_alpha_validation_sources'][ds]}"] for ds in NAMES])
    lines += ['The frozen TA-STVG checkpoints are Vid fbb1ed88 and HC ee72f0d9 (full SHA256 in CONFIG). A retains Uniform Rank-RKL on 1792 spatial parameters, Vid K1 / HC K8 and the original Paper48 two-offset sampling; it does not inherit Fig1 uniform64 sampling.', '',
        'Each fit uses all 32 search-cell candidates, including duplicates and ties: cached final sixth temporal decoder hidden, Inside[512:768] for P and Endpoint[0:512] for R. The old source-selected ridge penalties stay fixed. The equal-source normalization *procedure* is identical, while search mean/std and unpenalized intercept are re-estimated from target-search only. No confirmation features or labels enter fitting. Original source models are not refitted.', '',
        'Fit weights are sealed before GT-free readout of both panels. All scalar predictions and fixed A/W indices are sealed before confirmation labels are joined. This verifies the present execution order, not first-ever blindness to previously exposed confirmation data. Input hashes, guarded access, model/readout/label seals and source prediction parity passed root audit.', '',
        'Define P as event-intersection / candidate length and R as event-intersection / GT event length. Raw ridge predictions are used for regression; only the diagnostic analytic composition clips them to [0,1]:', '',
        r'\[F(P,R)=\frac{PR}{P+R-PR},\qquad F(0,0)=0.\]', '',
        'W is the previously frozen L32 winner, not a newly selected candidate. The offline replacement is accepted iff W differs from A and predicted F(W)−F(A)>0. This analytic acceptance is a new diagnostic control even for the old source-fit weights; it is not the predecessor L32/A arbitration rule. True helpful/harmful labels use cached ΔtIoU >±1e−12; no-ops and neutral cases remain reported but are excluded from binary discrimination.', '',
        '## 1. Regression: fitting the search panel is possible; confirmation does not recover', '',
        'The following R² values use the corruption search rows, which were included in target-supervised fitting. These are in-sample diagnostics, not evidence of generalization.']
    table(lines, ['Dataset', 'Readout', 'Source-fit R²', 'Target-search-fit R²', 'Source-fit MAE', 'Target-search-fit MAE'],
          [[NAMES[ds], role] + [num(data[ds]['search_corrupt']['regression'][fit][role]['metrics'][metric]['mean'])
           for metric in ['r2', 'mae'] for fit in FITS] for ds in NAMES for role in ROLES])
    lines += ['On confirmation corruption, all four MAE means increase in each dataset. The paired intervals below do not establish a broad MAE deterioration or improvement. Vid winner-recall R² deterioration does have a below-zero paired interval. Negative R² means worse squared error than the same weighted GT mean baseline; it does not imply the hidden representation contains no temporal information.']
    table(lines, ['Dataset', 'Readout', 'Source R² / MAE / rho', 'Target-search R² / MAE / rho'],
          [[NAMES[ds], role] + [' / '.join(num(data[ds]['confirm_corrupt']['regression'][fit][role]['metrics'][k]['mean'])
           for k in ['r2', 'mae', 'rho']) for fit in FITS] for ds in NAMES for role in ROLES])
    table(lines, ['Dataset', 'Readout', 'Paired ΔMAE [95% CI]', 'Paired ΔR² [95% CI]', 'Paired Δrho [95% CI]'],
          [[NAMES[ds], role] + [packed(data[ds]['confirm_corrupt']['paired_target_minus_source_regression'][role][k])
           for k in ['mae', 'r2', 'rho']] for ds in NAMES for role in ROLES])
    lines += ['The all-32 pooled metrics are retained to show why average candidate recoverability cannot substitute for quality of the particular A/W comparison. For example, HC source-fit pooled recall R² is high while the A/W recall roles have negative R²; target refitting improves search roles without repairing the confirmation roles.']
    table(lines, ['Dataset', 'All-32 variable', 'Source R² / MAE / rho', 'Target-search R² / MAE / rho'],
          [[NAMES[ds], role] + [' / '.join(num(data[ds]['confirm_corrupt']['regression'][fit][role]['metrics'][k]['mean'])
           for k in ['r2', 'mae', 'rho']) for fit in FITS] for ds in NAMES for role in ['P_pool', 'R_pool']])
    lines += ['Confirmation corruption clipping fractions are small but nonzero: Vid target P_W exceeds1 on 12.5% of source-weighted rows; HC source R_A and R_W exceed1 on 14.2857%, versus target R_A 2.8571% and R_W 0%. All four lower-than-zero fractions are0 in this panel. Clipping is an analytic-map requirement, not a reason to hide raw regression errors.', '',
        '![Matched search and confirmation MAE](../results/tastvg_pr_accessibility/2026-10-03/figures/pr_fit_generalization.png)', '',
        '*Bars: raw source-balanced MAE; error bars: whole-source 95% bootstrap intervals. Search bars include training data; confirmation has disjoint sources but historical exposure.*', '',
        '![Paired confirmation MAE changes](../results/tastvg_pr_accessibility/2026-10-03/figures/pr_paired_confirmation_mae.png)', '',
        '*These intervals bootstrap paired changes jointly, rather than subtracting two independent confidence intervals.*', '',
        '## 2. Analytic decision: retain the HC local positive result, without calling it recovery', '',
        'AUROC uses oriented predicted ΔF without sign flipping. Balanced accuracy uses the prelocked strict-zero decision. Helpful and harmful classes each give equal total weight to an informative source. Severe harm here means accepted cached ΔvIoU<−.05. Counts are cell counts; means/intervals are source-balanced.']
    table(lines, ['Dataset', 'Fit', 'AUROC [95% CI]', 'Balanced accuracy [95% CI]', 'Accepted helpful / harmful', 'Accepted severe v-harm'],
          [[NAMES[ds], fit, packed(d['metrics']['auc']), packed(d['metrics']['balanced_accuracy']),
            f"{d['counts']['accepted_helpful']} / {d['counts']['accepted_harmful']}", d['counts']['accepted_severe_v_harm']]
           for ds in NAMES for fit in FITS for d in [data[ds]['confirm_corrupt']['decision'][fit]['predicted']]])
    table(lines, ['Dataset', 'Paired ΔAUROC [95% CI]', 'Paired Δbalanced accuracy [95% CI]', 'Paired ΔFPR [95% CI]'],
          [[NAMES[ds]] + [packed(data[ds]['confirm_corrupt']['paired_target_minus_source_decision']['predicted'][k])
           for k in ['auc', 'balanced_accuracy', 'fpr']] for ds in NAMES])
    lines += ['HC therefore has a measured local balanced-accuracy improvement (paired CI above0) and fewer false-positive replacements. The helpful accepts remain9 while harmful accepts fall21→11. All seven leave-one-source-out balanced-accuracy changes remain positive (+.0700 to+.1700), but these are influence calculations, not new refits or independent tests. AUROC improvement remains uncertain, and10 severe v-harms remain. This supports partial relative-decision repair in this exposed small panel; it does not establish accurate absolute P/R regression, safe deployment or an unsupervised method gain.', '',
        'Vid keeps7 harmful accepts and loses6 helpful accepts (25→19). Its balanced-accuracy change is negative with an interval crossing0. The two datasets do not establish a common confirmation improvement.', '',
        'The offline accepted-decision utility below reuses cached A/W metrics on the expert subset only. These are simulated readout deltas relative to A, not full-flow inference results or new persistent-state gains.']
    table(lines, ['Dataset', 'Fit', 'Expert-subset ΔvIoU pp [95% CI]', 'Expert-subset ΔtIoU pp [95% CI]'],
          [[NAMES[ds], fit, packed(d['utility']['delta_v'],100), packed(d['utility']['delta_t'],100)]
           for ds in NAMES for fit in FITS for d in [data[ds]['confirm_corrupt']['decision'][fit]['predicted']]])
    lines += ['## 3. Oracle component ladder: errors must be compared on a coherent relative scale', '',
        'Each ladder arm keeps W fixed and replaces only the listed scalar components by GT. GT_anchor replaces both A components; GT_winner both W components; GT_precision both P components; GT_recall both R components. The same six arms are evaluated for both fits. Partial P/R mixtures need not correspond to a realizable interval and are not deployable candidate quality scores.']
    table(lines, ['Dataset', 'Fit', 'Replacement', 'AUROC', 'Balanced accuracy', 'Accepted helpful / harmful / severe'],
          [[NAMES[ds], fit, arm, num(d['metrics']['auc']['mean']), num(d['metrics']['balanced_accuracy']['mean']),
            f"{d['counts']['accepted_helpful']} / {d['counts']['accepted_harmful']} / {d['counts']['accepted_severe_v_harm']}"]
           for ds in NAMES for fit in FITS for arm in ARMS for d in [data[ds]['confirm_corrupt']['decision'][fit][arm]]])
    lines += ['The ladder is not monotone in “how much GT” it receives. Correcting the anchor alone removes many bad accepts, but can reject useful candidates whose remaining winner estimate is too low. Correcting only the winner can instead accept a worse interval because the anchor remains underestimated. Thus it is not justified to uniquely blame either anchor or winner estimation from one component replacement.', '',
        'Replacing both P values or both R values improves confirmation HC AUROC relative to all-predicted, for both fits; the paired intervals are below. This isolates a useful component intervention, while leaving the other predicted component imperfect. It is not proof that a learned replacement can reproduce the oracle benefit.']
    table(lines, ['Dataset / fit', 'Oracle intervention', 'Paired AUROC change [95% CI]', 'Paired balanced-accuracy change [95% CI]'],
          [[NAMES[ds] + ' / ' + fit, arm, packed(v['auc']), packed(v['balanced_accuracy'])]
           for ds in NAMES for fit in FITS for arm in ['GT_anchor', 'GT_winner', 'GT_precision', 'GT_recall']
           for v in [data[ds]['confirm_corrupt']['ladder_minus_predicted'][fit][arm]]])
    lines += ['HC target-fit GT_recall reaches AUROC1.0000 while still accepting two harmful replacements (balanced accuracy .8958). Perfect ordering within this small panel is not correct absolute zero-threshold calibration. Full GT restores AUROC1 / balanced accuracy1 and no harmful primary-binary accepts in every panel: an algebra/label positive control, not new evidence of predictive power.', '',
        '![Analytic P/R component replacement ladder](../results/tastvg_pr_accessibility/2026-10-03/figures/pr_oracle_ladder.png)', '',
        '*Confirmation corruption; whole-source 95% intervals. Ranking AUROC and strict-zero balanced accuracy are deliberately separated. Undefined resamples remain counted in SUMMARY.*', '',
        '## 4. Concrete preserved work/failure cases', '',
        'The first three rows below revisit fixed old cases; the final two are post-hoc most harmful target-fit accepted cases/dataset from the same sealed confirmation rows. They are diagnostic examples, not GT-selected training data or an online admission rule.']
    selected = []
    for key in ['vidstg/confirm/exposure_5/order2/12', 'vidstg/confirm/frame_drop_5/order1/4', 'hc2/confirm/frame_drop_5/order1/12']:
        selected.append(next(r for r in rows if r['cell_key'] == key))
    for ds in NAMES:
        eligible = [r for r in rows if r['dataset'] == ds and r['split'] == 'confirm' and r['condition'] != 'clean'
                    and r['eligible'] and r['ladder']['target_search_fit']['predicted']['delta_T'] > 0]
        selected.append(min(eligible, key=lambda r:r['delta_v']))
    table(lines, ['Cell / source', 'True Δt / Δv pp', 'Source ΔF', 'Target ΔF', 'Target GT_anchor ΔF', 'Target GT_winner ΔF'],
          [[r['cell_key'] + f" / source{r['source_id']}", f"{num(r['delta_t'],100)} / {num(r['delta_v'],100)}",
            num(r['ladder']['source_fit']['predicted']['delta_T']), num(r['ladder']['target_search_fit']['predicted']['delta_T']),
            num(r['ladder']['target_search_fit']['GT_anchor']['delta_T']), num(r['ladder']['target_search_fit']['GT_winner']['delta_T'])]
           for r in selected])
    lines += ['Vid source37/exposure/order2 is a correctly rejected bad replacement under both predicted fits. Giving only the winner GT turns it into a false accept because the anchor estimate remains much lower than its true quality. Vid source34/frame-drop/order1 is a genuinely useful replacement: the old source analytic map accepts it, while target fitting rejects it despite improving some individual P estimates. HC source34/frame-drop/order1 is a positive target-fit case: a false accept is corrected to reject. The preserved severe failures show why the HC aggregate balanced-accuracy gain is insufficient for safety.', '',
        '## 5. Clean, orders and uncertainty', '',
        'Clean is retained as a separate control, rather than pooled into the corruption claim. Only8 clean expert cells/dataset are available. Both orders are evaluated separately as point estimates; HC confirmation corrupt order2 has no helpful class, so its AUROC and balanced accuracy are undefined, not .5.']
    table(lines, ['Panel', 'Fit', 'AUROC [95% CI]', 'Balanced accuracy [95% CI]', 'Accepted helpful / harmful / severe'],
          [[NAMES[ds] + ' confirm clean', fit, packed(d['metrics']['auc']), packed(d['metrics']['balanced_accuracy']),
            f"{d['counts']['accepted_helpful']} / {d['counts']['accepted_harmful']} / {d['counts']['accepted_severe_v_harm']}"]
           for ds in NAMES for fit in FITS for d in [data[ds]['confirm_clean']['decision'][fit]['predicted']]])
    table(lines, ['Panel', 'Order', 'Source-fit AUROC / balanced accuracy', 'Target-fit AUROC / balanced accuracy'],
          [[NAMES[ds] + ' confirm corrupt', order] + [' / '.join(num(z['decision'][fit]['predicted']['metrics'][k]['mean'])
           for k in ['auc','balanced_accuracy']) for fit in FITS]
           for ds in NAMES for order,z in data[ds]['confirm_corrupt']['orders'].items()])
    lines += ['All12 dataset/panel summaries, raw regression metrics, source moments, both-order readbacks, within-source AUC, clipping counts, leave-one-source-out influence and paired source-bootstrap results are available under results/tastvg_pr_accessibility/2026-10-03. Primary confidence intervals use10000 whole-source paired draws, seed20261003; no multiple-comparison correction is claimed. Confirmation corrupt AUROC has45 undefined draws for Vid and198 for HC; undefined values are not replaced.', '',
        '## 6. Interpretation and decision scope', '',
        'The narrow supported conclusion is: target-search supervision can fit these frozen readouts locally, yet this fixed linear/alpha intervention does not restore their confirmation A/W absolute quality estimates. HC exposes partial relative-decision improvement; Vid does not. The analytic GT component ladder demonstrates coupled comparison errors and separates ranking from absolute acceptance.', '',
        'This result does not identify source→target domain shift as the sole cause. The refit changes training source count, candidate/anchor distribution, corruption mix, intercept and normalization statistics. It also cannot prove that final-layer features lack P/R information:16/14 training sources and one source-selected alpha per component are a limited supervised diagnostic. Conversely, fitting the search labels is not held-out recoverability. True P/R exactly reconstruct interval tIoU, but neither imperfect P/R nor even perfect temporal discrimination guarantees fixed-space vIoU improvement.', '',
        '**Decision: preserve A and production CURRENT_METHOD; do not deploy target-supervised weights or an oracle gate. No layerwise run, MLP, extra expert, full-query queue or further parameter search is started.**', '',
        '## 7. Verification, engineering history and resources', '',
        f"Eight meaningful CPU tests pass. Independent root audit passes {audit['total_scalar_checks']:,} scalar checks, including four verification refits with scipy SPD solve versus the producer eigen solver; original source-fit prediction parity; every payload/input binding; source separation; fit/readout/label ordering; physical candidate P/R labels; exact all-GT tIoU; all decision/metric/paired-bootstrap/leave-source values. Public scalar audit separately passes {read(OUT/'PUBLIC_AUDIT.json')['total_scalar_checks']:,} checks in the public checkout using no private features, weights or GT spans.", '',
        'One scoring-only schema repair is preserved in recovery/metric_key_001: the old metric rows store cell identity in separate fields, rather than a cell_key field. Original failure/log/runtime code remain archived. RUNTIME_REVISION_001 binds the corrected deterministic join while keeping the initial RUNTIME_LOCK, all fit weights, FIT_SEAL, PREDICTIONS and GLOBAL_READOUT_SEAL unchanged. No scientific setting, refit or readout was changed/repeated.', '',
        f"Scientific CPU wall time: fit {resources['search_fit_CPU_wall_seconds']:.4f}s, readout {resources['readout_CPU_wall_seconds']:.4f}s, diagnose {resources['diagnosis_CPU_wall_seconds']:.4f}s; root verification {audit['CPU_wall_seconds']:.4f}s separately. These are CPU wall times for cached work, not a full model runtime estimate. Four experimental ridge fits plus four independent verification refits; zero GPU, backbone, expert, new-candidate, decoder replay, backward, full online rollout or production parameter updates.", '',
        'Public export includes implementations, protocol/configuration, seals/bindings, anonymous scalar predictions/labels/negative cases, all summaries, audit receipts and vector/raster plots. It excludes fitted coefficients/normalizer arrays, latent features, GT spans/raw annotations, private video/media, weights and conversation attachments. The local research archive records the final remote byte verification receipt.', '',
        'Protocol: [protocols/tastvg_pr_accessibility_v1.md](../protocols/tastvg_pr_accessibility_v1.md). Results: [results/tastvg_pr_accessibility/2026-10-03](../results/tastvg_pr_accessibility/2026-10-03).']
    path = ROOT / 'docs/TA_PR_READOUT_ACCESSIBILITY_REVIEW.md'
    path.write_text('\n'.join(lines) + '\n')
    # Machine-readable table with means and CI for all roles/panels, from audited values.
    flat = []
    for ds,z in data.items():
        for panel,stats in z.items():
            for fit in FITS:
                for role,v in stats['regression'][fit].items():
                    for metric,m in v['metrics'].items():
                        ci=m['ci95'] or [None,None]
                        flat.append([ds,panel,fit,role,metric,m['mean'],*ci,m['bootstrap_defined'],m['bootstrap_undefined']])
    with (OUT / 'REGRESSION.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['dataset','panel','fit','readout','metric','mean','ci_low','ci_high','defined_draws','undefined_draws'])
        writer.writerows(flat)
    print(json.dumps({'status':'written','report':str(path.relative_to(ROOT)),'regression_table_rows':len(flat)}))


if __name__ == '__main__':
    main()
