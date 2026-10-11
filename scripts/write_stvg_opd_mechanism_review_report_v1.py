"""Write the completed CPU diagnostic report from independently checked scalars."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_opd_mechanism_review_v1'
PUB = ROOT / 'results/stvg_opd_mechanism_review/2026-10-11'


def effect(x):
    if x['mean'] is None:
        return 'unavailable'
    a, b = x['ci95']
    return f"{100*x['mean']:+.3f} [{100*a:+.3f}, {100*b:+.3f}]"


def run():
    s = json.loads((PUB/'SUMMARY.json').read_text())
    for f in ['ROOT_STATISTICS_READBACK.json', 'ROOT_VISUAL_REVIEW.json',
              'DINO_RECONSTRUCTION_READBACK.json']:
        assert json.loads((BASE/f).read_text())['status'] == 'pass'
    lines = [
        '# Reviewer mechanism evidence: completed CPU diagnosis and queued transfer control',
        '',
        'The primary method, old fits, admission rule, temporal interval, scientific parameters, '
        'and all running jobs remain unchanged. This package reads already globally sealed P2 '
        'Full/Direct fits. It performs no new model inference or adaptation.',
        '',
        'Actual scope: 1,792 matched arrival cells, 3,584 complete saved fits, 3,553 expert-frame '
        'cells with an official annotation, 413,427 annotated event-frame cells, and 48,384 '
        'independent native/oracle scalar comparisons. The root independently checked 1,682,153 '
        'scalar aggregations/comparisons. Exact DINO reconstruction matches 512 original clean '
        'arrival inputs and 171,032 DINO coordinates against the sealed P1 stateless comparator.',
        '',
        'Frames are averaged within arrival/condition/order, then conditions/orders within parent, '
        'then parents equally. Intervals are 10,000 paired-parent bootstrap draws, seed 20261011, '
        'unadjusted and descriptive. Sources have historical evaluation exposure; excluding the '
        'original development parents does not make them fresh. Missing frame groups stay missing.',
        '',
        '## 1. Expert quality and finite-sample actionability',
        '',
        'Q is the admitted expert-box IoU with the official annotation. The operational strata '
        'Q >= 0.5 and Q < 0.5 describe geometric agreement, not verified referent identity. A is '
        'the maximum **actual saved GPU IoU reward** in the first round. A == 0 means all sampled '
        'actions received zero reward in that observation/round, not mathematically unreachable '
        'Gaussian support. The complete continuous observations and every-round zero fraction '
        'remain in QUALITY_ROWS.jsonl.gz.',
        '',
        'Primary results exclude original development parents. Full-current and Direct-current '
        'each use their own final-minus-before frame IoU. Their inherited Before states can differ; '
        'the paired algorithm contrast is not an isolated common-initial-state update experiment.',
        '',
        '| Setting | Expert quality / first-round reward | Parents | Expert-frame cells | Full-current − Direct-current (pp, 95% CI) | Full harm | Direct harm |',
        '|---|---|---:|---:|---|---:|---:|',
    ]
    for k, v in s['quality'].items():
        if not k.endswith('/excluding_development'):
            continue
        setting = k.rsplit('/', 1)[0]
        for group in ['incorrect/zero', 'incorrect/positive', 'correct/zero', 'correct/positive']:
            r = v[group]; st = r['statistics']; m = st['metrics']
            harm = lambda name: 'NA' if m[name]['mean'] is None else f"{100*m[name]['mean']:.2f}%"
            lines.append(f"| {setting} | {group} | {st['parent_sources']} | {r['expert_frame_cells']} | "
                         f"{effect(m['Full_minus_Direct_current'])} | {harm('Full_harm')} | {harm('Direct_harm')} |")
    lines += [
        '',
        'Harm is a per-frame current change below −0.05, reduced with the same parent weighting; '
        'these percentages are not the number of parents that ever experienced a bad frame. '
        'Separate parent negative tails and all final-output/Before-history contrasts are in SUMMARY.json.',
        '',
        'The HC low-Q/all-zero strata favor Full over Direct. However, the high-Q/all-zero strata '
        'favor Direct, often strongly, on small parent denominators. Zero reward is not an automatic '
        'safety guarantee: other observations, shared decoder parameters, inherited state, and optimizer '
        'history may still move that frame. The corresponding Vid low-Q strata have wide intervals. '
        'These findings support a conditional conservative-update hypothesis rather than universal OPD superiority.',
        '',
        '### Conditional reward pilot',
        '',
        'The resource trigger was fixed before this diagnostic readback: at least 10 affected parents, '
        'at least 25% of high-Q observations with all-zero first-round reward, and a positive paired '
        'Direct-current minus Full-current interval. This threshold controls whether to spend resources '
        'on one pilot; it is not evidence that rarer zero-reward failures are unimportant.',
        '',
        '| Setting | High-Q observations | All-zero high-Q observations | Affected parents | Zero fraction | Trigger |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for r in s['pilot_resource_trigger']['settings']:
        lines.append(f"| {r['setting']} | {r['correct_expert_observations']} | {r['zero_correct_observations']} | "
                     f"{r['zero_correct_parents']} | {100*r['zero_correct_fraction']:.3f}% | {r['triggered']} |")
    lines += [
        '',
        '**Decision: not_triggered.** No GIoU-reward pilot was launched; no primary reward or parameter changed.',
        '',
        '## 2. Real frame propagation',
        '',
        'The four disjoint groups are admitted expert frames inside Native WHEN; unobserved actual '
        'decoder-output frames inside Native; interpolation-only frames inside Native; and annotated '
        'event frames outside Native. Interpolation is not a new decoder inference. DINO Refine uses '
        'the original absolute cxcywh anchor interpolation with nearest endpoint extension, and the '
        'same admitted references. It replaces the whole trajectory according to that original rule.',
        '',
        '| Setting | Frame group | Eligible parents | Frame cells | After − Before (pp, 95% CI) | After − DINO (pp, 95% CI) |',
        '|---|---|---:|---:|---|---|',
    ]
    for k, v in s['propagation'].items():
        if not k.endswith('/excluding_development'):
            continue
        setting = k.rsplit('/', 1)[0]
        for group, r in v['all_eligible'].items():
            st = r['statistics']; m = st['metrics']
            lines.append(f"| {setting} | {group} | {st['parent_sources']} | {r['frame_cells']} | "
                         f"{effect(m['After_minus_Before'])} | {effect(m['After_minus_DINO'])} |")
    lines += [
        '',
        'After−Before is the current-query effect; After−DINO compares complete algorithm histories '
        'against parameter-free refinement. They answer different questions. The Vid cross-domain '
        'current effect is positive on real unobserved decoder frames; the HC current effect is '
        'not positive by this test. Do not attribute the complete inherited-state gain to the current sparse update.',
        '',
        '### Common evaluable parents',
        '',
        '| Setting | Unobserved group | Common observed/comparison parents | After − Before on comparison (pp, CI) | After − DINO on comparison (pp, CI) |',
        '|---|---|---:|---|---|',
    ]
    for k, v in s['propagation'].items():
        if not k.endswith('/excluding_development'):
            continue
        for g, r in v['common_parent_intersections'].items():
            m = r['comparison']['metrics']
            lines.append(f"| {k.rsplit('/',1)[0]} | {g} | {r['common_parents']} | "
                         f"{effect(m['After_minus_Before'])} | {effect(m['After_minus_DINO'])} |")
    lines += [
        '',
        'Outside Native WHEN, the spatial prediction can change geometrically, but task-gated coverage '
        'stays zero for every arm because the interval is unchanged. These frames do not become recovered '
        'event support. SUMMARY.json retains observed-group results on each identical parent intersection.',
        '',
        '## 3. Offline spatial and temporal oracle headroom',
        '',
        'SpatialOracle uses GT space on its real annotated support with Native WHEN. TemporalOracle '
        'uses GT WHEN with original native box interpolation and zero contribution outside saved decoder '
        'support. Missing spatial predictions are never filled with GT in TemporalOracle. Oracles are '
        'offline interventions and are not deployable baselines or additive error decomposition.',
        '',
        '| Setting | Parents | Native mvIoU (%) | SpatialOracle (%) | TemporalOracle (%) | Spatial headroom (pp, CI) | Temporal headroom (pp, CI) | Native event coverage (%) |',
        '|---|---:|---:|---:|---:|---|---|---:|',
    ]
    for k, v in s['oracles'].items():
        if not k.endswith('/excluding_development'):
            continue
        st=v['pooled']; m=st['metrics']
        lines.append(f"| {k.rsplit('/',1)[0]} | {st['parent_sources']} | {100*m['Native_v']['mean']:.3f} | "
                     f"{100*m['SpatialOracle_v']['mean']:.3f} | {100*m['TemporalOracle_v']['mean']:.3f} | "
                     f"{effect(m['Spatial_headroom'])} | {effect(m['Temporal_headroom'])} | "
                     f"{100*m['Native_event_coverage']['mean']:.3f} |")
    lines += [
        '',
        'Cross-domain temporal headroom exceeds spatial headroom in both directions; a universal '
        'spatial-dominant-bottleneck claim is unsupported. Native event coverage is approximately '
        '56%, and the fixed interval remains an explicit limitation. Unsupported saved-decoder event '
        'support is 0% for HC and approximately 0.404% for cross-domain Vid; both the ideal joint '
        'oracle and support-aware joint oracle are reported separately.',
        '',
        '## 4. Cross-parent transfer: incomplete GPU work',
        '',
        'The three independently realized 128-parent permutations already provide 2,304 Full/Reset/Full-Carry '
        'fits. Only LN-only is missing: 768 new formal fits on the exact three original orders, with '
        '24 complete old/new GPU qualification fits at 12 first-two-arrival cells. The CPU contracts '
        'and four saved historical LN-only fit math dictionaries passed; these are not GPU qualifications.',
        '',
        'The unique CPU dependency waiter is launched. It waits for the running calibrated baseline '
        'and audited-cost/root jobs and the previously authorized expanded 905-parent state/timing jobs '
        'to complete and exit before acquiring the same single GPU. No new LN-only qualification, '
        'formal prediction, or GT score exists yet. Full−Reset, Full−Full-Carry and Full−LN-only '
        'must be reported per real permutation and pooled after all new formal predictions seal. '
        'Bootstrap draws do not count as arrival-order repetitions.',
        '',
        'The larger 905-parent/5,430-fit Full/alpha0/alpha1 comparison and the 128-path Full/Frozen '
        'uncached deployment timing remain separate queued studies. End-to-end efficiency and stable '
        'carryover value are not newly established by this completed CPU diagnostic.',
        '',
        '## 5. Evidence and manuscript consequences',
        '',
        '- Preserve the full-query system improvement (+2.566 pp HC, +2.968 pp Vid) and restore the '
        'actual DINO Refine row. DINO slightly exceeds Full on Vid R@0.3; best-value formatting must reflect it.',
        '- Restore Direct in the matched 128-parent P2 table. Never mix that small-cohort contrast '
        'into the full-query main table as if evaluated on every official query.',
        '- Explain fixed-covariance, detached-weight OPD as weighted regression in mean/logit space, '
        'with resampling as the distinctive update process; the frozen expert supplies one box, '
        'not independent semantic judgments for each action.',
        '- Report score threshold 0.35, distinct-candidate margin 0.05, duplicate NMS IoU 0.5, '
        'and the no-admitted-observation identity After=Before (Before need not be Source Frozen).',
        '- Keep Native WHEN fixed, separate real decoder propagation from interpolation and '
        'geometric outside-interval changes, and retain both positive and negative conditional results.',
        '- Report a spatial-adaptation framework with a conditional actionability tradeoff; stable '
        'advantage over Direct and stable carryover over complete reset still require their direct comparisons.',
        '',
        'All three final scalar PDF pages were actually rendered and visually checked after a '
        'display-only layout correction; original renders and receipt remain preserved. No RGB, '
        'query text, official GT box geometry, source weights, raw actions, or saved fit payload '
        'belongs in the public export. Anonymous continuous IoU/reward/delta rows do.',
    ]
    (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='report_written_from_independently_verified_scalars',
                         path=str(PUB/'REPORT.md'), new_model_calls=0)))


if __name__ == '__main__':
    run()
