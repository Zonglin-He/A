"""Describe the measured matched diagnostic, with its limits and contrary cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_ur_decomposition_common_v1 import *


def run():
    s = read(PUBLIC / 'SUMMARY.json'); resources = read(PUBLIC / 'RESOURCES.json')
    audit = read(PUBLIC / 'ROOT_READBACK.json'); influence = read(PUBLIC / 'SENSITIVITY.json')
    data = s['corruption']; roles = data['roles']
    def effect(role, branch):
        m = roles[role]['metrics']['delta_'+branch+'_v']
        return f"{100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]"
    def contrast(field):
        m = data['contrasts']['metrics'][field]
        return f"{100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]"
    decision = dict(status='completed_diagnostic_not_promoted',
        supported_in_scope='same-state Routed residual improves current query relative to common pre-state',
        donor_macro_broader_negative=roles['broader']['metrics']['delta_Specific_v']['ci95'][1] < 0,
        self_positive=roles['self']['metrics']['delta_Specific_v']['ci95'][0] > 0,
        text_near_positive=roles['near']['metrics']['delta_Specific_v']['ci95'][0] > 0,
        semantic_context_fast_memory_supported=False,
        generalization_beyond_R_base_supported=False,
        generic_Uniform_memory_gain_established=roles['broader']['metrics']['delta_U_v']['ci95'][0] > 0,
        full_Slow_Fast_started=False, current_prediction_overwrite_started=False,
        no_new_experiments_from_GT_influence=True,
        scope='exploratory historical HC32 first-step diagnostic; no full online or fresh confirmation')
    write(PUBLIC / 'DECISION.json', decision)
    table = '\n'.join('| '+role+' | '+effect(role, 'U')+' | '+effect(role, 'R')+' | '+effect(role, 'Specific')+' |'
                      for role in ['self', 'next', 'near', 'far', 'broader'])
    effects = roles['broader']['effects']['Specific']
    counts = sum(audit['checks'].values())
    minimum = {role: min(influence[role]['leave_one_donor_out'], key=lambda x: x['mean']) for role in ['self', 'broader']}
    maximum = {role: max(influence[role]['leave_one_donor_out'], key=lambda x: x['mean']) for role in ['self', 'broader']}
    body = f'''# Same-state U/R Write Decomposition Audit

## Result and decision

The matched residual improves the **current query**, while its donor-source
macro utility across **all future nonexpert targets** is negative. The stronger
claim of useful semantic-context fast memory is not established: text-near
residual utility is approximately zero, and near-minus-far is not positive.
This is a functional diagnostic, not a tested full Slow–Fast method.

All gains below compare one matched spatial SGD write with its **common saved
R pre-arrival state**. They are not full-method Ours-minus-Frozen results, and
not the earlier eight-step historical A/R writes. The temporal interval is fixed
to each target's common pre-state native interval; units are percentage points.
Intervals are paired 95% bootstrap intervals over donor video sources.

| Target | Uniform U | Routed R | Specific: direct gradient R−U |
|---|---:|---:|---:|
{table}

The paired Specific self-minus-broader difference is
**{contrast('delta_Specific_self_broader_v')} pp**; self-minus-far is
**{contrast('delta_Specific_self_far_v')} pp**. In contrast, text-near-minus-far is
**{contrast('delta_Specific_near_far_v')} pp** and text-near-minus-broader is
**{contrast('delta_Specific_near_broader_v')} pp**. The positive next-target
point estimate has a wide interval crossing zero and changes sign across orders.
It does not set a reliable short horizon.

The direct Specific intervention psi−lr(g_R−g_U) is evaluated independently.
It must not be confused with utility(R)-utility(U): nonlinear prediction and
evaluation make these two quantities different. Both are in the complete rows.

## What was actually run

- HC-STVG-v2 original 32 historically exposed development video sources, one
  query/source, two frozen original orders, clean plus five existing transient
  5% corruption conditions. Six conditions are repeated observations.
- Original expert arrivals 0,4,...,28: 96 donor-condition-order cells, 14 distinct
  donor video sources. Each corruption role has 80 cells; clean has 16.
- For each donor, use the historical **R pre-arrival** 1792-parameter state as
  the common base. This preserves the cached Routed5 route and supplies an exact
  R first-step positive control. Generate one identical nine-probe support for
  both critics from the source-fixed offsets.
- The original official HC-trained TA-STVG checkpoint, rank reverse KL,
  full-clip L1+GIoU compatibility, lr=.006097133675874025, teacher/student
  temperatures=1, rho=.05 and four antithetic directions are fixed.
- One matched step U, R and FP32 gradient difference R−U. No normalization,
  cap, extra loss, new expert, optimizer search, temporal rerank or full online
  method. Original K8 remains historical context rather than a matched effect.
- Cached corrupted input/H and Uniform5/Routed5 evidence are hash/pixel bound.
  Empty evidence is a branch no-op; nonempty flat rewards preserve original
  rank semantics. No GT filtering or gate.
- Frozen checkpoint RoBERTa lexical-mean cosine chooses maximum/minimum among
  **future nonexpert** targets, with earliest-arrival tie breaking. Next is the
  first such target; broader includes all of them. This eligibility differs from
  the older transfer audit, whose near/far could be scheduled expert arrivals.
- 384 logical named-role cells plus 1296 broader cells reduce to 1392 unique
  donor-target pairs; aliases are explicit. For each, compare common pre/U/R/
  Specific with no target training or expert observation.
- Seal all writes and predictions globally before CPU labels and official dense
  metric scoring. These sources were used by earlier research; this is not a
  fresh test-set claim.

## Strength, uncertainty and negative cases

The two orders give Specific self gains of
{100*roles['self']['metrics']['delta_Specific_v']['order_values'][0]:+.4f} and
{100*roles['self']['metrics']['delta_Specific_v']['order_values'][1]:+.4f} pp;
broader gains are
{100*roles['broader']['metrics']['delta_Specific_v']['order_values'][0]:+.4f} and
{100*roles['broader']['metrics']['delta_Specific_v']['order_values'][1]:+.4f} pp.
The main donor-source aggregation makes each donor equally weighted, averages
targets within that donor, then conditions and orders. Target-source aggregation
is a different, also published sensitivity estimand; its broader Specific
interval crosses zero. Neither repeated targets nor orders are independent data.
Intervals are exploratory and unadjusted for the multiple diagnostic readouts.

One donor has a large effect on magnitude. Leave-one-donor-out self means range
from {100*minimum['self']['mean']:+.4f} to {100*maximum['self']['mean']:+.4f} pp;
omitting anonymous donor source {minimum['self']['omitted_donor_source_id']}
reduces self utility to {100*minimum['self']['mean']:+.4f} pp. Broader means range
from {100*minimum['broader']['mean']:+.4f} to {100*maximum['broader']['mean']:+.4f}
pp. No source is removed, and this posthoc influence check is not used for an
online rule or scientific parameter selection.

Across 1080 corruption broader cells, Specific has {effects['positive']} positive,
{effects['negative']} negative and {effects['zero']} unchanged effects; there are
{effects['harm_gt5pp']} losses greater than 5 pp. Correct-to-incorrect versus
incorrect-to-correct cases at .3/.5, gross gain/loss, both-nonempty sensitivity,
target aggregation and positive/negative examples are all retained. Interval
changes and freely decoded temporal/spatiotemporal effects are independently
reported; fixed-time gains isolate spatial utility rather than a readout switch.

Clean is not uniformly protected by the residual: self is
{100*s['clean']['roles']['self']['metrics']['delta_Specific_v']['mean']:+.4f} pp,
broader is {100*s['clean']['roles']['broader']['metrics']['delta_Specific_v']['mean']:+.4f}
pp with a crossing-zero interval. The current-query pattern is not exclusively
a deployment-corruption repair mechanism.

The gradient cosines/norms and same-probe critic pairwise diagnostics help locate
the difference in evidence, but parameter cancellation does not establish GT
harm, and text-only cosine does not establish shared event or visual geometry.
The geometry distribution is a compatibility surrogate, not a native tube policy.
All results are conditional on the precommitted historical R bases; another
base-history population and long-term persistent Uniform transfer are untested
by this one-step diagnostic.

## Validation and computation

Two clean smoke donors passed before full replay. Across all 96 donors the
recomputed R route, nine probes, loss/distributions, gradients, one-step state
and self prediction match the saved R first step bitwise. Independent CPU reward,
rank/KL, residual and high-precision SGD reconstruction pass, including empty
feedback. Root checked {counts:,} scalar/binding items; maximum dense metric
disagreement is {audit['max_metric_error']:.3g}.

There were {resources['suffix_replays']} actual cached native-suffix replays,
{resources['backward_calls']} backward calls and 192 cached expert reads.
**Zero new backbone forwards and zero specialist inference calls.** Combined
smoke/full worker wall time was {resources['worker_wall_seconds']:.3f} seconds
({resources['worker_wall_seconds']/60:.2f} minutes), including load and I/O,
not pure GPU kernel time. Peak allocated VRAM was
{resources['peak_allocated_vram_bytes']/2**30:.3f} GiB. The model state is restored.
The anonymous public auditor independently reconstructs metrics, contrasts,
coverage, gradient norm identities, ranking/distribution and bootstrap scalars.

## Scope of the next method decision

The audit establishes a narrow useful lead: Routed's extra evidence can improve
the current query while harming generic future utility under these matched
states. It does **not** establish a text-context horizon, reliable cross-query
fast-state reuse, or a proven persistent Uniform memory advantage. A complete
Slow–Fast mechanism is therefore **not implemented or promoted in this batch**.
No current output is overwritten, no semantic memory/gate is selected with GT,
and no latest-reset, new token cosine model or historical queue is launched.
Temporal cross-view reliability and functional token attribution remain separate
later questions. Production CURRENT_METHOD is unchanged.

## Artifacts

- Scientific protocol: `protocols/tastvg_ur_write_decomposition_v1.md`.
- Anonymous rows: `ROWS.json`, `WRITE_ROWS.json`, `CONTRAST_ROWS.json`.
- Complete summaries/controls: `SUMMARY.json`, `SENSITIVITY.json`, `CASES.json`.
- Configuration, runtime provenance, resources, root/public audits and decision.
- `corruption_write_decomposition` and `clean_write_decomposition` in PNG/PDF/SVG.
- Private labels, captions/media, raw H, weights and state/gradient tensors are
  excluded from public export; historical inputs remain unmodified.
'''
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / 'REPORT.md').write_text(body)
    (ROOT / 'docs/TA_UR_WRITE_DECOMPOSITION_REVIEW.md').write_text(body)
    archive('同状态P0实际完成；Specific self+.4025pp而broader−.1753pp、near近零；保留源影响/clean/反例，完整Slow–Fast未获资格，准备公开')
    print('Report/decision recorded; complete diagnostic, no method promotion')


if __name__ == '__main__':
    run()
