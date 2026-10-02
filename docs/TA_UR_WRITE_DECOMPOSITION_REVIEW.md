# Same-state U/R Write Decomposition Audit

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
| self | +0.0244 [-0.1166, +0.1348] | +0.3066 [+0.0702, +0.7219] | +0.4025 [+0.0175, +1.1453] |
| next | -0.2340 [-1.0152, +0.2622] | +0.3565 [+0.0116, +1.0068] | +0.6345 [-0.3147, +2.2621] |
| near | +0.1563 [+0.0173, +0.3854] | +0.1549 [+0.0223, +0.3774] | -0.0062 [-0.0280, +0.0129] |
| far | +0.0182 [-0.0356, +0.0610] | +0.0353 [+0.0093, +0.0626] | +0.0082 [-0.0293, +0.0590] |
| broader | +0.0870 [-0.0459, +0.2746] | -0.0506 [-0.2557, +0.0733] | -0.1753 [-0.4714, -0.0078] |

The paired Specific self-minus-broader difference is
**+0.5778 [+0.0272, +1.6150] pp**; self-minus-far is
**+0.3943 [+0.0272, +1.0960] pp**. In contrast, text-near-minus-far is
**-0.0144 [-0.0647, +0.0224] pp** and text-near-minus-broader is
**+0.1691 [-0.0100, +0.4709] pp**. The positive next-target
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
+0.6796 and
+0.0400 pp;
broader gains are
-0.2851 and
-0.0305 pp.
The main donor-source aggregation makes each donor equally weighted, averages
targets within that donor, then conditions and orders. Target-source aggregation
is a different, also published sensitivity estimand; its broader Specific
interval crosses zero. Neither repeated targets nor orders are independent data.
Intervals are exploratory and unadjusted for the multiple diagnostic readouts.

One donor has a large effect on magnitude. Leave-one-donor-out self means range
from +0.0370 to +0.4351 pp;
omitting anonymous donor source 20
reduces self utility to +0.0370 pp. Broader means range
from -0.1903 to -0.0359
pp. No source is removed, and this posthoc influence check is not used for an
online rule or scientific parameter selection.

Across 1080 corruption broader cells, Specific has 419 positive,
397 negative and 264 unchanged effects; there are
10 losses greater than 5 pp. Correct-to-incorrect versus
incorrect-to-correct cases at .3/.5, gross gain/loss, both-nonempty sensitivity,
target aggregation and positive/negative examples are all retained. Interval
changes and freely decoded temporal/spatiotemporal effects are independently
reported; fixed-time gains isolate spatial utility rather than a readout switch.

Clean is not uniformly protected by the residual: self is
+0.0877 pp,
broader is -0.0421
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
feedback. Root checked 895,918 scalar/binding items; maximum dense metric
disagreement is 1.55e-15.

There were 5898 actual cached native-suffix replays,
182 backward calls and 192 cached expert reads.
**Zero new backbone forwards and zero specialist inference calls.** Combined
smoke/full worker wall time was 454.529 seconds
(7.58 minutes), including load and I/O,
not pure GPU kernel time. Peak allocated VRAM was
1.093 GiB. The model state is restored.
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
