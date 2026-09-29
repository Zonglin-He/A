# J0.1 — Online Schedule Robustness

Five source-hash orders locked before execution. Exact J0 method code and hyperparameters unchanged. Fast positive orders: 5/5. Research recipe freeze for subsequent evaluation: **True**.

## Primary five-order comparison

Differences in percentage points. Corruption: average five conditions within source, then16source whole-stream or12source nonexpert macro. Across-order row uses equal order weights and sample SD(ddof1), not a confidence interval.

| Order | Whole Fast−Frozen vIoU | Whole Final−Fast vIoU | Whole Final−Frozen vIoU | Future Final−Fast sIoU | Future Final−Fast vIoU |
|---|---:|---:|---:|---:|---:|
| order1 | +1.949680 | +0.037739 | +1.987420 | +0.177227 | +0.052204 |
| order2 | +2.226935 | -0.002779 | +2.224157 | -0.013591 | -0.003254 |
| order3 | +1.431016 | +0.018859 | +1.449875 | +0.067394 | +0.034033 |
| order4 | +1.835035 | +0.028263 | +1.863299 | +0.122259 | +0.022202 |
| order5 | +0.038530 | +0.025122 | +0.063653 | +0.118661 | +0.022337 |
| Five new orders: mean ± SD | +1.496239 ± 0.863637 | +0.021441 ± 0.015157 | +1.517681 ± 0.859943 | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 |
| Original J0, separate reference | -0.072005 | +0.035686 | -0.036318 | +0.078608 | +0.047181 |

Fast whole-stream sign counts: 5positive/0negative/0zero. Final whole-stream: 5positive/0negative. Spatial future vIoU: 4positive/1negative; mean+0.025505pp, SD0.020209pp. Spatial future sIoU: 4positive/1negative. Do not infer stable spatial transfer from positive combined Fast-dominated performance.

The original order is not included in the primary five-order mean. INCLUDING_J0.json separately reports all six including the earlier unfavorable stream; no result was discarded, no best-order selection. Orders share the same16sources and expert caches, so these are descriptive order variations, not five independent cohorts. Per-order conditional source-bootstrap intervals are in SUMMARY.json.

## Locked specialist availability

| Order | Expert source aliases, in arrival order |
|---|---|
| order1 | Q02, Q12, Q07, Q11 |
| order2 | Q01, Q11, Q14, Q04 |
| order3 | Q04, Q12, Q11, Q07 |
| order4 | Q12, Q08, Q06, Q11 |
| order5 | Q16, Q15, Q10, Q03 |

Hash rule: sort original source IDs by SHA256(UTF8("J01-source-order-20260929|j|source")),then source, j=1..5. Same order used across all six conditions, expert positions0/4/8/12. No rejection/resampling based on data or scores. Anonymous complete sequences are in ORDERS.json; reproducing original source-ID hashes requires authorized private roster.

## Clean control

| Order | Whole Fast−Frozen vIoU | Whole Final−Frozen vIoU | Future Final−Fast sIoU | Future Final−Fast vIoU |
|---|---:|---:|---:|---:|
| order1 | +1.942018 | +1.981082 | +0.173928 | +0.054424 |
| order2 | +1.621335 | +1.619312 | -0.016593 | -0.002963 |
| order3 | +1.511931 | +1.533655 | +0.068558 | +0.039668 |
| order4 | +1.481150 | +1.510673 | +0.097701 | +0.024247 |
| order5 | +0.642092 | +0.673749 | +0.134218 | +0.028505 |

## Tier0: exact finite-roster subset and margin audit

All1820four-of16historical corruption source subsets enumerated. Original J0 selected-subset gain-0.288018pp; inclusive percentile10.219780%,strict percentile10.109890%. Negative subsets226/1820=12.417582%.

Selected-subset mean gain quantiles (pp): 5%=-1.024633, 25%=+0.942575, 50%=+2.652339, 75%=+5.293954, 95%=+8.554318.

These are four-selected-source means; whole16arrival Fast gain is one quarter of each. The1820overlapping subsets do not estimate a population deployment failure rate or repeat persistent adaptation. The five actual online orders above are separately executed.

Pooled-cell margin/gain Pearson: all96=0.538135,scheduled24=0.128486. Scheduled upper-quartile margin subset gain=-0.283408pp (6cells). Purely descriptive, repeated source/conditions; no margin/entropy/confidence threshold deployed.

## Execution, validity and costs

480new Final arrivals across30source-reset streams; 102actual spatial updates and1080current-policy spatial rollouts. Encoder/experts reused, new expert inferences0. Persistent states recomputed for every order; old S1/J0 learned states were not reused. Independent order1/clean Slow-only replay matches all32pre/post states and outputs, with4extra validation backward steps.

State links480/resets30; 182784independent SGD coordinates, 102rankKL reconstructions, 1920independent temporal scores, 1920dual metrics;4full spatial and2all-six-layer temporal native reinsertions;3order contracts;35709public scalar checks. Current temporal support changed in0/120expert cells; selected interval changed in0/120. Nonexpert callbacks never read specialists; current spatial output always pre-update.

GPU process total211.644529s, failed attempts0. No downloads, new expert inference or encoder capture. Per order including clean: Fast24T,Slow24S,Final24T+24S logical calls; all share25%availability, not equal total specialist cost. Cached runtime is not uncached deployment latency.

## Decision boundaries

User branch outcome: A_schedule_dependence. Method recipe freeze=True; no order selected for deployment. Preserve the original J0 negative and all new order results. If Fast is mostly positive, the evidence supports schedule-dependent temporal performance on this roster, not universal sparse-temporal failure. It does not imply each schedule is safe. Spatial future gains are separately reported; any mixed/negative order results prevent calling online transfer uniformly stable.

Do not add a gate, rerun favorable schedules, change Spatial or restart temporal/parameter-space OPD. Larger/fresh/order stability evaluation remains a subsequent scoped phase; historical production CURRENT_METHOD unchanged. Repeated exposure and the same source corpus limit generalization.
