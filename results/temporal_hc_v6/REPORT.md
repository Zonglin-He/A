# Case-based temporal readout: final-refinement continuation

**No continuation candidate met both Source-duration-group criteria.**
This is an executed cached-only study; the full queue remains paused.
128 tuning parents and the same35 previously exposed development-recheck parents
are retained. The35 prior failure cases are used to diagnose mechanisms, not to
claim independent validation. No new media/model/teacher inference was performed.

## What the long-event cases reveal

In the previous MBR³35-source recheck there are8 long events harmed versus Source.
Seven have final-layer tIoU at least as high as all shallower individual-layer MBR
readouts; P157 is the exception. The total tIoU changes over those8 are−47.9812pp
for Source→Global6, then an additional−12.6647pp for Global6→MBR³. These are sums
over8 cases, not aggregate dataset improvements. The first-stage layer averaging
accounts for most damage in this selected failure slice.

P131 loses87.023→69.466% before MBR³ changes anything. P139 loses68.912→64.767%
from averaging, then→56.995% from MBR³. P141 similarly loses39.583→33.249→30.073%.
[All44 long harmed cases from tuning/recheck](case_audit.json) retain each layer's
IoU, signed changes, better-candidate posterior ranks and Source posterior dilution.
They do not export video/query text, GT coordinates or raw features.

Some candidates better than Source occur among the first few posterior ranks, while
others are much lower. This oracle analysis is offline only; existence of a better
candidate does not establish a label-free rule that identifies it. Seven-of-eight
is a selected failure statistic, not evidence that the final layer is universally best.
Across all24 long recheck cases, layer4 averages53.167% versus final52.102%.

The frozen upstream TA-STVG pipeline sends the last temporal layer to `pred_sted`
and uses earlier outputs for auxiliary supervision. Its temporal training objective
matches separate endpoint distributions to Gaussian targets. Equal depth averaging
is our output readout; it is not the upstream native inference rule. Shallow and
final stages can therefore represent different stages of correction rather than
exchangeable ensemble members. This is consistent with the cases, not a universal
claim that early layers contain no useful information.

## Implemented minimum change

Keep final-layer endpoint log-probabilities as anchor. Let d5=l5−l4 and d6=l6−l5.
The bounded candidate adds sign(d6) min(|d5|,|d6|) only at locations where d5*d6>0.
Normalization removes arbitrary layer-wise logit shifts before testing signs.
The physical two-offset hull and ordinary MBR remain unchanged; no temperature,
strength, event-length branch, fitted weight or GT online condition is added.

The unbounded control adds d6, while the reverse control subtracts the bounded step.
Source and Global6 controls reproduce their temporal outputs exactly. Distinct
layer feedback also appears in [DoLa](https://proceedings.iclr.cc/paper_files/paper/2024/file/edc36117f795ca52a0cbf6a7b3882859-Paper-Conference.pdf)
and its [official implementation](https://github.com/voidism/DoLa), which concern
language-model token decoding. Our endpoint continuation is a separate hypothesis;
that literature does not validate its STVG performance.

| Readout | Short Δt vs Source pp | Long Δt vs Source pp | Overall Δt vs Source pp | Full Δv vs Source pp |
|---|---:|---:|---:|---:|
| [consistent](tune/consistent.json) | -0.3445 | +0.0978 | -0.0266 | +0.6841 |
| [contrast](tune/contrast.json) | -0.9267 | +0.2487 | -0.0819 | +0.6869 |
| [global6](tune/global6.json) | +1.4263 | -0.0774 | +0.3455 | +1.0567 |
| [reverse](tune/reverse.json) | -0.1021 | +0.0355 | -0.0032 | +0.7586 |
| [source](tune/source.json) | +0.0000 | +0.0000 | +0.0000 | +0.7042 |

The bounded candidate improves long events by+.0978pp but reduces short events
by−.3445pp versus Source. The unbounded candidate trades an even larger short loss
for long gains. Even two consecutive same-sign corrections can overshoot or reinforce
a mistaken boundary. No candidate was frozen for35 recheck and no full was launched.
No strength sweep follows this failure. The next isolated mechanism study instead
constrains the ensemble decision to a late-supported range:
[study v7](../temporal_hc_v7/REPORT.md).

[Protocol](protocol.json), [fixed5-mode schedule](schedule.json), [outcome](outcome.json).
Original model/precision/sampling/eight observations and BoxFixed remain unchanged.
All signed positive/negative query metrics and paired parent CIs remain in linked JSON.
Costs are cached CPU posterior/MBR/DP/file-write cost, not full deployment cost.
