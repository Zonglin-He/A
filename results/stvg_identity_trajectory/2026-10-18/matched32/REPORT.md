# Matched identity-hypothesis trajectory OPD32

Fixed previously exposed16 parents/domain, same frozen Source initial1792 coordinates and domain optimizer budgets. No inherited LN/writeback, additional DINO, Qwen, diffusion or trained association head. All32×4 complete final Student outputs sealed before target scoring. Historical online P7-A is not a matched baseline.

| Domain | Arm | Native vIoU % | MBR vIoU % | Native tIoU % | Native sIoU % |
|---|---|---:|---:|---:|---:|
| hc2 | Source | 18.831315 | 22.052668 | 45.792579 | 37.714391 |
| hc2 | top1_frame | 18.471518 | 21.620403 | 45.792579 | 35.073558 |
| hc2 | multi_frame | 18.465866 | 21.764193 | 45.792579 | 36.539915 |
| hc2 | multi_tube | 17.091043 | 19.880490 | 45.792579 | 33.468383 |
| hc2 | multi_direct | 20.585376 | 24.078576 | 45.792579 | 38.124700 |
| vidstg | Source | 7.147781 | 7.004092 | 31.818184 | 36.753536 |
| vidstg | top1_frame | 11.496033 | 11.353518 | 31.818184 | 47.183985 |
| vidstg | multi_frame | 11.672552 | 11.459453 | 31.818184 | 46.133595 |
| vidstg | multi_tube | 6.936525 | 6.751141 | 31.818184 | 31.919237 |
| vidstg | multi_direct | 11.118448 | 10.892626 | 31.818184 | 47.333639 |

## What changes and what remains uncertain

Top1 uses original framewise OPD, original admission/reward/seed. Multi-frame changes to cached association hypotheses, Native event-membership weighting and joint cross-anchor reward weights. Thus that contrast is a reference/feedback bundle, not a pure extra-box-count effect. Multi-tube shares exactly the latter feedback, adds smooth correlated complete-tube proposals and an exact mixture policy containing distinct interpolated identity paths. It is a policy change, not a dense-label claim. No identity coordinate mean or per-frame best-identity reward.

Same-evidence Direct maximizes the same differentiable marginalized compatibility with the same state, optimizer and maximum rounds; it can have zero gradient for distant disjoint boxes. Its failure alone would not prove generic OPD superiority over every direct objective. Association uses cached query-conditioned Source ROI appearance and DINO evidence. Coherent paths can still all identify the wrong person, and Top3 may omit the true object. Identity correctness remains unknown without pixel annotation. Outside-event evidence remains for association; event-membership weights geometry only and DINO never chooses boundaries.

Native time is primary; identical no-Actionness MBR time is secondary in all arms. MBR is a standard metric-aware readout, not new semantic supervision. Temporal64 uses a different larger historical cohort and must not be joined into this32 as a factorial population.

All signed per-source effects, unobserved actual decoder-frame geometry and damage tails are retained in RESULTS. Confidence intervals are paired parent bootstrap10000, unadjusted. These are development measurements, not new independent confirmation or full-scale evidence. Sparse reward and structured actions do not guarantee actual identity, event semantics or unobserved geometry.

Actual cached acquisition/fit costs are supplied; fresh deployment latency is unknown. No outcome automatically promotes the candidate, restarts an old queue or triggers parameter/temperature search.

## Decision in this finite development cohort

Multiple-hypothesis frame OPD minus original Top1 is HC -0.005652pp
(CI95[-0.594873,+0.652817]), Vid +0.176519pp
([-0.750874,+1.150051]): no stable improvement established. Structured tube
OPD minus multiple-hypothesis frame OPD is HC -1.374823pp
([-5.985078,+3.259368]), Vid -4.736027pp
([-9.662845,-1.223906]). Tube OPD minus same-evidence Direct is
HC -3.494333pp ([-7.945942,+1.108450]), Vid -4.181923pp
([-7.603748,-1.345252]). The tested structured policy is not promoted and
no larger run or automatic tuning follows. This does not prove every
association model or trajectory policy impossible.

Mean fixed-support sIoU on actual unobserved decoder frames (%):

| Domain | Source | Top1 frame | Multi frame | Multi tube | Multi Direct |
|---|---:|---:|---:|---:|---:|
| HC |37.712875|35.193196|36.885652|33.077317|37.116964|
| Vid |37.113932|47.390089|46.225425|31.930067|47.725885|

All16 sources/domain have evaluable unobserved decoder frames. These geometry
means use common annotated event support, independently of the selected
Native/MBR time, so a temporal-readout change cannot explain this geometry
contrast. Causal attribution to proposal shape, mixture-mode behavior or
actual wrong-identity association has not been isolated. No new identity
labels are inferred from these IoUs.

In this smaller cohort, Source MBR is HC22.052668%/Vid7.004092%, versus
Native18.831315%/7.147781%; the Vid change is negative. The cached64 MBR
mean gain must therefore not be presented as a universal per-cohort gain.

## Structural scope of this negative policy result

For the tested mixture, each round freezes component offsets relative to the
current actual Student logit tube. The likelihood gradient is a responsibility-
weighted sum of inverse-covariance residuals z-mu-offset_h. Selecting a distant
component by reward therefore does not itself write that component displacement
into the deterministic Student center. This interface can explore distant
proposals without transferring the identity jump. Its causal share of the
observed loss has NOT been isolated, and this negative run cannot reject every
whole-trajectory OPD design. The finite pilot stops without redesign/sweep.

Four private panels (fixed first/domain and posthoc worst tube-harm/domain)
were actually viewed:160 output cells, observed plus unobserved actual decoder
positions. H009 shows strong box shrinkage; V003 shows enlargement and shift.
These checks add no Query-specific identity labels; all identity correctness
remains unknown. No private pixels, GT boxes or Query text are public.
