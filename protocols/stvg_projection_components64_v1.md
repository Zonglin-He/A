# Exact cached64 actionness projection component readouts

Explicit user authorization 2026-10-18; retrospective cached original exposed32
parents/domain, same TA-STVG cross-domain Source checkpoints and original P6
zero logits/actionness. CPU only, four threads, no models/GPU/expert/Student
fitting, downloads, new streams, GT feedback or production change. New independent
scope; original cached predictions, scores and paused queues remain immutable.

Claim: separate the readout effects of weighted-mean actionness length pressure
and its zero-weighted-mean fluctuations, preserving Native prior and all physical
readout semantics. This is a diagnostic, not a claim of new temporal training.

Use the existing projection implementation's float32 actionness conversion then
float64, median/MAD normalization, legal indices s<e, physical time-cell omega,
prior lambda=.1, original standardization/config. bar_u=sum_j omega_j u_j,
sum omega=1. For I=[s,e] in sampled index space, L(I)=bar_u sum_I omega and
F(I)=sum_I omega(u-bar_u). Apart from the candidate-independent constant
C=sum omega logsigmoid(-u), original score is lambda*lp0+L+F. Omega length is
Voronoi physical-cell mass and differs from endpoint interval seconds/frame
length: the cell edges are midpoint boundaries, while output is [fid_s,fid_e+1).

Four primary arms: Native (unchanged float32 original MAP); full_projection
(exact original project target and reduction order); length_only argmax
lambda*lp0+L; fluctuation_only argmax lambda*lp0+F. Components preserve float64,
same legal-index ordering and first torch argmax tie rule. Independently enumerate
every legal interval by direct sliced sums to verify both terms, their sum with
C, and original score, errors and selected maxima. Full projection offset/fused
must match old ActionnessProjection and cached interval-decision readouts exactly.
Existing native_MBR is a reference copied from its sealed decisions; do not run
or change its algorithm. Record all decision coincidences and weighted/endpoint
length differences, do not retune/tie-break with GT.

Seal complete64 (all arms, both offsets, fused physical envelope, all geometry
references and mathematical audit) before reading GT or full-feature strata.
Then cross each readout with fixed Source, original online Before, original
online After (primary), no spatial recomputation/parameter/state changes. Report
v/t/s, R03/R05 using historical strict >.3/>.5, actual legacy official HC
endpoint convention, each offset and fused. Independent official and DenseTube
metric checks plus independent aggregate/paired/old-result checks. All queries,
positive/negative/neutral and unknowns remain. Parent-cluster bootstrap10000 draws
fixed seed20261018, query and equal-parent means; repeated supports/geometries
are conditional diagnostics, not independent sources.

Only after prediction seal join existing full-cohort pooled event-length/start
bins. Report all four bins incl small/late event strata, pred/GT physical event
length ratio (physical GT duration uses the existing corrected feature), legacy
metric-span length ratio separately, predicted duration/input-span ratio,
newly added/lost known-GT support IoU and unknown frames, vs Native. sIoU uses
fixed all-known GT frame support and is invariant to temporal-only readouts.
Intervals added outside annotation are unknown, not zero-quality targets.

Do not force arm improvements to add: argmax and two-offset envelope are nonlinear.
Full-minus-single-component is conditional readout evidence, not a causal share,
not proof that length/semantic fluctuations alone explain all benefit. MBR and
old projection claims/negatives stay. Retrospective exposed64, no full13785 claim,
no automatic TTA or deployment. No new semantic annotations.

Publish exact code/config/protocol/aggregate and anonymous signed metrics, with
all uncertainty and limitations. Exact GT coordinates/time spans and per-query
exact GT-length ratios stay private (public ratio bins and aggregate ratios).
No media/weights/tensors/raw query/parent IDs. Verify real remote bytes/SHA/blob/
mode/tree, fresh CAS both refs, CURRENT/MAIN/PAPER and52 untracked preservation,
archive current+append/read-latest and check-snapshot-check completion receipts.
