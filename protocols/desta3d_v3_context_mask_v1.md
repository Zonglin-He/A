# Source context-support control, v1

Status: registered before new GPU results. User attachment1d8cf856 authorizes this
single-factor experiment. Current large late correct masks did not beat B1: event
tIoU -2.532000pp, spatial sIoU -1.712647pp. Local gains and wrong-control damage
remain. These results weaken magnitude-alone in this setting; they do not prove
all scales/locations impossible or the vector alignment of a good mask direction.
Question: does preserving one observed boundary neighbour / one latent-cell
spatial neighbourhood improve evidence directionality at the same fixed scale?

Same original PANEL16 (one query per16 exposed Vid training parents,7-32frames),
official frozen PTD4B and B1 snapshot SHA6fe8d07e2a11c16bcb68eed11264a18b71985e97fdbf7939422f306ba81bf0d1,
adapter4a2ef2cf87e1753fad7c0c5c1f582f499c14b0b2945a839680b296abb0d9c7eb.
Exactly5 arms: original, event_context_correct/wrong, spatial_context_correct/wrong.
Same late post-reader/LN/SiLU alpha.25 scalar attenuation and native event-first,
fresh-spatial-KV decode. Keep ordinary B1 residual B=F+R. Scale signed modified
mask minus B1 delta independently per case/arm to ||delta||/||stockF||=.087687 event,
.170316 spatial. FP64 norm, FP32 multiplication/addition, original BF16 cast;
continuous norm matched, not BF16 difference count/norm. Same tolerances as large
mask:1e-6 scaled relative and1e-3 realized FP32 addition, absolute1e-10 floor.
No optimizer/backward/reader training/OPD/target/external expert/GT decoder prefix.

ONLY mask support changes. Temporal: start inclusive/end exclusive physical GT
interval, plus the nearest existing observation strictly before start and nearest
existing observation at or after end. At an unavailable clip boundary add nothing.
Do not interpolate/sample pixels or use percentage dilation. Apply identical rule
to the old fixed wrong interval; do not choose a new wrong interval.
Spatial: one Chebyshev-radius1 2D dilation per time slice, precisely 3x3 maximum
of original fractional occupancy values (not binary conversion); clip neighbourhood
to the existing latent grid, no wrap or temporal mixing. Missing/unannotated box
frames remain all-one neutral. Use the old translated wrong boxes with identical
operator. Border clipping may break old equal-area/weight matching; report both
expanded weights and do not reselect wrong boxes. Induced nonzero merger deltas
are separately norm matched. This tests context-preserving scalar modulation, not
spatial segmentation GT or a different visual representation.

Before GPU, save all64 masks and independently compute each physical-neighbour /
cell neighbourhood using CPU scalar NumPy loops. Prespecify all16 primary support,
old13 temporal eligible, new distinct/non-neutral matched eligible, border and
one-sided neutral cases. If a direction is zero, retain Base, record inapplicable
positive requested norm; never invent a direction/drop source. CPU full-panel
readback finds13 temporal and16 spatial distinct/non-neutral,0 one-sided neutral;
these counts are outcome-blind, not a post-score screen.

New Base16 physical inputs/preprocess/query/grid/time/reference/endpoint/boxes/
full recorded logits must replay old sealed oracle exactly. Recompute old late
raw directions once without native decode and require exact old raw identity;
only then form new context delta. Save full stockF, B1 FP32 endpoints, context
masks, full new raw/scaled deltas and every sparse BF16 change before/after/hash.
Save predictions before structure verification. Invalid grammar/zero boxes remain;
no token repair or sample exclusion. Spatial event branch must remain exactly
Base; format failure zero scoring is separated from direct endpoint-only tIoU.
80 outputs and all evidence sealed/hash-verified before offline source scoring.
Source GT is already used for masks, is allowed only for this oracle/diagnostic;
no new target data. This is exposed source development, not unlabeled TTA.

Report t/s/v parent means, correct-Base and correct-wrong all16/old13/neweligible,
paired bootstrap10000 seed20260927, raw cases, native-good v/t>.5 retention,
>5pp tails, actual pre/postcast changes, reference/interval/support and conditional
KL. Scalar/tensor geometry, independent NumPy raw/FP32 addition/integer BF16 hash
reconstruction and independent root parent/CI reduction. Compare context outcomes
to sealed large_mask002 per case at same scale; no old intervention GPU rerun.

Decision fixed before inference: teacher advantage requires correct>Base AND
correct>wrong in the corresponding event tIoU/spatial sIoU; distinguish positive
means from uncertainty and broad case support. Failure to establish this pair
stops further scalar-mask variations, not the THW/dual-reader mainline. Mixed or
uncertain effects are not a pass. No follow-on actual-reader/OPD here. Directional
residual remains a separately designed, untested alternative, not a new training
job. Reachability demonstrated on two old outcome-selected source cases does not
establish directionality, learnability, distillability or latent sufficiency.

Engineering: one1800s serial allocation, at most16GiB new evidence, >=8GiB free;
all real model load/fail/replay/finalization receipt time added to42340.06753310408s,
cumulative cap=null. CPU pre-GPU pin hashing separate. No separate subprocess
wrapper charge. Failures preserve original pins/partial/receipt and require an
isolated recovery version. Single existing Luna max one process/log/disk check,
30min heartbeat, no GT/prediction/raw/GPU actions. Public code/protocol/anonymous
aggregate only; CURRENT/full-source cancellation/old optimizer corrections stay.
