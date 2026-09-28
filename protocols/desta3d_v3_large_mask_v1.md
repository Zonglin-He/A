# Large-magnitude source mask control, v1

Status at registration: proposed scientific effect; CPU engineering contracts checked.
Question: do the original late correct masks gain directional utility when their
merger perturbation is raised to a magnitude already reached by the two completed
source span controls? Prior tiny masks and optimized span directions differed in
magnitude, direction and objective. The prior matched-norm location study did not
establish an early-location advantage. This experiment changes only magnitude.

Same original PANEL16, one query per16 exposed Vid training parents, official
frozen PTD4B and fixed B1 (snapshot SHA6fe8d07e2a11c16bcb68eed11264a18b71985e97fdbf7939422f306ba81bf0d1,
adapter4a2ef2cf87e1753fad7c0c5c1f582f499c14b0b2945a839680b296abb0d9c7eb).
All model/adapter/gate parameters frozen, no optimizer/backward. Same observed
pixels, source-derived sealed oracle001 masks and matched wrong masks, missing
box support neutral, native official cached event-first/spatial-second readout.
No target inputs/GT, external expert, new native prefix, sample replacement,
loss/GT output selection, insertion change, context expansion or OPD.

Exactly5 arms: original, event_large_correct, event_large_wrong,
spatial_large_correct, spatial_large_wrong. No dual. Fixed old late modulation
post-reader/LN/SiLU, before branch output projection, alpha .25. Let baseline
B_b=F+R_b be the frozen B1 branch endpoint and d_b=masked_endpoint-B_b be the old
signed FP32 late delta. Inject B_b+c_b*d_b, keeping B1's existing residual.
The norm denominator is original captured stock F, not B_b. Independent scales
for correct/wrong enforce ||c*d||/||F||=.087687(event), .170316(spatial).
These are the user's boxed six-decimal constants from the prior two span cases,
not an amplitude search or an independently validated universal scale. Full old
ratios .08768700551364271/.1703159515581316 remain in original evidence.

FP64 norm accumulation, FP32 multiply/add, real original BF16 cast, no precision
change. Scaled norm relative tolerance1e-6; realized FP32 addition norm tolerance
1e-3 (absolute1e-10 floor). Postcast norms/counts need not match. A zero raw
mask direction cannot attain positive norm: preserve baseline, record requested
norm and applicable=false/realized0; never synthesize a direction. Old metadata
fixes neutral time cases4591/28649/15560. All16 remain in primary full-panel
report, with the pre-existing13 eligible temporal negatives separately reported.
Unexpected one-sided zero correct/wrong is a contract failure, not a sample skip.
Spatial16 are eligible. Raw directions must exactly replay the old late raw.

Store complete original stockF and both FP32 B1 endpoints, full raw/scaled mask
deltas, all sparse BF16 changes and hashes, native predictions/distributions/
reference/interval/box support, actual branch injection and frozen-state checks.
Save native outputs before validating structure. Invalid syntax/zero boxes remain
scored failures; missing downstream pass is explicit. Spatial-only event outputs
must equal baseline, but a spatial format failure is not prohibited. New base
all16 must exactly replay prior native outputs/logits/physical identity. Seal all
80 outputs plus evidence before offline source scoring. SourceGT has already
supplied oracle masks; this is not an unseen-label experiment or unlabeled TTA.

Primary decision uses BOTH correct-minus-original and correct-minus-wrong:
event tIoU and spatial sIoU. Full t/s/v, per-parent cases, bootstrap10000 seed20260927,
all16/eligible13, native-good v/t>.5 retention and >5pp negative tails are required.
Report real pre/postcast magnitudes, fixed-condition KL (coordinate1001 conditional,
not full vocabulary), GT class margins on actual native positions, format and
reference/support changes. Compare with sealed small late results as historical
same-direction readback, not new inference. Independent NumPy raw/addition/cast
and scalar/tensor geometry plus second parent/CI reduction before conclusions.

Interpretation: correct benefit AND correctness separation motivates a separately
registered actual-reader supervised control; strong effects without correctness
separation may motivate a same-operator/norm context-support test; widespread
collapse weakens this amplified scalar direction and may motivate directional
residuals. Mixed/uncertain results stay mixed. No next experiment auto-launched.
Reachability in two source examples does not establish latent hidden-information
sufficiency, actual reader learnability, distillability, target or generalization.
Large equivalent latent coefficients alone are not a stopping rule. No claim that
one null configuration refutes latent adaptation. All prior positives/negatives
and permanent optimizer restoration corrections remain.

CPU preflight: real hidden128 branch isolation, exact exception restore, scalar
normalization sign, stock-vs-B1 denominator, independently matched wrong direction,
neutral/nonfinite/support rejection, reconstructable FP32 addition/BF16 cast.
One1800s serial allocation, max16GiB new artifacts, free disk>=8GiB, cumulative
GPU cap=null; receipt includes actual imports/load/fail/replay/finalization. No
separate model subprocess wrapper; CPU pin checks accounted separately. Root
handles failures in a new isolated version if needed; single Luna max does one
read-only process/log/receipt/disk check, no predictions/GT or GPU action.
