# DESTA: dual-expert native-guided R16 adaptation

Status: registered implementation of the user's 2026-09-29 attachment, before new inference.

The prior LLaVA-ST pixel-privileged policy NO-GO and all internal direction
failures remain valid historical results. The user explicitly reopens a different
mechanism: two independent frozen specialists provide native pseudo-targets;
PTD supplies full-merger gradients; only an ephemeral shared R16 field changes.
No offline predictor, pixel transformation, model-weight update, Adam, or OPD.

## Data and experts

Use the existing exposed A0 Dev64 (16 parents, four queries each) and the exact
A0.5 SHA Dev16. Fresh31/388 and target inputs/labels stay untouched. Both experts
only see the existing decoded PTD observations and original full query.
UniversalVTG (fixed local best checkpoint and PE-Core-L14-336) is the temporal
specialist. Uniform physical-time 2fps slots select nearest existing observations;
all segments/scores are saved, highest score chosen once, seconds mapped to physical
frames and then nearest observed endpoint tokens. The spatial implementation is
the attachment's second-choice GroundingDINO-tiny + SAM2.1-large tracking, explicitly
not a task-trained RVOS model. Full caption, box .25/text .30 thresholds, maximum
detector-score anchor over all observations, stable earliest tie, one referent
propagated in both directions; masks >0 produce normalized xyxy. Missing boxes
stay missing. No GT pseudo-label repair. Provider training overlap is unknown.
Weights and runtime source hashes are saved before inference; providers are not
selected or changed based on native results.

## Native update

Freeze official PTD4B, B1, union256 and Train-only B16. Use exact fixed B1-native
reference/interval/anchors/trace across updates. Temporal CE averages two observed
endpoint actions. Spatial CE uses actual coordinate token IDs and all 152775
classes on available native anchors, mean over supported coordinates. No GT prefix.
Each step separately differentiates both losses with respect to full F, normalizes
in F space, then weights and projects into Q_C=Q_U B16. Missing branches contribute
zero. Normalize the combined projected direction; step size is rho*||F||/K.
Project C using the actual Q_C Gram norm to radius rho*||F||. Fixed terminal K;
no loss/best-step selection. Free original native decoding uses the same corrected
F in both passes. Final temporal support may change naturally.

## One automatic experiment

27 configurations: K={1,3,5}, rho={.03,.07,.135}, lambdaT/lambdaS={.5,1,2}, lambdaS=1.
All run on Dev16, seal then offline source scoring. Select top6 by parent-macro
vIoU, ties within 1e-6 pp by fewer query vIoU losses >5pp then lexical config ID.
Extend these six to Dev64 (reuse Dev16 outputs), seal and select best1 identically.
At that selected config run T-only and S-only on Dev64. Reuse saved B1 and GT-R16
after input/state identity checks; no-update is an actual zero-field identity
replay. GT-R16 is a source diagnostic, not an upper bound or a label-free method.
Invalid/zero boxes, missing pseudo-targets, positive and negative cases remain.

Report parent-macro t/s/v, paired parent descriptive bootstrap CIs, query harms
>5pp, B1-good retention. Hyperparameter influence comes from the complete Dev16
factorial matched comparisons, marginal spans and interactions; selected Dev64
results cannot stand in for a complete Dev64 factorial. Selection is explicitly
on exposed development data, not a fresh generalization claim. No success guarantee.

## Engineering verification and resources

CPU tests cover physical mapping, exact native coordinate IDs/full vocabulary,
full-normalize-before-project, zero/missing support and radius. First actual
native update replays B1 logits exactly at C0; all parameters remain frozen.
Save projected raw branch gradients plus full-F norms, each actual C update,
support and input hashes, exact pseudo-targets, terminal fields and predictions.
These sufficient statistics permit independent NumPy update reconstruction;
all enormous full-F gradients are not retained. First-case raw full-F gradients
are retained for an independent projection/order audit.

GPU stages are serial, each allocation up to 7200 seconds as engineering protection;
resume only at completed episode/config boundaries. Failures/partial outputs and
all actual loading, failed execution, worker and wrapper seconds remain accounted.
Cumulative cap=null; free disk >=8GiB and <=25GiB new artifact allocation.
No automatic full618/full447/fresh388/target/OPD. Code, protocol and anonymous
aggregates are pushed to the user's authorized GitHub repository after checks.
