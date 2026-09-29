# C2.5 transfer, conditional one-step C3, independent three-seed C0.6

Authorized by user attachment a2fe2a5b, registered before new outcomes. Original
historically exposed VidSTG32 parents/one query each, same Vid-source TA checkpoint,
same pixels/operators, 1/5/10%, observed grids and precision. No fresh test claim.
Use research-idea-iteration workflow. No spatial, reliability gate, preservation,
R16, optimizer state, multistep or parameter sweep. Production unchanged.

## C2.5-T first
Original first16 parents x15 seed0 transient conditions=240, plus16 exact reused
clean controls. Seed0 is the existing C0.5 realization. Capture final six-layer
and final legal-span candidates using the unchanged C1 generator (max8).
Require bitwise equality of all new native boxes/logits/preprocessed inputs to
sealed C0.5. UniversalVTG checkpoint, PE encoder, nearest-observed 2fps input,
confidence*proposal-IoU bridge and native-first ties unchanged. Cache once per
distinct pixels; exact pixel/candidate matches can reuse clean evidence.
Freeze original C2 margin tercile boundaries .01579105688924609/.06899146198758946;
do not recalibrate on transient data. GT only after prediction/score seal.
Primary average15 corruptions within each of16 parents; bootstrap10000 parents
with seed20260929, original ordinal order. Report Native/Expert/Oracle/Uniform,
pairwise accuracy (GT ties excluded, critic ties .5), clean/corrupt excess,
negative tails, and candidate support. Gate inherited from C2: corrupted tIoU
gain CI lower>0 AND high-margin pair accuracy CI lower>.5. This is a finite
resource decision; same-turn C3 is authorized only if it passes.
Input-only strata d_obs=actual changed observed RGB frames / observed frames:
zero, (0,.05],(.05,.10],>.10. Record parent/cell counts. Report no-hit changes
against native AND equality with clean rerank. Do not implement no-hit abstention;
the requested no-op ideal is a measured safety property, not silently added gate.

## Conditional C3-T (all same256 cells including clean controls)
Frozen and Rerank exact reuse. Hard and Preference each start at same H0; one
gradient each, only H_motion, H_app/text exactly zero. Native H exact replay
recomputes TTS/selection/ASA/query/both decoder passes each evaluation; preserve
official detach Jacobian. Backbone captured once per episode, never per trial.
H is held only in RAM; no large persistent feature corpus.
Hard target is the same critic-selected STUDENT interval. Map its physical
interval into each offset using first/last included observation, nearest endpoint
only when empty. Use unchanged native temporal Gaussian loss sigma/source weight.
Preference uses same deterministic endpoint map for every original candidate:
ell(I)=mean_offsets(log p_start(s)+log p_end(e)). Mean softplus(-(ell_i-ell_j))
over all strict critic-ordered pairs; ties<=1e-12 ignored. No teacher softmax,
temperature, beta tuning or margins weighting. Save endpoint-map collisions.
Candidate set and expert preferences stay fixed during this ONE update; final
output is the updated model's native decoding, no additional reranking.
Normalized step norm=.004*||H_visual|| (exact Round2 2*.02/10 nominal step),
alpha=[1,.5,.25,.125], first own-loss decrease >1e-8*max(1,abs(initial loss)) wins.
No accepted descent -> no-op. No GT in loss, step, acceptance or candidate choice.
Report Frozen/Rerank/Hard/Preference, paired Preference-minus-Rerank and Hard,
tIoU/vIoU harm tails and same d_obs controls. Positive Frozen gain alone does
not establish adaptation value. Continue this OPD recipe only if BOTH primary
Preference-minus-Rerank tIoU/vIoU CIs have positive lower bounds; otherwise stop
this recipe, retain successful cases and do not make universal OPD impossibility
claims. No C4 even if mean gain with harms. Audit full reinsertion on first clean
and first transient cell for both arms; all zero-H replay exact against C2.5.

## Independent C0.6 benchmark estimation
Same32 x5 x3 x3 seeds=1440 corrupted cells. Seed0 exactly reuses480 old C0.5
predictions. New realizations1/2 use seed namespace DeploymentBurst-20260929|seed=N|
plus source identity, with same uniform-start arithmetic; operators' own seeded
angles/occluders unchanged. Freeze donor determined by physical start, not just
observed mask. Reuse only identical pixel hash, never only a freeze mask.
All seeds and outputs retained; no resampling to create harm. Average45 cells
within each parent then bootstrap32. Report each seed/family/severity, d_obs,
no-hit exact controls, within-parent seed variance, between-parent uncertainty,
and >5pp harms. Three-seed CI need not shrink; do not promise improvement.
C0.6 is not used to design/select C3. GT-independent generation and no adaptation.

## Accounting and deliverable
Serial single GPU; C2.5 capture+critic cap3600s, C3 cap3600s, C0.6 cap3600s,
8GiB new artifacts, 8GiB free floor. Reuse sealed outputs; no model downloads.
Preserve failed allocations and engineering revisions. CPU math/gradient tests,
dual task metrics and independent anonymous-scalar readback. Update archive,
publish sanitized implementation/config/all scalar results and verify remote.
