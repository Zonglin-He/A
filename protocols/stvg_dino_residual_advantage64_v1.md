# DINO advantage on E's nonlocal/scale/instance residual errors, cache diagnosis64

User-authorized finite offline investigation,2026-10-19. Not a new OPD method or
parameter search. Frozen TA-STVG cross-domain source, E+finalMBR and four original
DINO observations unchanged. No new expert, model, GPU forward, training,
backbone, image decode, online history, optimizer or local-OPD hyperparameter change.

## Sources and exposure

Use all existing NH16/NV16 and CH16/CV16 exposed development parents (64 unique
parents, one query each). Baseline is E, not Source or Fine-Live. No FH/FV formal
labels or outcomes used to select this analysis. NH/NV32 additionally have existing
aligned conditional-DINO center/24-state/native900 traces. Do not fabricate missing
traces for CH/CV or unsupported NH/NV. Keep all64 and unknown/no-reference cases.
Inventory source fields, bind hashes and seal GT-free extracted geometry/readouts
for every parent before this analysis reads annotations. GT then used offline only.

## Comparisons and fixed diagnostic strata

At each original planned physical frame compare E, the original retained candidate
pool (up to3 boxes including below-threshold alternatives), original admitted
Top1, native900 highest target-score candidate satisfying existing legality and
score>=.35 (available subset only), and cached aligned T(E) if admitted. Original
frame admission and E selection are recorded, not changed. Raw pool capacity,
admitted support, confidence rank and actual readout must be separate quantities.
GT-best in a pool is a labeled diagnostic, never an online choice.

GT-known E residual strata (overlap allowed, not identity labels):
- bad E: IoU<=.3;
- far: bad E and max(|dcx|/GTw,|dcy|/GTh)>=1;
- scale: bad E and max(Ew/GTw,GTw/Ew,Eh/GTh,GTh/Eh)>=2;
- scale_near_center: scale and max normalized center error<=.25;
- instance mismatch evidence, Vid only: target IoU<=.1 and IoU>=.5 with a
  DIFFERENT annotated VidOR object at the same physical frame. This is evidence of
  overlap with another annotated instance, not a semantic/relation explanation.
HC has target-only metadata: missing distractor identities remain UNKNOWN.
Report continuous IoUs/displacements/ratios too. Recovery: teacher IoU>=.5 and
teacher-E>=.1 (10IoU pp), fixed for diagnosis, not deployment. Do not sweep cutoffs.

At known frames report parent-macro gains, recovery incidence, GT-best candidate
coverage, teacher top-score realization, rejected-frame opportunities and harms.
Report exact counts at frame/query levels; repeat candidates do not become
independent parent videos. Conditional subset and all64 never share denominators
implicitly. Four cached observations do not establish teacher ability on unobserved
frames. Classifying far or scale error does not prove wrong identity.

## State refresh evidence

On NH/NV compare cached T(E) with T(b_k) for each actually evaluated legal sampled
state at the SAME physical frame, same high-score content, aligned mask and .35.
Use raw and common-admitted comparisons separately; quantify newly admitted/lost
feedback, output similarity, target-IoU changes, newly correct recoveries and harms.
Match samples through candidate_indices/inverse into exact cached initial references.
A useful changed output is not automatically a need for on-policy refresh: check
whether it actually improves over T(E), and whether fixed native900 already contains
it. A deterministic fixed teacher may still benefit on-policy; do not demand new
teacher knowledge. These24 states were sampled around initial E, NOT an actual
updated-student multi-round trajectory. This diagnosis can establish state dependence
or a potential opportunity, not a measured Live-over-Fixed OPD gain.

## Complete tube scope and readouts

Seal simple cache readouts before GT: E; original admitted retained Top1 at planned
nodes; native900 top-score .35 at available planned nodes; conditional T(E) at
available admitted planned nodes. All other actual nodes retain E; time/width-height
outside changed anchors/interpolation unchanged. These are deployment-free cached
readout counterfactuals, not newly trained or adapted predictions. GT-oracle boxes
are used only for local coverage diagnosis, not promoted to a method or a full-tube
oracle. No teacher trace on unobserved frames is invented. Complete official vIoU,
ALL-known unobserved actual-node sIoU and local observed IoU are reported separately.

Parent bootstrap10k seed20261018; panels and datasets separate plus combined
exposed-development descriptions; conditional unknowns/nulls kept. Publish anonymous
signed per-query/frame results, coverage, instance-evidence limitations, trace lineage,
all negative findings and costs, excluding identities/text/images/rawboxes/labels/
weights/caches. Independent scalar geometry and full metric/aggregation audit.
CPU-only <=900s per finite phase using original resource guards; initial inventory,
GT annotation IO and audit counted separately. Production/current/formal64 files
hash-protected. Any data-format correction preserves initial failure and sealed data.
No automatic new model run, confirmation, parameter calibration or exploration.

Pre-GT inventory amendment:10 Vid parents (5 NV/5 CV) have no original DINO
observation receipts at any planned slot. Keep all40 slots as missing teacher
evidence, not detection failure or negative feedback. First18 extracted records
and initial KeyError preserved; canonical extracted002 adds explicit availability
and empty non-evaluated pool, no invented boxes/scores or new model calls.
