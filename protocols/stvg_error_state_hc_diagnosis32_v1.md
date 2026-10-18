# Why HC oracle correction and actual DINO differ

CPU-only reanalysis of sealed error_state32, focus HC16 with Vid16 context.
Fixed original Before start, Source sensitivity, same Source final-MBR, original
Native Uniform4 positions and DINO references. No training, model, expert,
pixels/video reread, optimizer, parameter update, method promotion or new TTA.
Old deployment and all failure/positive results remain unchanged.

## Three distinct estimands, locked before new diagnostic scoring

1. **Reference categories**: reuse old model-assisted VISUAL_REVIEW only after
query/parent/pixel-panel hash binding to the identical rollback32. CI is apparent
consistent identity, not GT geometry certification; CP partial/uncertain extent,
WI visibly wrong participant, MX merged instances, UN uncertain. If the old
query has semantic_role_unknown, override every slot to conservative UN for this
analysis, retaining original codes. Never derive identity from IoU. No new labels.
On the same exact-GT common positions, replace each category's DINO boxes by
their cached sparse GT. Evaluate all subsets of categories present (at most four
per query) and exact finite Shapley contributions to GT_common minus DINO_common.
These are category-conditioned oracle repair effects, not additive real-world
causal percentages or validated automatic filters. Retain all16/domain, and
report category-present counts and conditional means as secondary descriptions.

2. **Geometry**: on exactly the same common references, use four hybrids:
DINO center/DINO size, GT center/DINO size, DINO center/GT size, GT center/GT
size. Same constant-error LS and global scope. Exact two-factor Shapley separates
center-coordinate and log-size effects. A wrong person changes both, and a
partial box can also shift its center; center is not identity, size is not all
extent error. Describe CI/CP/MX/UN separately; tiny classes cannot support broad
identity/extent conclusions. Save per-anchor IoU, target-relative center error,
log-size error and exact sum of category contributions to delta_DINO-delta_GT.

3. **Application scope**: keep the estimated delta unchanged. Apply it globally,
only inside the original four-planned-frame physical hull, or only at original
admitted decoder nodes. Masks do not use GT; they are fixed diagnostic controls,
not new deployed smoothing. Apply each to DINO_full, DINO_common, GT_common.
Changes in benchmark interpolation near mask boundaries are retained. Report
within/outside planned-hull actual unobserved frames and signed contribution to
the complete unobserved endpoint. Also distinguish incorrect global delta from
GT delta that is itself inappropriate for some unobserved positions. The old
LocalAnchor replacement is retained as prior reference, not confused with
same-delta node-only application.

## Label and execution boundaries

Use only already sealed sparse GT inputs for generating reference repairs. Seal
all diagnostic candidate outputs before this run's dense scoring; all GT use is
posthoc and already development-exposed. Original dense truth is parsed only
for scoring. No target labels enter a claimed deployment rule. References
without exact GT remain unknown, not wrong: compare old full-versus-common
outputs descriptively and disclose changed support. Outside-event reference
removal is not a valid automatic event/identity filter.

Primary endpoints: full vIoU and known unobserved actual-decoder sIoU, original
Before/final-MBR. Secondary all-known dense sIoU, within-MBR support, Source,
per-source tails and saved-score leave-one sensitivity (not online replay).
Keep all32 and exact no-op queries. Unknown supports are null, not fabricated
zero GT. Parent bootstrap 10,000 fixed seed20261018 is descriptive on the
inherited fixed cached trajectories, not an independent online confirmation.

Do not add identity, center/size and scope effects together: these are overlapping
controlled contrasts. No selection of a winning hybrid, group mask, time mask,
query subset or data-domain recipe is permitted. CPU cap3600s, new GPU/model/
expert/training/Student updates all0. Publish anonymous scalar evidence, negative
and unknown results, source/runtime hashes and limitations; no raw coordinates,
GT, media, q/tensors/weights or personal text. Complete archive and verified
GitHub synchronization without automatically opening a follow-up experiment.
