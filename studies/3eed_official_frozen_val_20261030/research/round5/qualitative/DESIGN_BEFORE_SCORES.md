# Full official validation: descriptive errors and local figures

This design is written before round5 prediction/score access. No new TTA, losses, training or parameter search is included. All full-caption and noun-only predictions must first be sealed by the source team, followed by a parent-authorized CPU score release. The figure selection consumes the complete scored table after that release; it cannot change prediction selection, inference, or metrics.

## Coverage and exposure

The independent filename/header audit found 8,620 available officialVAL frames and 9,706 raw ground_info objects: Waymo 2,708/3,794, Drone 2,984/2,984, Quad 2,928/2,928. This is a raw expression count, before the original loader's GT-class synonym-span eligibility test. Every raw expression must either produce a prediction or retain a stated eligibility/error record; never silently remove a failed sample. The official loader traverses all ground_info objects; earlier ground_info[0] pilots are not an exhaustive official benchmark.

Waymo has 93 available scenes of 97 listed. The four absent scene directories have zero members in the verified official archive. Their unknown frame/query count cannot be invented or included in a numerical missing-frame denominator. All available frames have point/header byte consistency, RGB headers and finite camera matrix shapes. The full release has RGB even though earlier selective pilot downloads omitted it. RGB is for qualitative diagnosis only: the evaluated model consumes XYZ, so colors/clothes in captions remain unobservable from its direct input.

Known previous VAL frame exposure is Waymo118, Drone44, Quad45; previously used scene exposure covers76/93 Waymo scenes and all5 Drone/Quad VAL scenes. The remaining17 Waymo scenes contain303 frames absent from known previous scene registries. P0's four CPU validation-item checks did not preserve source-scene mapping. Therefore this is a complete officialVAL descriptive evaluation, not a fresh preregistered confirmation study. See coverage/EXPOSURE.json for exact path-based provenance; exposure does not imply every expression in a historical multi-expression frame was scored.

## Primary figure selection (proposed for parent freeze)

After the parent chooses the official primary frozen full-caption readout and freezes it, select one case per platform and each following disjoint tag, up to12 primary cases. Selection within an eligible stratum is the smallest SHA256("3eed-round5-qualitative-v1|"+sampleID); it is not chosen by attractive image appearance or desired method outcome. Strata use the same native IoU conventions as the parent scorer. For each platform, report denominator and empty stratum explicitly. No target labels determine inference, sampling, calibration or PCA.

1. Success: full selected targetIoU >0.5.
2. Annotated same-class other-instance error: full selected targetIoU <=0.25 and its maximum IoU to an annotated same-class nontarget >0.5. This is evidence of overlap with another annotated instance, not an exhaustive proof that every other object was labeled.
3. Localization boundary: 0.25 < selected targetIoU <=0.5, full own-pool oracleIoU >0.5, and no annotated same-class other overlap >0.5. The tag describes boxes; it does not prove the semantic instance was correct.
4. Candidate coverage failure at0.5: full own-pool maximum targetIoU <=0.5, after excluding the other-instance tag by priority. The threshold is task-specific pool coverage, not proof that the object is absent from the point cloud.

If a query matches multiple failure tags, apply priority2→3→4. Queries in other failure patterns remain in the quantitative error table, not forced into a named stratum. Do not replace unavailable strata with hand-selected easier examples. Additionally the parent may request a fixed worst-meanIoU-delta full→noun tail per platform; this would be explicitly supplemental descriptive selection rather than a primary representative case.

## Paired language intervention and pool validity

The full caption and noun-only conditions can change both query features and candidate geometry. Compare selected targetIoU, condition-specific oracleIoU, rank of best-overlap proposal and annotation-other overlap for each pool separately. Query index equality is not instance identity; detached Hungarian geometry assignment is only a proposal correspondence. A full→noun prediction difference alone does not establish correct language grounding. It can arise from changes to token-softmax length/denominator, the cross-conditioned point/text backbone, candidate generation or classification. Preserve matched official readout controls and record fixed/dynamic padding policy from the frozen source plan.

PCA, if retained, is an explicitly offline geometry diagnostic: canonical source selected query, center and size fixed, source crop fixed, standard axis-aware orientation rule and degenerate fallback frozen from round3. It must not silently replace the main native yaw0 evaluation. Score both native and PCA boxes and show geometry-oracle analyses separately. No ground-truth fit is a method prediction.

## Per-case local figure

Use the same physical coordinate/camera convention in every panel. A two-row figure shows: RGB with GT/full/noun projected wireframes; ground-plane BEV with raw sampled points and the same three boxes; 3D view with common limits; and full-vs-noun selected/oracle/rank scalar table. Put the complete original caption and exact noun-only model input underneath without truncating the referring context. Explicitly label whether RGB was model input (here false), old exact-frame/scene exposure, source/native/PCA and the selected-readout name. Include a separate compact pool panel only when needed to show a recoverable alternative and its rank; do not hide failed proposals.

Camera contract: undo source Waymo suffix rotation/Drone z offset, then use the documented Waymo camera→vehicle inverse plus camera-axis conversion; Drone/Quad keep their verified matrix convention. Use real image headerW/H, not2cx. Before drawing verify finite positive-depth projection and clip wireframe segments at the near plane; do not connect corners across negative depth. Failed calibration becomes an explicitly unavailable RGB overlay while BEV/3D is retained, not a guessed mirror/sign. Source points/candidates/GT must use one declared frame. Calibration geometry does not prove caption-camera semantic association.

All media, original captions, numeric GT and source IDs stay in local private outputs. Public export may contain code, aggregate scalar tables, hashes, synthetic fixtures and appropriately redacted figure selection receipts. No original image/point cloud/caption/GT is pushed without separate user authorization and license review.

## Quantitative tables accompanying images

Report all-platform and per-platform official eligible/raw query counts, missing scenes, inference errors, class/span eligibility exclusions, native Acc25/Acc50/meanIoU, full vsnoun paired deltas, own-pool oracle coverage/topk, and counts of named error strata. Break down known exact-frame exposure, previously usedscene/newframe, and unknown-prior scene groups. Retain positive/negative language tails and do not infer a universal mechanism from a small figure gallery. The full per-query table remains authoritative.
