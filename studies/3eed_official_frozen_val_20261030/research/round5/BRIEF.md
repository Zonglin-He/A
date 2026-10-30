# Round5: official frozen 3EED baseline

Claim and boundary: reproduce released union-checkpoint frozen inference on the complete publicly available official VAL population of Waymo, Drone and Quad. No source-only, training, parameter adaptation or novel TTA efficacy claim. Preserve previous cancellation.

Evidence: previous round4 demonstrated candidate rank gaps only on subsets; official evaluation uses GT-class synonym span assistance, while previous studies used text-only first-noun readout. Official release lacks four Waymo VAL scene directories.

Minimal unresolved question: what are complete-population official-head accuracy/coverage and label-free text-only differences under a fixed single-GPU evaluation configuration, and which errors survive candidate reranking?

Next action: GT-numeric-free caption/input inventory and deterministic 16-expression per-platform smoke, freeze and seal candidate outputs, CPU score and audit, then all-VAL frozen evaluation with identical fixed configuration. All raw ground_info expressions are forwarded; official GT-class eligibility is applied only in disclosed offline evaluation, alongside all-raw text-only metrics.

Measurements: last BBF/BBS Top1/5/10 and 256 oracle Acc@.25/.5, last Top1 mIoU, proposal metrics, legacy seven-head mixed mIoU if source bridge passes, latency and memory with explicit windows. Full-caption vs text-derived first-classword inputs use exactly matched points and no target class fallback. No PCA in official primary; geometric annotation interventions are post-seal diagnostics only.

Data: old research exposure disclosed. Full official split replication is not a newly untouched confirmation set and cannot be used to claim new held-out method selection. Scores and diagnostics do not change the frozen model/configuration.

Reproducibility: immutable finite plan/code/input/checkpoint hashes, seal-before-numeric-GT, independent geometry scoring, full missing/skipped/failure counts, scene-cluster summaries, fixed qualitative strata, retained negatives. Export only public code/protocols/scalar evidence; no raw media, captions, labels, weights, points or embeddings.

Resource boundary: reuse local environment and datasets, B1 inference only and bounded CPU postprocessing; one serialized GPU owner, no changes to unrelated work or ToDesk. Full finite VAL evaluation is explicitly user-authorized; smoke will establish its measured time estimate. No training automatically queued.
