# Targeted reviewer experiments

Authorized on 2026-11-03. The current shared-corrected ROI method, same eight expert observations, kappa0/beta1/context1.5, one temporal update and final interval-conditioned spatial DP remain fixed. No manuscript or Fig.1 edits. No additional EdgeTAM/ReferDINO combination. Old queues and cancelled CG experiments are not resumed.

The user explicitly permits necessary GPU acquisition **after their existing GPU task finishes**. CPU work starts from frozen cached inputs. The separate full GPU owner waits for the CPU prediction-control suite, absence of foreign compute processes, and the global GPU lock. ToDesk is never signalled or counted as an experimental worker.

|Question|Execution and scope|
|---|---|
|A: Source-only temporal refinement|Both complete domains, 1902/10303 queries. Source ROI features + q0, own interval, original Source boxes, zero expert dependence. Separate fixed-E interval controls and an interpolation + Source-only TR baseline.|
|B: Spatial regions for temporal evidence|192 retained raw grids/domain first. Source box, shared-corrected box, all-valid global, outside expanded-E background, SHA whole-box temporal permutation. Same fixed E output trajectory. Independently frozen full-domain counterparts share one necessary Source acquisition per query with uniform-gamma controls.|
|C: Distance and event membership|Reuse audited per-frame IoUs. Query-equal comparisons on all annotated unobserved frames, current-prediction interval, and GT interval, with fixed physical-distance bins and counts.|
|D: Query form and expert quality|Exact metadata joins, fixed quality thirds and missing group, spatial and temporal comparisons separated. Offline association only.|
|E: Tent/SAR diagnostics|All existing saved logits, genuine update traces, norms, before/actually-after entropy, matched native/joint readouts. No new optimization or method retuning.|
|F: Evidence quality and temporal risk|Per-query GT event/background evidence means, tie-aware AUROC, Source tIoU groups, gain/harm, confidence/prototype separation summaries. Offline descriptive labels only.|
|Parameter origins and fixed strengths|S/initialization cutoff derive from source-training residual statistics; historical gamma used exposed target development labels. Both gamma=.5 and gamma=1 are independently tested in both full directions, not selected afterward. This is not resuming the old five-point full-gamma sweep.|
|Negative outcomes|Gain >1pp, loss <-1pp, severe <=-10pp; strong-Source strata, existing quality/form groups and all signed tails retained.|
|Corruption|Reuse all36 completed units/4608 repeated-condition views. Explicit clean-minus-corrupt degradation and paired difference of degradation; no new corruption inference.|

Plans: `outputs/reviewer_key_controls_v1/plan_v1.json`, `statistics_plan_v1.json`, and `full/plan_v1.json`. Scientific CPU entry: `run_v1.py`; statistical entry: `statistics_v1.py`; separate audit: `audit_statistics_v2.py` (v1's empty-stratum reporting error preserved). Full acquisition entry: `full_gpu_v1.py`. Root must actually run `root_gpu_bridge_v1.py` for each domain's first two new paid inputs while that owner is waiting; never synthesize acknowledgements.

Whole task/domain predictions seal before independent official CPU scoring; independent cold-array/SHA/mean audits precede results. Query-equal pointwise 10000 parent-cluster confidence intervals, not frame-independent or familywise intervals. All target sets were previously exposed. Positive ROI controls cannot by themselves establish action-semantics understanding, mutual-reinforcement causality, or universal superiority. A null interval does not prove equivalence. No new parameter/default adoption from these results.

Private media, captions, identities, GT coordinates, raw features, masks, weights and credentials are excluded from publication. Root review, all signed positive/negative reports and actual per-PID/start cost records, clean-export archive check/snapshot/check, fresh-CAS dual refs and independent remote byte/tree/ancestry verification are required for complete publication. Startup/partial results are never suite completion.
