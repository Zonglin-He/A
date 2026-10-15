# Evidence-constrained spatio-temporal TTA: last bounded algorithm iteration

Authority: latest human attachment 3037c4a2, accepted 2026-10-15. The previous no-Bootstrap instruction persists. This is research; CURRENT is protected. No new detector, segmenter, DINO, YOLO or SAM worker is introduced.

## Data and evidence

Use the existing32 coordinate-development parents (16 per direction), with explicitly prior label exposure. Exclude previous64 coordinate parents and original8 route parents when selecting a further32 parents for confirmation. These sources are excluded from this algorithm selection, but are not claimed globally unseen. No test-time optimization, candidate construction, prompt, reference, update validation or acceptance reads GT. Each complete phase seals every arm/source before a separate official/independent scorer reads GT. Report all positive/negative/neutral/null results and exact parent pairs; no Bootstrap.

## Round1

Same frozen source features, original Caption–Query feedback, temporal candidate pool and fixed original spatial evidence/output. Compare Projection, original five-step Adam, stable original objective, and offset-aware full-grid-trust objective. Stable and trust variants use positive initial step, bounded L-BFGS with deterministic descent search. Trust: sum offset candidate KL plus lambda_grid=1 times sum KL(p0_grid||p_theta_grid) and L2=.001; tau1, at most50 accepted iterations. No-information and true zero-gradient queries may no-update. Record actual updates, objective monotonicity, each offset interval/jump, grid KL, output decode and parameter norms. These constraints do not correct an erroneous teacher ranking.

Only stable and offset_trust are eligible for the revised temporal algorithm. Select one globally by the equal-weight mean of the two domain means of full T+fixed-S vIoU on development; tie chooses offset_trust. Projection and original are controls, not hidden fallbacks used to report fake parameter learning. A weak result remains weak; no extra parameter grid is opened.

## Round2

Freeze Round1 temporal. Keep final previous spatial config per domain. Factorial arms: old_crop, new_crop, old_scene, new_scene. Old clock is original QA interval; new clock is adapted final interval. Each clock gets four eligible fixed grid centers; crop and scene share identical candidate geometry, mask arrays and ID mapping. Candidate algorithm remains SelectiveSearch256/GrabCut24/DIS, up to8 new candidates plus Source. Old crop evidence can be reused only after exact source/input/interval checks. Full-scene input changes scene representation/prompt explanation; WHO task and abstention stay fixed. No proposal-completeness redesign in this matrix.

Select one complete arm globally by equal domain-mean joint vIoU after full seal; ties prefer new_scene, new_crop, old_scene, old_crop. Report all four as matched controls. If chosen arm still has at least4 development sources where accepted reference agreement increases but common-support unsupervised sIoU declines by at least1pp, allow one independent fixed-reference Output-Space Preservation ablation (lambda1, no sweep); record this conditional development-GT exposure. No further proposal/prompt/parameter searches follow.

## Freeze, confirmation, Table1

Freeze code and one recipe before new32 confirmation. Confirmation cannot reselect or tune parameters. Then proceed to the authorized official13785-query clean Table1 using the frozen recipe regardless of whether it reaches old8-case performance. Engineering failures can be repaired under separate captured revisions; scientific thresholds do not reopen search. Single GPU child, own-cgroup resource guard, resumable per-query seals, parent/video-safe caching, full prediction barrier before official scoring. Native YOLO/SAM waiting queues must not resume. HC official validation and Vid official test retain their distinct names. Component controls F/T/S/TS use actual redecodes, no three fake independent order repeats for episodic inference.

Table1 throughput and bounded reconstructible cache policy must be measured on the frozen confirmation before launching all queries; changing teacher output limits is a protocol change requiring matched evidence. Runtime/source/receipt pins and all actual costs are recorded. External teacher training overlap remains unresolved and disclosed.
