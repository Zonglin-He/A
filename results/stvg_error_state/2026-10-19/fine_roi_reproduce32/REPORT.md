# Fine ROI scoped OPD32: actual same-cache deterministic reproduction

The original exact combination was already completed in848b88d1. Following the repeated execution request, this run rebuilds the teacher templates and executes all three two-round fits plus direct readout in a new directory. Same NH16/NV16, four cached DINO, rawlayer3, orderedROI3x3, original scoremean, E/finalMBR, original noise/permutation, two17/tau.1/step.5/scale.25, pure interpolation scope and fixed planned/width/time. No model, expert, DINO, training, GPU, annotation loader or newGT scoring.

All32 contexts and actual96fits/192rounds reproduced exactly. Recursive comparison covers candidate feedback/probabilities/gradients/state/readout including the second Live sampling center. Maximum exact tensor difference 0.0. New output barrier precedes reading old saved aggregate metrics; all9 outputs per query match, so original signed/unknown metrics are inherited byte-for-byte, NOT newly scored or independent evidence.

CPU rerun with input pinning/comparison 87.525702s (old cache generation and root audit/publication excluded). Independent NumPy probability/gradient/state/readout error <= 1.11e-16; original independent ROI audit remains separate.

## Original inherited development results, changes vs E in pp

| Arm | HC fullv | Vid fullv | HC ALLunobserved actual-s | Vid ALLunobserved actual-s |
|---|---:|---:|---:|---:|
| Coarse_live | +0.079164 | +0.116067 | +0.149538 | +0.163168 |
| Fine_direct | +0.121182 | +0.144941 | +0.224729 | +0.065812 |
| Fine_fixed | +0.199084 | +0.141773 | +0.369208 | +0.280322 |
| Fine_live | +0.207571 | +0.145613 | +0.381899 | +0.292150 |
| Fine_shuffle | +0.132418 | +0.038628 | +0.213956 | +0.150695 |

All original32 signed/unknown cases and descriptive paired intervals remain in RESULTS_PUBLIC.json. Same inputs, same seed and code are not independent scientific confirmation. Original FineLive−Fixed completev +.008487/+.003840pp, both actual-s intervals cross0; FineLive−Coarse allfour intervals cross0; FineLive−samebudgetDirect fullv bothintervals cross0. Preserve old positive development evidence and later a6fb38e2 parent replication (HC positive vsE, Vid not replicated). Scope protection is not new visual information.

## New user-directed dataset-specific candidate, not evaluated by this rerun

The human explicitly withdrew the extra requirement that both datasets share identical numerical hyperparameters. Freeze HC step1.0, Vid step.5, temperature.1 and all other FineCenter settings in configs/stvg_fine_center_dataset_candidate_v1.json. This is a research candidate for future locked evaluation, not a new deployment registration and not proof HC step1 superiority. HC step1−.5 old/new fullv point means +.041681/+.083969pp, both intervals cross0 and gains concentrated. Vid new-parent difference −.041980pp, so retain.5. Both panels now participate in parameter selection and must be disclosed as target development evidence, not independent final test.

Adaptation remains GT-free; offline dataset-level target-development GT selection is permitted and disclosed. No per-query GT selection, final-test-based parameter switching or unknown-dataset automatic fallback. Old common-parameter gates remain historical protocols, not future requirements for method generality. More-complete-fit negative results remain unchanged. No new prospective evaluation started. This exact cached rerun uses originalstep.5 in both domains and cannot validate the newly frozen HC1 setting. CURRENT/MAIN/PAPER and heartbeat remain unchanged; no automatic expansion.
