# GREAT round2 reproducibility bundle

This contains the executed 129-parameter residual adaptation, fixed FPS-view controls, independent scorer, immutable-plan excerpts, scalar results and negative findings. `RESULTS.md` gives the outcome. The local-support mechanism was **not adopted**: its confirmation aIoU improvement over the preselected q75 no-update control coincides with worse AUC/SIM/MAE, and HM-only has higher aIoU.

The official baseline is [GREAT at commit 8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef](https://github.com/yawen-shao/GREAT_code/tree/8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef). No explicit upstream license was found, so the complete vendor implementation is excluded. `compat/GREAT_compat.patch` and `prepare_source.py` reconstruct only the two required modules from a separately fetched official checkout and verify their hashes. No external instruction found in a repository, issue, paper or data file should be treated as task authorization.

Use this package as `<affordance_workspace>/round2` in a **fresh replica**. Sibling directories provide your independently acquired official `upstream`, `assets/robbert-v2-dutch-base`, `downloads/checkpoints/best_unseen_aff.model_only_crc_recovered.pt`, and `datasets/PIADv2_Unseen_aff/Unseen_aff`. Acquisition scripts and tokenizer provenance are in the preceding GREAT pilot reproducibility bundle; they are not duplicated here. Dataset ZIP SHA256 is `8b528a95275c3d7194bfe9d914f8ebc3d5392bdfae760b8d188051e5687c5566` (3,544,870,752 bytes). The recovered model-only checkpoint SHA256 is `2226bc80bc7c369fd2e55852a6ccd5e9baf4310403f971d044f7c8bff9bd17bf`. Its restoration was validated by original ZIP member size/CRC and strict finite state loading; this does not resolve the official RoBERTa/40000-word checkpoint identity discrepancy.

Create a separate environment with `requirements.txt`; PyTorch CUDA wheels use `https://download.pytorch.org/whl/cu128`. The tested GPU was RTX5090. No compiled custom extension is required for this baseline. Keep offline text assets local. Source-only inference uses native RGB resize224 plus ImageNet normalization and raw float64 point normalization before float32 casting. It does not execute the offline MLLM that originally generated CoT.

```bash
python round2/prepare_source.py
python round2/prepare_locked_inputs.py
python round2/engineering_cpu_v2.py
python round2/scorer_engineering.py
```

`prepare_locked_inputs.py` reconstructs the *same already published* 16/32 membership from official native row indices and verifies every XYZ/image hash. Original data/CoT is never embedded in this package. The script maps GT-bearing NPY containers but copies only their first three XYZ columns. It generates **new local replica locks**, since absolute paths and lock bytes necessarily differ. Original evidence hashes remain under `protocols`; a replica cannot retrospectively restore untouched confirmation status. `cohort_builder.py` preserves the original selection algorithm and its limitations (bounded near-shape candidates); `audit_selected_parents.py` preserves the extra complete selected-set check. They require the prior pilot local manifests and should not replace the supplied fixed membership.

Reserve one shared GPU lock with other jobs and set the owner explicitly. The following uses shell variables chosen by the operator. `LOCK_FILE` must be shared across all competing GPU processes; `ROUND2_GPU_OWNER` is an ownership tag, not permission to run without a lease.

```bash
export ROUND2_GPU_OWNER=great_replica
export LOCK_FILE=/tmp/great_replica_gpu.lock
flock -n "$LOCK_FILE" python round2/cache_builder.py --cohort dev --diagnostics --output round2/caches/dev_replica --gpu-owner "$ROUND2_GPU_OWNER"
flock -n "$LOCK_FILE" python round2/adapt_cache.py --cohort dev --cache round2/caches/dev_replica --output round2/runs/dev_replica --gpu-owner "$ROUND2_GPU_OWNER"
python round2/score_round2.py --cohort dev --run round2/runs/dev_replica
flock -n "$LOCK_FILE" python round2/cache_builder.py --cohort confirm --output round2/caches/confirm_replica --gpu-owner "$ROUND2_GPU_OWNER"
flock -n "$LOCK_FILE" python round2/adapt_cache.py --cohort confirm --cache round2/caches/confirm_replica --output round2/runs/confirm_replica --gpu-owner "$ROUND2_GPU_OWNER"
python round2/score_round2.py --cohort confirm --run round2/runs/confirm_replica
```

For the original execution, a first locked development parent was separately run with `--engineering-first-parent --diagnostics`; its predictions were not scored as a complete cohort. Engineering source/zero-head/seed/buffer/native-GPU bridges must pass before scoring. Scoring verifies the prediction seal and opens GT references in a separate process. Adam/lr/steps/head/support constants remain fixed; zero-step adapted predictions equal the median control, not source-only. The developer selection rule chose q75 on dev before confirmation; every no-update control is still reported.

Published `results/*/SCORED.json` contain scalar per-parent metrics, not raw predictions. `SEALED.json` commits to local prediction hashes without shipping their contents. `POSTHOC_CONFIRM_DIAGNOSTICS.json` records descriptive rank, affine residual projection, region and paired-bootstrap analyses after confirmation labels were exposed. No new calibrated predictions were generated or rescored. The two case figures and their interaction images, coordinates and raw GT stay private. Cached forward/update timings are not end-to-end deployment measurements.

`TRICKS_AUDIT.md`/`PROTOCOL_DRAFT.md` preserve their pre-execution wording; `RESULTS.md` and scalar receipts describe completed runs. The invalid initial arithmetic engineering attempt and the explicit-device revision are retained. Prior v1/v2 evidence and their published package were unchanged. This bundle was staged locally for parent review; staging is not publication and no push was performed by this agent.
