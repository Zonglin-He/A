# Reproduce the round6 Waymo-only supervised source run

This directory records a genuine source-only run using the released 3EED implementation and source Waymo train/VAL data. The primary is the fixed epoch100 model after 15,300 optimizer updates. Source validation every ten epochs is monitoring and can define a separately labeled best-source secondary model. Target-platform labels, TTA and the released mixed-platform task checkpoint are outside this run.

The historical plans and receipts are immutable evidence, not portable jobs to rerun under a changed workspace. Use a fresh workspace, regenerate its private source-data joins, register new local hashes and releases, and retain the original provenance beside the new replication. Public files intentionally exclude source annotations, original captions, physical scene/frame IDs, points, predictions, trained weights, optimizer tensors and RNG tensors.

## Layout and original code

Preserve the exported `crossplatform/round6` and `research/round6` relative layout under a new workspace root. The training factory reads an upstream checkout at `crossplatform/round3/p0/upstream`; this path is a historical directory name only. It does not import or execute the old cancelled P0 timing entry. Clone [3EED](https://github.com/worldbench/3EED) at commit `843a901f5367a6be5cc484b97c6ecad462961dfa` into that directory, then apply `UPSTREAM_COMPATIBILITY.patch` there. The patch only removes the obsolete unused `cgi` import and disables the unused optional Open3D visualization import. It makes no changes to the trained architecture, matcher, losses, optimization or point geometry.

The original repository contains a machine-specific `data/roberta_base` symlink. Replace that symlink only inside the fresh checkout with a local directory or a link to legally accessible generic RoBERTa-base assets. Do not copy the original symlink target from somebody else's machine. The factory locates those assets through `crossplatform/upstream/data/roberta_base`. A fresh-workspace alias `crossplatform/upstream` pointing to the same patched upstream checkout is sufficient; the independent source-data auditor also imports through that alias. The live study used two equivalent loader copies with a comment-only difference, recorded by the loader-equivalence audit.

The model's `models/bdetr.py` is the original training implementation. Its RoBERTa constructor calls `RobertaModel.from_pretrained`; no mixed task weights are loaded. Generic RoBERTa checkpoint/config/tokenizer hashes are in the formal plan. The two absent pooler tensors are initialized by the original Hugging Face loader, then frozen; grounding uses `last_hidden_state`, not the pooler output. Preserve the explicit initialization seed and verify the cold model/text fingerprints before running updates.

The original loader uses unsorted filesystem frame enumeration. Source join/order hashes therefore matter. The private annotation order is regenerated locally and is not included in this public package. A filesystem-order or software change requires a newly registered replication with its own order hashes; do not claim a byte-identical replay merely from setting seed0. No deterministic reordering was silently added to the original run.

## Environment and custom operators

The actual training environment used Python3.11, PyTorch2.7.0+cu128, CUDA12.8, NumPy1.26.4, SciPy1.15.3 and Transformers4.52.4. It reused an isolated environment without installing into other projects. Use a matching CUDA-capable environment and record the compiler, package and device versions. The original custom operators must be built for the local Python/PyTorch ABI and GPU architecture; the RTX5090 study used sm120. Compiled binaries are deliberately excluded from the export.

The three operator locations are `pointnet2`, `ops/teed_pointnet/pointnet2_batch`, and `ops/teed_pointnet/roiaware_pool3d`. Their source is provided by the pinned upstream checkout. The pointnet2 setup names its extension `pointnet2._ext` but expects its source-relative current directory. Building from that directory with a build output root one level above avoids the observed nested `pointnet2/pointnet2` inplace-copy failure:

```text
cd <fresh-upstream>/pointnet2
python setup.py build_ext --build-lib ..

cd <fresh-upstream>/ops/teed_pointnet/pointnet2_batch
python setup.py build_ext --inplace

cd <fresh-upstream>/ops/teed_pointnet/roiaware_pool3d
python setup.py build_ext --inplace
```

Set CUDA_HOME and the appropriate TORCH_CUDA_ARCH_LIST for the reproduction GPU. These commands describe the setup layout; compilation was not repeated during export preparation. Preserve upstream third-party notices and terms. Import success alone is insufficient: the locked twelve-step real forward/backward engineering check below establishes runtime and gradient compatibility.

## Source data and CPU preparation

Place the released data at `crossplatform/dataset_full/3eed_dataset`. Access and archive verification must precede execution. Only Waymo train/VAL are read by the source run. The actual source data has 3,682 eligible training expressions in 94 available scenes and 3,782 eligible validation expressions in 93 available scenes. Six train and four VAL split-listed scene directories are absent from the released data. Raw counts are 3,687/3,794; original class-synonym filtering excludes 5/12. Actual source train/VAL scene and frame intersections are zero.

Run the exported `research/round6/source_data_audit/audit_source.py` locally to regenerate private source joins, source data audits and view/positive-map bridges. Those generated private indices contain annotation references and numeric source GT; keep them local. The factory checks its loader sequence against these indices. CPU preparation opens source annotation containers; this is supervised source training, not an assertion that source labels were never opened.

Run `crossplatform/round6/cpu_preflight.py` to check CPU model initialization, group sizes, source order and the generic RoBERTa fingerprint. It prepares the new timing template but grants no GPU authorization. Run `audit_train_token_lengths.py` to measure actual source token lengths. The recorded training maximum is125, whereas the twelve timing batches only reached92; the formal first epoch covered125 without OOM. These are source-input diagnostics, not source-validation scores.

## Twelve-step engineering check, then formal training

A trusted local coordinator must write a new timing plan binding all local source, generic RoBERTa and new authorization hashes. The executable requires `GPU_execution_authorized:true`, `mode:timing12`, FP32 batch24, four warmup plus eight timed steps. Use one shared physical-GPU file lock across every concurrent project.

```text
python crossplatform/round6/time_training.py \
  --plan <new-timing-plan.json> --plan-sha256 <its-SHA256> \
  --output <new-engineering-directory>
```

The check initializes real rank-one NCCL and DDP with broadcast_buffers=False. It performs actual supervised forward, official loss, backward, clipping and AdamW updates; its weights are discarded. The live check averaged0.403511s after warmup and peaked at29,026,556,928 allocated bytes. Its initial cold startup was much slower than steady-state updates, so it is not a complete-run ETA.

Formal execution uses a separate registered plan with `mode:waymo_source_train100` and the active final `train_source.py`, not `train_source_v1_snapshot.py`. The v1 snapshot retains the epoch100 finalization edge bug for provenance only. The active source SHA is recorded in the formal lock.

```text
python crossplatform/round6/train_source.py \
  --plan <new-formal-plan.json> --plan-sha256 <its-SHA256> \
  --output <new-source-run-directory>
```

Training preserves the official FP32 batch24, 153 steps/epoch with drop_last, four workers, original three AdamW groups (including the empty frozen-text group), matcher class1/bbox0/GIoU2, original loss weights and topk4, clipping0.1, and training BN momentum0.1. The original scheduler is advanced after each optimizer update. Nominal milestones25/26 with warmup=-1 actually reduce LR after steps3978/4131, i.e. the ends of epochs26/27; the lower rates are first used in epochs27/28. No silent correction changes this behavior.

Global seed0 is an explicit reproducibility choice because the original rng_seed flag did not globally seed the entry. Worker spawn replaces unsafe fork after CUDA/NCCL initialization while retaining the original four-worker seeding and shared train/VAL generator. No AMP, gradient accumulation, frozen text-feature cache or reduced batch was introduced. Frozen RoBERTa weights still run training dropout under model.train().

## Checkpoints, validation and completion

Every ten epochs an immutable model file is saved before validation. The source-only validation loader uses batch24, 158 batches and a final14-expression batch. It retains original dynamic token padding, raw/evaluation-clamped 6D candidate boxes, source GT-class-assisted BBF/BBS, and text-first-noun BBF/BBS from the same full-text forward. It is not a separate noun-only model pass. No optimizer or scheduler step occurs in validation. Predictions are sealed before a separate CPU metric process.

The exported source scorer verifies program/index/model/prediction hashes and independently checks selected Top1 IoU with a general3D implementation. Scoring requires a new explicit CPU source-VAL release. The recorded metrics use yaw0 predictions, actual source GT7D float32 preprocessing promoted to float64, strict IoU>.25/.5, nativeGPU Top10 tie order, and separately labeled full-pool GT-assisted oracle coverage. Only last/proposal heads are cached; no seven-head legacy mIoU is fabricated for this run.

After each completed epoch, including its scheduled validation, `resume_latest.pth` is atomically replaced with model, optimizer, scheduler, Python/NumPy/torchCPU/allCUDA RNG and the shared loader generator. It is a mutable recovery pointer, not an immutable published artifact. Resume from that run directory with the same registered plan and `--resume <run>/resume_latest.pth`. Incomplete epochs are discarded; mid-epoch worker-state restoration is not claimed. Keep any interrupted partial write as forensic evidence before a manual recovery. CPU fixtures verify safe weights_only loading and an identical next AdamW update after state restoration; CUDA RNG storage/restoration uses the standard API and was separately code-reviewed.

Epoch100 finalization checks an existing same-plan epoch100 checkpoint and prediction seal against the actual restored model fingerprint. A completed epoch100 resume finalizes without further training or validation. `TRAINING_COMPLETE.json`, the final independent audit, all ten source-validation scores, final report and decision must exist before the public export builder will run. The primary remains epoch100; a best-source-VAL model is secondary only. No target-platform generalization or TTA benefit follows merely from successful source training.

The export contains code, plan/protocol projections, aggregate audit/cost receipts and complete epoch-level scalar history. Weights, optimizer/RNG payloads, private indices, captions, raw physical identifiers, point clouds and prediction arrays remain local. Every projection retains both the original-source SHA and exported SHA; sanitization does not change the historical plan's identity.

Audit helpers retain historical expected plan/program hashes. Their optional workspace-path portability edit is recorded in export provenance. A fresh replication must register its own source/program expectations and private-data joins under a new audited registration; do not bypass a historical hash check or rewrite an old lock to make it pass. Source code and model/loss mathematics remain unchanged.
