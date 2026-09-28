# External-guided privileged OPD: qualification before adaptation

Status: implementation/qualification in progress; OPD efficacy untested. User attachment 2026-09-28 redirects the research and cancels full-source v3 training. It does not establish teacher superiority.

## Question and preserved controls

Can a released STVG specialist supply useful localization evidence, and can the same frozen PTD model turn this evidence into better temporal/spatial predictions without changing the action space? The old B1 source candidate is weak, and final-head FP32 supervision harmed native tubes on the source panel. These negative controls remain valid. We retain the dual spatial/event reader skeleton; the cancelled 35-step v3 checkpoint is an incomplete source-preparation artifact, not a qualified teacher.

## Stage A: external specialist qualification, no optimizer

- Official LLaVA-ST code `appletea233/LLaVA-ST`, commit `bacf6d61e1de27a78fc0025083aa445d3d41b0b3`.
- Official `appletea2333/LLaVA-ST-Qwen2-7B`, revision `2f7261b55b0368e9c2971120b62c77545b3a792d`, all LFS hashes checked before use. Its complete checkpoint includes SigLIP weights; no second untrained/replacement vision encoder is allowed.
- Isolated official Transformers commit `1c39974a4c4036fd641bc1191cc32799f85715a4`, retaining installed Torch 2.7/CUDA128 for RTX5090. No downgrade of the PTD runtime. Inference only, full FP16 teacher, SDPA instead of unavailable FlashAttention2; execution distinction recorded, not a claimed bitwise reproduction of the paper.
- Fixed existing source-training PANEL16, one query per Vid parent. First list entry is the engineering smoke, then all 16 without substitution. Source training exposure and possible teacher pretraining overlap are explicit; this is source qualification, not held-out generalization. No target inputs or GT.
- Decode exactly the same original RGB frames used by the PTD panel. LLaVA-ST requires 100 fast positions and 20 slow positions. Repeat the nearest observed frame at each of 100 linearly spaced *observed index* positions; introduce no unseen pixels. Save the entire index-to-physical-frame mapping. Predictions are mapped through this piecewise physical-time function, never by assuming equally spaced original timestamps. This is a matched-evidence interface adaptation, not the official benchmark's 100 unique-frame evaluation.
- Use the official STVG prompt pattern, greedy generation, max_new_tokens=1024, fixed seed20260928. Full-frame SigLIP resize, no crop/padding coordinate offset. Raw generated token IDs/text always saved, including empty/invalid/truncated outputs. No GT-derived prompts or teacher support. No sample retry based on quality.
- Parse only explicit time tokens and time-box pairs. Preserve malformed/duplicate/out-of-range/invalid geometry flags. Missing teacher evidence produces an unchanged PTD view and a recorded fallback; do not discard the episode.
- A finite initial engineering allocation is 900 seconds. Later full panel allocation at most3600 seconds, serial GPU, 8GiB disk floor, cap=null. All initialization/failure/overhead is added to39977.51307785203 seconds. Do not run cancelled training.

## Stage B: same-PTD evidence views, no optimizer initially

Frozen official PTD4B and B1 are named separately. For the existing B1 control, replay original inputs and native predictions before reuse. Compare observed, temporal privileged, spatial privileged, and combined views with fixed transformations. All retain original frame IDs, image dimensions, ordering and absolute box coordinates. Temporal outside-interval dim factor0.25; spatial outside-box Gaussian blur radius8 pixels, inside-box original observed RGB, no crop/resampling. Coefficients are fixed before new outputs, not selected using historical target results. Corrupted episodes, if later authorized, construct every view from that same corruption, never a clean counterpart.

New whole-panel predictions and physical-input/support manifests are sealed before independent source-GT scoring. Use parent-macro v/s/t, paired parent bootstrap CI, native-good retention and all >5pp negative tails. Sparse external boxes are interpolated only within their explicit time support; absence is not silently extrapolated to the whole clip. Temporal mapping uses physical frame IDs. Publish support and mapping rules before inference. The first engineering smoke does not establish qualification.

## Stage C: native branch positive controls and conditional OPD

Only if evidence views show useful source branch evidence, register native temporal/spatial positive controls on the fixed source panel. They must use actual native PTD supports and on-policy student reference/time prefixes. Source GT is restricted to explicitly supervised positive controls and offline evaluation, never the unlabeled function.

Only then qualify same-PTD teacher recomputation on the student's actual prefix and test vanilla/temporal/spatial/dual reverse-KL in matched, separate registration. External LLaVA tokens (100 spatial/time bins) are never aligned directly by KL to PTD tokens (1001 coordinate bins and actual T). External box pseudo-label supervision is not mislabeled OPD. Freeze gates for pre-gate objectives. Do not call decreasing loss native tube benefit.

## Decision scope and resources

If external evidence is poor, record qualification failure and consider the already available STVG-R1 specialist only under its own interface check. If external evidence is useful but same-PTD views fail, the privileged-view hypothesis weakens; do not begin OPD merely because the external model scores well. If branch controls fail, examine native support and actuation rather than resume full-source training or sweep learning rates/gates/steps.

InternVideo2/VideoMAE frozen priors, source-feature caching and parent-balanced source preparation remain contingent later options. Any cache must be keyed by physical pixels, frame IDs, preprocessing and query-dependent geometry, not parent ID alone. No new full-source fit, automatic target/64 expansion or CURRENT promotion. The deleted heartbeat remains deleted. User subsequently authorized timely GitHub publication of code/structure; publish reviewable code and aggregate status only, no weights/data/private raw evidence.
