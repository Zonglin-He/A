# Round6 Waymo-only training: independent CPU recipe audit

Status: official recipe reviewed; the new round6 implementation has not yet been code-locked or GPU-tested by this auditor. The new user authorization applies to a new run. `crossplatform/round3/p0/CANCELLED.json` and the old zero-epoch/failed-timing evidence remain unchanged. This audit does not start a GPU process or read Drone/Quad target labels.

Audited source: `crossplatform/round3/p0/upstream`, official commit `843a901f5367a6be5cc484b97c6ecad462961dfa`, with the existing unused-import/operator compatibility setup. The applicable recipe is `scripts/train_waymo.sh`, not the released mixed-platform checkpoint configuration. The script sets Waymo train/Waymo validation, batch24,100 epochs,six decoder layers,self-attention,soft-token and contrastive losses,topk4 point supervision,lr1e-4/backbone1e-3,weight decay5e-4,validation every10 epochs. Script save-frequency is100; saving additional recovery checkpoints is an explicit operational change, not a different initialization or method.

## Initialization and source-only identity

The active visual backbone is `Point_Backbone_V2(..., input_channels=3)`. The apparent optional PointNet-pretrained load in `models/bdetr.py` is commented out; `pp_checkpoint=None` therefore means random visual/task modules. `RobertaModel.from_pretrained('./data/roberta_base/')` loads generic pretrained RoBERTa-base and freezes its parameters. A source-only cold start must not load the mixed `final_6384` checkpoint, including a mixed-checkpoint text encoder extracted as an initializer. Config-only text construction without loading generic pretrained text weights is also not equivalent.

The model factory should record generic RoBERTa asset provenance and SHA, initialization seed and initial state fingerprints, and clearly separate cold initialization from a verified same-run source-only recovery checkpoint. Timing must use throwaway parameters/optimizer/BN state, followed by a fresh formal initialization. A completed twelve-step timing model is not a source-only scientific checkpoint and must not silently supply the first twelve formal updates.

## Actual optimizer, module modes and stochastic state

Official AdamW groups are: trainable parameters outside `backbone_net`/`text_encoder` at1e-4; trainable `backbone_net` at1e-3; trainable `text_encoder` at1e-5. The third group is empty because all text-encoder parameters are frozen, but the official optimizer retains it. Weight decay5e-4 applies to every group through AdamW defaults; there is no special no-decay split for biases or normalization parameters. Default AdamW betas/eps apply.

`train_one_epoch` calls `model.train()`. Visual BN parameters and running statistics train; `init_bn_momentum` sets BN1d/2d momentum to.1. Frozen RoBERTa parameters do not update, but its dropout stays in training mode, along with the.1-dropout text projector/cross encoder/decoder. Freezing parameters is not the same as calling `.eval()`. Validation switches the whole model to eval; the next epoch must restore training modes. DDP uses `broadcast_buffers=False`; SyncBN is not enabled by the script.

The entry initializes actual NCCL rank/world-size before the trainer. This matters even for one GPU because `SetCriterion.forward` calls `dist.get_world_size()` unconditionally after its guarded all-reduce. A fake world-size constant or altered loss denominator would be a recipe change, not the same engineering repair.

The official `rng_seed=0` option is parsed but is not applied to global Torch/Python/NumPy RNGs by the active entry. The loader instead uses a Generator seeded0, the default DistributedSampler seed0 and `sampler.set_epoch(1..100)`. Train and validation loaders share the same Generator. Workers seed NumPy/Python from `torch.initial_seed()%2**32`, then add worker_id again to the NumPy seed. The loader's frame enumeration uses unsorted `os.listdir`. A new explicit global seed0 is a reproducibility choice and must be recorded, without claiming an existing global-seed call was faithfully reproduced. Record annotation-order hashes and actual sampling/RNG states. Source validation can advance the shared worker Generator.

The source entry sets cuDNN enabled, benchmark=True and deterministic=True. It does not specify TF32, matmul precision, AMP, gradient accumulation or deterministic-algorithm enforcement. Record the actual current-runtime flags. The planned FP32, no-accumulation batch24 timing tests the actual training batch; a smaller microbatch with accumulation is not automatically equivalent because BN and token-padding denominators change.

## Loss details that must survive the wrapper

Seven heads (proposal,last,heads0–4) each receive their own Hungarian assignment. `HungarianMatcher(1,0,2,True)` has constructor order `(cost_class,cost_bbox,cost_giou,soft_token)`: **class cost1,bbox-L1 matching cost0,GIoU matching cost2**. An early verbal message reversed the first two; this document is the corrected interpretation verified against the constructor. The outer training objective is

`8 * point_objectness + (sum_heads CE + 5 * sum_heads bbox + sum_heads GIoU + sum_heads contrastive) / 7`.

Matched bbox regression is the summed `center L1 + .2 * size L1`, normalized by number of matched targets. The loss uses raw predicted sizes for L1; its box-to-corner utility clamps dimensions internally for geometry. The separate official evaluation size clamp is not a license to clamp away training gradients before regression.

Soft-token CE preserves normalized positive maps and their entropy term; unmatched queries target the final classifier slot with EOS coefficient.1. Contrastive temperature is.07, with bidirectional query/token loss averaged by2, EOS/query and token weights.1, then divided by matched-target count. Background positives are at `attention_mask.sum()-1` and the preceding position (the actual indices, not an assumed generic replacement of sentinel semantics). True mention maps are binarized inside contrastive loss. No new attention/padding denominator mask should be introduced silently.

Point-objectness uses nearest top4 seed points under GT-size-normalized distance and point-instance membership, focal gamma2/alpha.25, per-example weights normalized across seed count and a final division by batch size. It is weighted8 outside the seven-head average. Dropping that term would also change the language-conditioned query generator's supervision.

The matcher computes matched-target count per rank, all-reduces if initialized, divides by world-size and clamps the denominator to at least1. With a real single rank and one valid target per item, this is batch-size24 for the seven criteria, not7×24 or256×24. Ordinary DDP gradient averaging is not an extra manual division of the scalar loss. Clip the full model gradient norm to.1 **after** backward and before AdamW; step the scheduler after the optimizer.

## Scheduler: boundary and effective update epochs

The script requests milestones25,26, but `warmup_epoch=-1` enters `(milestone-warmup_epoch)*n_iter_per_epoch`, while the warmup wrapper is disabled. Using the audited reference3682 eligible Waymo training expressions and drop-last batch24 yields153 steps/epoch and15,300 updates in100 epochs. Actual scheduler events are steps3978 and4131, immediately after the last optimizer update of epochs26 and27. Since optimizer.step precedes scheduler.step, the reduced rates are first used in **epochs27 and28**. These conventions must not be conflated.

`scheduler_contract.py` independently executes this exact official scheduler with a CPU scalar AdamW optimizer (including an empty frozen-text group). `SCHEDULER_CONTRACT.json` records before/after rates around both boundaries. This is an implementation fixture, not a model-training or throughput measurement.

## Source loader and timing/continuation acceptance

The active loader returns XYZ despite `use_color`; reflectance is computed but not returned. The defined geometric augmentation helpers are not called by the active single-object getitem path. Actual variation is random16384-point sampling (replacement when short), with fixed Waymo view rotation; no synthetic rotation/flip/noise/scale should be credited as an active official training trick. Waymo train/val scene isolation, eligible counts, frame/object order and any source-internal physical duplication are checked separately by the source-data auditor. No target Drone/Quad loader may participate in construction, selection or scheduler decisions.

The finite twelve-step engineering run should retain4 warmup and8 timed *completed* forward/backward/clipped-AdamW updates at real batch24, original loss, real rank-one distributed context and current module modes. Record finite loss/components, gradient norm before/after clip, trainable parameter updates, frozen text fingerprints, BN-buffer changes, optimizer/scheduler counters, synchronized GPU-step and loader-inclusive wall windows, peak allocated/reserved memory and actual batch/token/point shapes. An OOM or incomplete step remains a failed timing; it is not evidence that batch24 is feasible. Validation/startup/checkpoint costs are outside a training-step extrapolation, and twelve batches do not certify every later peak-memory case.

For the formal run, bind exact code/data/initializer identity before launch; keep fixed epoch100 as the primary model, use only Waymo data/validation for training and any operational source monitoring, and retain failures/recovery lineage. After100 epochs, independently check15,300 completed optimizer steps (or the declared actual count if the independently audited source population changes), scheduler state, provenance from fresh source initialization, frozen text and changed nontext/BN states. A later target evaluation needs a separate freeze/score release; target labels are not training, checkpoint-selection or adaptation inputs.
