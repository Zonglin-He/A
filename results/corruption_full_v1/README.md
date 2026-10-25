# Full current-kappa0 corruption and physical severity extension

The user requests complete targets and DECOTA-style realistic corruption types with severity. The previous Blur/JPEG/Dim experiment covered only128 queries/128parents per domain at one fixed strength (768views total); it was not a full-target robustness benchmark.

## Finite locked scope

TA-STVG current user-selected no-shrinkage (kappa0, beta1, eight native-selected expert observations, q1 jointMAP and actual new-interval exactDP), on the existing maintained complete HC1902queries/227parents and Vid10303queries/732parents. These are the same current target manifests used in the main tables, not the older DECOTA HC3482 or one-query/video roster. All prior full/development exposure is disclosed. Source remains original native per-offset MAP and physical hull. No model/method/severity tuning from outcomes, new backbone/TubeDETR or MBR comparison.

|Corruption family|Physical operation|Severity levels|
|---|---|---|
|Frame drop / blackout|Zero-filled missing frame at original timestamp; no deletion or relabeling of time|2.5%,5%,10% physical burst coverage|
|Frame freeze|Repeat the actual physical frame immediately preceding the burst, decoded if absent from sampled grid|2.5%,5%,10% physical burst coverage|
|Motion blur|Archived source-seeded directional line kernel, length scaled by native short-side/224; original magnitude retained|2.5%,5%,10% physical burst coverage|
|Occlusion|Archived seeded fixed half-width×half-height black occluder (quarter frame area), independent of target location|2.5%,5%,10% physical burst coverage|
|Exposure|Archived saturating uint16 RGB+100→uint8 operation|2.5%,5%,10% physical burst coverage|
|Legacy Gaussian Blur|Original5×5 sigma1.5 across sampled clip|Original fixed strength, now full-target|
|Legacy JPEG|Original quality30/subsampling2 across sampled clip|Original fixed strength, now full-target|
|Legacy Dim|Original uint8(float32RGB×.5) across sampled clip|Original fixed strength, now full-target|

Five burst families ×three physical coverage levels plus the three existing fixed controls = **18conditions×12205queries=219690views** (34236HC+185454Vid). Every view has same-input nativeSource/currentFull outputs, with audited same-query clean references. The family/condition average is an additional summary, not a replacement for every individual condition, level, signed tail or duration result.

The selected archived deployment scheme measures affected physical-frame fraction; it does not equalize photometric strength between families. `ceil(N*p/100)` preserves2.5 as a float rather than truncating to2. The source-identity seed uses current manifest metadata.source and total physical frame_count. Burst position never consults caption, prediction, GT event, target box or eight observations. No sampled hit and no event hit remain in the primary denominator. Requested/realized coverage and sampled-hit strata are reported; saturated/unchanged pixels are not automatically damage. Current clip/query supports can miss a burst in the full physical video.

These are deterministic controlled deployment/sensor simulations, not proof of natural-shift robustness or semantic invariance. No cropping/flip of coordinates or artificial GT-event hit. The archived seven-family1–5-intensity protocol is a separate historical scheme and is not silently mixed into this18-condition protocol. See reference/provenance.json in the source directory for the exact original commit and blobs. Archived method/optimizer/checkpoints/results/rosters are not reused as current results.

## Actual status and validation

Scientific fingerprint: `7ee8ded3135432ba8cec04effedcb8bf6c12d53f4dc7572b64844e3859a18da0`.
Execution-v2 fingerprint: `77f1e7b00ed3ce480769efeb9b69627b0b84bfab502f737c33468c6bb8ce1bbe`.
Active entry: `studies/corruption_full_v1/dispatcher_runtime_v2.py queue`.
Controller2159886/start246100435 is actually waiting without the GPU lock for existing owner1924421/start245882503. It does not signal or modify that owner, its paid HC collector1518134, current CPU domains, teacher or user services. Existing fullTTA→currenttiming→combinedVidgrid/TTA and all audits continue first. New corruption Source/expert calls and new evaluation predictions are currently **zero**. This publication is protocol/CPU validation/actual waiting startup, not a completed full corruption benchmark.

CPU preflight:12 physical ceil/coverage contracts; four actually decoded inputs across both domains×18conditions=72 repeat-exact real-pixel checks, unchanged positions/dtype/shape and clean parent-array bindings. Four additional actual old current-kappa0 queryview array rederivations/repeats match exactly. No model/optimizer/teacher/CUDA/GT scoring. Root actually viewed the private real HC/Vid operator QA image; public only image/code hashes and inspection receipt. First newly paid model input must still pass its actual checkpoint/native/current-pooling/cold-array checks during production; CPU pixel tests are not GPU validation.

Execution-v2 only accepts both successful raw-audit and subsequently root-published parent status after actual parent exit and every required audit-SHA gate. The original zero-work waiting dispatcher2138820 was stopped only after exactPID/start/nochild/noGPU/0Source/0expert proof and preserved privately. Original frozen scientific source/plan stays byte-identical. This is a before-GPU scheduling compatibility correction, not a scientific failure or hash waiver. Never restore the previous waiting owner.

## Reuse, actual computation and finite budget

The previous768exact-view outputs are eligible only after actual transformed/originalRGB hashes, manifest binding, parent prediction/compact/receipt SHA and currentkappa0 rederivation. Identical no-hit pixels reuse the fully audited clean Source/NoShrink predictions with a fresh exact-pixel identity proof; queries are not dropped. Every genuinely new input runs original Source224FP32/native grid and same-corrupted-frame eight-expert context, shared initialization/geometry/auxDP/actual raw-grid ROI pooling/oneprototype/no-shrinkage posterior/newinterval exactDP. Old clean expert evidence cannot be reused on changed RGB/context. Exact-context old teacher hits remain nonzero inherited acquisition cost; genuine misses invoke the existing frozen single teacher only.

Conservative finite upper bound:218922newSourcequeries and218922newjoint-eight-expert requests after768eligible oldviews; actual same-input/cache hits reduce these counts. No extra training/optimizer or parameter sweep. Original Source/teacher and updated perPID/start costs are retained, not cache-based deployment speed. Nested forward/request/finish/query times are not added.

## Durable evidence within storage limits

One GPU worker and one subsequent CPU scorer/auditor, sequential by complete condition×domain. Runtime numerical Source/real pooling staysTorch4; independent CPU scoring/replay uses one thread, CPU14–15/nice19/idleIO. Floors8GiBavailableRAM/80GiBdisk, study-private160GiB,72h per whole condition/domain worker.

Each newly paid Source.pt+SHA is checkpointed before teacher; failure/uncommittedSource is preserved. After prediction+compact+receipt commit, cold-array exact replay and readback, only that new active raw scratch may be removed with originalSHA/disposalreceipt. Every complete1902or10303unit seals before its independent officialGTscore. Then audit every prediction/receipt/compact/teacherSHA, every array and every mean; retain all signed/severe>5pp t/v/union-once/rescue/short≤5s/long parent-pointwiseCI and sampled-hit/clean-corrupt analyses.

After that entire unit has a successful official score, exact-array audit and signed report, only this NEW unit's temporary geometry/pooledfeatures and ownedteacher payloads are disposed with every originalSHA/disposalreceipt. All predictions, original teacher/context/Source hashes, costs, scalar metrics, reports, audit/seal/receipts remain. This does **not** preserve arbitrary future geometry/repool/optimizer replay after disposal. No old capture/root/cache is deleted or changed. Without this explicit bounded-residency policy, hundreds of thousands of multi-MiB geometry/teacher payloads would exceed the disk budget. Failed/uncommitted paid state must never be pruned to hide failure or blindly reacquired.

## Completion

Each unit ends at scored/audited pending root analysis/figure/publication. Root must read/export every new valid unit, notify material results including negative tails, render and inspect complete severity figures, update ledger/archivecheck→snapshot→check, publish all anonymous positive/negative results and preserved failures, then verify both GitHub refs plus every remote mode/blob-byte/SHA/tree/ancestor. Keep existing heartbeatACTIVE through all earlier work AND all18newconditions/two complete domains. Do not mark fullTTA or this study complete from queued/partial results. No new agents/chats/automation/manuscript edits. Media/captions/GTcoordinates/rawgrids/masks/weights/credentials stay private.
