# Single-write transfer and frozen CLIP token qualification

2026-10-02. User-authorized continuation of a71f9e4. These are two separate
mechanism diagnostics, not a new online method. Existing predictions, states,
GT files, checkpoints and CURRENT_METHOD remain read-only. No memory, new
online Vid R, temporal multi-view, fusion, parameter search or old queue.

## A: HC-STVG-v2 single-write audit (primary)

Reuse the original 32 historically exposed development sources, one query per
source, two orders, clean and five transient 5% corruptions. Each order has eight
scheduled expert arrivals. Keep all 96 scheduled writes per arm, including
empty/no-op writes. A Uniform and R routed retain their own historical pre/post
1792-parameter spatial states, HC K8, lr .006097133675874025, teacher T1,
student T1, rho .05 and D4. No new training, backward or specialist call.

Frozen semantic context z: official HC checkpoint's text-only RoBERTa body,
mean final lexical token hidden states (exclude start/end/padding), unit norm.
Compute all 32 in eval mode before any new GT read. This is a frozen semantic
proxy, not a guaranteed notion of correction applicability. Do not use post-MM
H text (which also depends on visual input). Lock full 32x32 cosine matrix and
target table. For every donor arrival i: self; earliest future nonexpert;
cosine-nearest and cosine-farthest among ALL future different queries j>i in
that order. Same targets for A/R and all conditions; exact ties choose earliest
future arrival. Near/far targets can be expert-scheduled in the original stream,
but this diagnostic makes no target update and calls no expert. Report that
coverage separately. Deduplicate computation, preserve all role rows.

For each donor/target/arm, run cached native suffix under the saved pre and post
states, with dynamic native routing unchanged. Do not transport the delta to
initial state. Primary cross-query spatial effect uses the TARGET's frozen
checkpoint native temporal interval, identical before/after and A/R; evaluate
dense vIoU and sIoU with official HC clipping and physical-time mapping. Report
free native decode as a secondary readout (possible temporal side effects).
Self control uses each arm's saved pre-native interval to exactly reproduce the
previous local-update definition; additionally report the common frozen interval
for all roles. Self pre/post boxes and decoded indices must bitwise match saved
outputs for all writes. Checkpoint tensors restored after every replay.

Expected logical pairs: 96 writes x 4 roles x 2 arms = 768; 576 cross-query and
192 self. Primary corrupt subset: 80 writes/arm. Report all scheduled, actual
write and no-op denominators; no GT-based positive-write filtering. Bootstrap
10000 times, seed 20261001, donor-source clustered primary (average conditions
and both orders within donor source), target-source clustered sensitivity and
paired R-A/near-far. Repeated donor/target role aliases are not independent.
Donors only cover the original scheduled-source union (14 sources), not all32.
Report cosine, lag, selection coverage, selected target diversity, gross harm,
>5pp negative tail and positive/negative examples. No confidence-bound gate.

Transfer effects are causal state-pair comparisons at their historical base
states; text near/far association is not proof of general semantic causality.
Positive isolated transfers alongside an adverse online mean do not uniquely
identify accumulation/interference. Memory remains a future hypothesis.

## B: aligned-projection CLIP token binding P1 (secondary)

Reuse exactly the prior qualification's 60 fixed cells (10 sources per dataset,
10 clean +20 corrupt per dataset), first-step A nine tubes and A native interval:
540 candidates. S = already sealed routed Sa2VA rewards on those same tubes.
No new Sa2VA, no TA-STVG update, no new online output or fusion.

Frozen VLM: OpenAI CLIP ViT-B/16, Hugging Face official converted checkpoint
openai/clip-vit-base-patch16 revision57c216476eefef5ab752ec549e440a49ae4ae5f3.
Bind upstream LFS SHA256, file hashes and installed transformers implementation.
All weights frozen. Use final vision patch hidden states (exclude CLS), native
vision post-LN and pretrained visual_projection; lexical text token hidden states
(exclude SOT/EOT/padding), pretrained text final-LN/projection. Unit-normalize
both in the shared512-dimensional CLIP projection space. This is NOT a FILIP
checkpoint: its training objective supervised pooled CLIP features, not this
token-wise score. Local lexical/patch alignment is precisely the empirical
qualification and is not assumed from equal dimensions. Do not claim a negative
result falsifies FILIP-style pretrained local alignment or all token approaches.

Deterministic Stanza English ewt tokenize/mwt/pos/lemma/depparse from existing
cached models, CPU, no download/training/LLM. Query referent: wh-determiner noun,
who/whom -> person, nominal subject, then first noun fallback (record). Object
phrases contain referent modifiers/compound nouns and attached nominal clothing
attributes; event phrases contain lexical verbs (except referent wear-attribute)
and their object/oblique nominal complements. Keep phrase lists separately,
encode each in the same CLIP text encoder, concatenate its lexical BPE tokens,
never pool the full sentence or each branch to one vector. Empty branch means
unavailable, not a synthetic reward. Publish parser/version/counts/rules, not
private captions/phrases.

Decode original sampled frames with original dataset binding, apply exact old
corruption, verify whole-clip pixel digest. Resize entire RGB frame to224x224
with PIL bicubic, no center crop, normalize CLIP mean/std. This preserves all
ROI coordinates but changes aspect ratio; disclose as fixed preprocessing.
Batch <=8, FP32 eager inference, no autocast. All original sampled frames,
not only expert frames. Cache projected patch tokens once per unique input.
ROI consists of 14x14 patches with strictly positive area overlap with normalized
candidate cxcywh box clipped to[0,1], no nearest-patch filling. The hard membership
can make similar probes identical; preserve ties and report unique ROI/score
signatures. Empty ROI/in/out pools remain unavailable; selection fallback is
native candidate0 but ranking availability is separately reported.

Object diagnostic: mean over object lexical tokens of max cosine over candidate
patches in the common native event interval. Binding T (only ranking score):
mean over event lexical tokens of [max inside-I ROI cosine - max outside-I ROI
cosine]. No object-score sum, temperature or support/weight sweep. Unequal in/out
pool sizes and static per-frame VLM's lack of explicit motion are limitations;
publish counts. Analytic max/cosine/ROI/empty/tie tests; common projection pooled
CLS/EOT parity to original CLIPModel; scalar CPU readback of stored tokens.

All60 token scores sealed before new token GT reads. GT only offline fixed-tube
qualification: pairwise accuracy on strict GT pairs (reward ties get half credit),
top1 vIoU gain vs candidate0, S/T complementarity with tie counts, regret, clean,
source-bootstrap10000. Source-macro metrics, same availability vs full fallback
both reported. T>S on a few examples is not proof of an online method. No event
outside object labels are fabricated. No conditional fusion automatically runs.

## Execution and close

GPU serial, transfer first; CPU download/parser preparation may overlap.
No total deadline or indefinite polling. Free disk floor8GiB. Model workers
forbid GT/summary/outcome reads; scoped prediction barriers precede CPU scoring.
Preserve engineering attempts and additive revision pins. Root independently
checks state bindings, all self parity, target selection, dense metrics and all
late-interaction scalars; retains positive/negative results. Archive
check/snapshot/check, reviewed public export of code/protocol/anonymous scalars/
plots to Zonglin-He/A, verify every remote byte before FINAL_COMPLETION.

Primary sources: [CLIP official code](https://github.com/openai/CLIP/blob/main/clip/model.py),
[official checkpoint](https://huggingface.co/openai/clip-vit-base-patch16),
[FILIP](https://arxiv.org/abs/2111.07783),
[MaskCLIP ECCV22 official](https://github.com/chongzhou96/MaskCLIP).
MaskCLIP motivates auditing the local-feature interface; its altered dense
vision computation and class EOT targets are not claimed as this lexical P1.
