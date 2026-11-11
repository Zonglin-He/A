# R75 — Native query–text dense semantic recovery TTA

## Decision

**Do not adopt or expand this fixed recipe.** Dense semantic recovery improves held-mask row and column KL and outperforms output-only consistency TTA, but fails the required incremental grounding comparison against the same-observation frozen readers. No new optimizer/layer/masking-rate search is justified by this prototype.

On DEV336, dense A+M1+M2 readout is **22.088554%**, versus **22.173178%** for each of the three frozen direct readers: **−0.084624 percentage points**. It exceeds output-only adaptation by +0.257796 pp but is only +0.004380 pp above shuffled-row adaptation; this does not establish useful correspondence-conditioned adaptation. On complete A alone it is +0.008161 pp above source, one IoU improvement and one loss, while shuffled-row adaptation is higher. Held M3 grounding falls by −0.085051 pp.

## What was actually changed

Fixed Waymo-only epoch100, original A256 boxes, all 256 source query seed anchors and all six source decoder position arrays. No source training, expert model, CLIP output, RGB branch, new pretrained network, EMA, new decoder layer or new reconstruction head. Only the existing 288→288→288→64 query projection MLP (184,960 parameters) is trainable. Text projection, text backbone, point backbone, cross encoder, decoder and geometry are frozen.

This is **query tokens × valid language tokens**, not seed-level semantic segmentation. R74 pooled multiple language-conditioned queries into regional vectors and optimized regional InfoNCE. R75 never pools regions or constructs region negatives: it constrains the original query/text alignment matrix directly, with soft many-to-many distributions rather than object pseudo-identities.

The fixed cohort is the already exposed DEV336, 168 observations per platform and five recording groups. No error-dependent or expert-dependent filtering, and no new confirmation/test split. Existing split rows are reported descriptively, not presented as fresh holdouts. FINAL was not accessed. No automatic population expansion is authorized or queued after this negative result.

## Complete reference and real pre-fusion masking

The complete reference is the exact existing source-A forward saved in R66, including pre-cross-encoder visual/text/position features, source point/sample indices, source query anchors and the six decoder positions. Two fresh full-model source forwards in engineering checks reproduce the complete cache and zero-mask transformer replay **bit-exactly**. Existing cached reference use does not mean that the new masked query vectors are fabricated.

For M1 and M2, independently hash-select floor(20%×768)=153 non-anchor seeds from the 1,024-seed source representation. All 256 query seed anchors stay visible. Thus the actual removed fraction is about 14.94% of all seeds; there is only this one fixed strength, no masking-rate grid. Complete text is unchanged. Before **each** cross-encoder layer, masked visual values are zeroed and its visual keys are blocked; values are zeroed again after the layer. Decoder visual key/value reading remains masked in all six layers. Query features are freshly sampled from the resulting fused features at the fixed source anchor indices and passed through the actual query projection and all six decoder layers. The source fused query is never used as the masked-query initializer. Source position arrays remain fixed by design; final predictions still select the original A256 boxes.

This is transformer-context token removal after the frozen point backbone, **not raw sensor-point deletion**. The retained backbone tokens still have their original receptive fields. It tests recovery from missing seed context, not reconstruction of a truly unseen raw point cloud or new camera view.

Engineering replaces masked visual values with +999; independent stratified audit replaces them with −1234.5. The final masked query/text results remain exact. All blocked attention mass is zero. No anchor query key is masked. Dynamic text vectors do change under masking, as expected from cross-modal encoding; the adaptation loss deliberately **does not use those changed vectors**.

## Fixed semantic target, loss and update

Let U0 be the complete input's normalized projected text and Q0 its normalized projected final queries. For each masked input, recompute its final hidden query Hm and calculate Sθm=normalize(MLPθ(Hm)) U0ᵀ/0.07. U0 is the same stopped, complete-reference array in every arm and view. Merely freezing RoBERTa is not used as a substitute for this explicit reference.

Dense objective: average over M1/M2 of mean-query KL(softmax_tokens(S0) || softmax_tokens(Sθm)) plus mean-token KL(softmax_queries(S0) || softmax_queries(Sθm)); column coefficient one. Uniform weights on all valid rows and token positions, including the original `not mentioned` suffix and special positions. There are no padding tokens in these batch-one inputs, no forced one-word/one-object matching, no target query labels, and no hard assignment of every candidate to the noun. The complete reference retains source background/sentinel behavior; this is **soft self-distillation**, not the official supervised positive-map loss and not a ground-truth background assignment.

All arms add the same weight-one cosine source-query preservation on complete A. Output-only control replaces the dense term by the mean M1/M2 KL of the complete-reference candidate distribution to the masked candidate distribution. Shuffled control deranges the complete-reference matrix rows by a fixed one-cycle permutation, preserving its values, row/column target normalization and update budget.

Each input/arm starts from the source head. One Adam proposal, lr 0.0005, clipping one; transactionally test scales 1…1/256 and accept only an unlabeled total-objective decrease of max(1e−12,1e−6×|old|). Roll back weights and optimizer state otherwise. Same number of proposal opportunities, same features and source preservation for all three arms. No GT selects a scale, a row, a mask, a sample, or a method.

Dense and shuffled accept 336/336 updates; output-only accepts 330/336 and rolls back six. **1,002 accepted updates, 1,008 stored states, zero accepted objective increases.** Mean dense training objective decreases 0.000627114→0.000503389. These facts establish working optimization, not improved grounding.

## Readout and controls

Task matching uses the existing full text, original first-noun mask, all-token denominator/background alternatives and stable target-token log-odds followed by candidate normalization. Every complete-source native Top-1 agrees with the original saved source readout. No new parser or annotated span enters the experiment.

Same-budget frozen probability mean, token-logit-matrix mean and normalized-query representation mean each use **A, M1 and M2**, fixed candidate identity and the same U0; none updates parameters. Their vectors differ but all selected boxes coincide on this cohort. Adapted full-A and A+M1+M2-mean readouts are both retained. Historical ABC and R59 are strong references using three independently sampled raw-point forwards, a different perturbation protocol, not a claim that their source signals are identical to context masking.

## Complete DEV336 results

| Method | Drone mIoU % | Quad mIoU % | Balanced mIoU % | Acc@25 % | Acc@50 % |
|---|---:|---:|---:|---:|---:|
| Frozen complete A | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| Frozen A+M1+M2 probability mean | 15.863233 | 28.483122 | 22.173178 | 41.666667 | 21.428571 |
| Frozen A+M1+M2 token-matrix mean | 15.863233 | 28.483122 | 22.173178 | 41.666667 | 21.428571 |
| Frozen A+M1+M2 representation mean | 15.863233 | 28.483122 | 22.173178 | 41.666667 | 21.428571 |
| Historical frozen ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| Historical R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| Dense update, complete A readout | 15.583277 | 28.093706 | 21.838492 | 40.773810 | 21.130952 |
| Output update, complete A readout | 15.539892 | 28.093706 | 21.816799 | 40.773810 | 21.130952 |
| Shuffled dense update, complete A | 15.702795 | 28.281102 | 21.991949 | 41.071429 | 21.726190 |
| Dense update, A+M1+M2 mean | 15.693986 | 28.483122 | 22.088554 | 41.369048 | 21.428571 |
| Output update, A+M1+M2 mean | 15.540747 | 28.120770 | 21.830758 | 40.773810 | 21.130952 |
| Shuffled dense update, A+M1+M2 mean | 15.887247 | 28.281102 | 22.084174 | 41.369048 | 21.726190 |
| Frozen held mask M3 | 15.755518 | 28.725104 | 22.240311 | 41.964286 | 21.428571 |
| Dense-updated held mask M3 | 15.585416 | 28.725104 | 22.155260 | 41.666667 | 21.428571 |
| Output-updated held mask M3 | 15.432177 | 28.483122 | 21.957650 | 41.071429 | 21.428571 |
| Shuffled dense held mask M3 | 15.863608 | 28.308166 | 22.085887 | 41.369048 | 21.726190 |

The dense ensemble loses one IoU case and rescues none relative to its matched frozen ensemble; other selected-box IoUs remain unchanged. On Drone the loss is −0.169247 pp; Quad is unchanged. Complete-A dense editing changes two selections, one improved and one harmed. Its small +0.008161 pp over source is not an incremental gain over direct use. Dense versus shuffled ensembles have five improvements and five losses; +0.004380 pp alone is not evidence that correct row binding produces useful overall task correction.

## Held M3 semantic recovery and task transfer

M3 is a new independently hashed mask generated **only after every M1/M2 adaptation state was sealed**. It is not used in the objective, step acceptance, feature selection or parameter choice. It is a third missing-context version of the same physical input, not a new scene or viewpoint. The complete input's S0 remains the evaluation reference.

| Method | Held row KL ↓ | Held column KL ↓ | Row Top-1 agreement % | Column Top-1 agreement % |
|---|---:|---:|---:|---:|
| source | 0.000053805 | 0.000406814 | 99.450102 | 95.945666 |
| dense | 0.000041440 | 0.000392426 | 99.215262 | 96.371917 |
| output | 0.000053641 | 0.000562622 | 99.152483 | 96.361563 |
| shuffled | 0.003063367 | 0.116842893 | 91.010975 | 60.353365 |

Correct dense supervision improves both held semantic KL means, unlike the shuffled target. This is a real positive auxiliary result and should not be discarded. It does **not** establish target correctness: the source relation matrix is already very stable under this mild mask (row Top-1 agreement 99.4501%), row Top-1 agreement falls after dense updating even though KL improves, and held grounding drops **22.240311→22.155260%**, zero IoU rescues and two harms. Output-only held grounding is 21.957650%; shuffled held grounding is 22.085887%.

This differentiates R75 from R74: a native semantic auxiliary objective can generalize numerically to a held mask, yet that recovery still does not add useful grounding information. It is consistent with restoring source semantic relationships without resolving a stable incorrect instance choice. It does not prove that all complete references are wrong or that every denser objective must fail.

## Measurement and implementation checks

All 336×16=5,376 readouts are sealed before the saved original-A candidate-IoU join. Zero new raw GT/geometry scoring. All 1,325 ordered negative comparisons, six rejected-update cases and all unchanged rows remain in the exported cohort. Five recording groups and 336 point/image frame pairs are reported; no i.i.d. confidence interval or unseen-scene claim is made from these development inputs.

Independent NumPy checks replay every stored matching state and both loss families, fixed U0, row derangement, source matrices, frozen/adapted readouts, mask identities and original box selection. Maximum probability discrepancy 3.281e−7, objective discrepancy 3.495e−7, reference matrix discrepancy 2.235e−6. Independent GPU replay covers two inputs per recording (10 inputs), all three masks plus zero-mask replay, 70 transformer suffix calls, exact features and masked-value leakage checks. Complete source state fingerprint is unchanged throughout GPU stages. The initialization warning about a RoBERTa pooler does not indicate newly trained parameters: strict checkpoint loading and the complete state fingerprint are checked before execution, and the source forward bridge passes.

Production new suffix forwards: 672 for M1/M2 plus 336 for M3. Engineering adds 10 suffix calls and two full source forwards; independent GPU audit adds 70 suffix calls. No other full source, visual expert or raw-GT forward is hidden. Matching updates use cached newly computed final queries on CPU; this is exact for the fixed update scope. Source full-forward/backbone prefixes are reused, not retrained. Costs are incremental replay costs and do not assert end-to-end deployment latency.

No scientific producer failed or changed configuration in this round. All fixed readouts and rejection cases are retained. No post-score method or prediction adjustment was made.

## Literature boundary and adoption

MIC provides the masked-student/complete-input semantic target mechanism but is a UDA system using teacher-generated supervision, rather than this fixed-source one-step TTA. Its official implementation can share or instantiate an EMA teacher and supplies confidence-weighted pseudo-labels; those components are not silently claimed here. [Official MIC code](https://github.com/lhoyer/MIC/blob/master/seg/mmseg/models/uda/masking_consistency_module.py).

MATE uses a source-prepared masked reconstruction task for point-cloud classification. We add no random reconstruction decoder and do not inherit its source-preparation assumptions. [Official MATE repository](https://github.com/jmiemirza/MATE).

EDA supplies fine-grained supervised grounding alignments, with source annotation-derived positive maps and background handling. The local 3EED contrastive loss was checked directly: both query→token and token→query directions exist. R75 reuses that **interface**, not its unavailable test-time GT correspondence. [Official EDA loss](https://github.com/yanmin-wu/EDA/blob/master/models/losses.py).

The prespecified stop condition is met: better auxiliary recovery and better performance than output-only updates do not beat same-information frozen fusion. **Stop this fixed dense recovery recipe, no larger masking/step/layer grid, no automatic expanded evaluation.** Maintain expert-free scope; this result does not authorize external feedback, source retraining or FINAL access, nor does it refute all expert-free 3EED TTA.
