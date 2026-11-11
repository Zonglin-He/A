# R70 — Frozen CLIP evidence and independent random RGB adaptation

**Decision:** retain frozen CLIP direct reranking as a development candidate; neither trainable adapter establishes extra grounding value in this round. Branch A and B are separate experiments, not a combined system.

All 336 historically exposed DEV observations are included: 168 Drone, 168 Quad, five scene clusters. The frozen Waymo-only epoch100 grounder supplies the original 256 boxes and source scores. No source retraining, FINAL input/prediction/scoring, new target GT geometry scoring, per-case GT selection, or new hyperparameter sweep occurred. Existing candidate IoUs were joined only after prediction seals and independent audits. These are development results, not independent confirmation.

| Arm | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| A_subject_tight | 17.1881 | 29.8448 | 23.5165 | 44.0476 | 22.0238 |
| A_subject_joint | 16.9349 | 29.1044 | 23.0196 | 42.8571 | 21.7262 |
| A_full_joint | 16.7747 | 29.0197 | 22.8972 | 41.9643 | 22.3214 |
| A_projectonly_joint | 18.6357 | 30.8080 | 24.7219 | 44.0476 | 22.0238 |
| A_mismatch_joint | 14.8536 | 27.5701 | 21.2118 | 38.6905 | 20.8333 |
| A_adapter3 | 15.5551 | 28.4831 | 22.0191 | 41.0714 | 21.4286 |
| A_adapter16 | 16.6263 | 29.4949 | 23.0606 | 43.7500 | 21.4286 |
| B_random0 | 15.4948 | 28.7424 | 22.1186 | 41.6667 | 21.4286 |
| B_step3 | 15.4940 | 28.7445 | 22.1193 | 41.6667 | 21.4286 |
| B_step32 | 15.5342 | 28.5965 | 22.0654 | 41.9643 | 21.1310 |
| B_wrongtest32 | 15.5698 | 28.5965 | 22.0832 | 41.9643 | 21.1310 |
| B_wrongtrain32 | 15.5698 | 28.5965 | 22.0831 | 41.9643 | 21.1310 |
| B_NW | 15.5979 | 28.2266 | 21.9122 | 41.3690 | 21.1310 |
| B_NW_wrong | 14.7721 | 28.2353 | 21.5037 | 40.4762 | 20.5357 |
| B_mean | 15.3089 | 28.3986 | 21.8537 | 41.0714 | 20.8333 |
| frozen_ABC | 15.7363 | 28.8686 | 22.3025 | 42.2619 | 21.4286 |
| R59 | 16.3856 | 28.8367 | 22.6111 | 43.1548 | 21.7262 |

## Branch A: useful new evidence, unproven adaptation increment

Projection-only joint CLIP evidence scores **24.721868%**, **+2.891537 pp** over source and **+2.110752 pp** over historical R59. It improves 61 observations and worsens 27 versus source; the other 248 have unchanged IoU. Its largest single positive contribution is 0.207389 pp, well below the total gain. All five scene means improve. This supports a distributed development signal, not a single rescued example.

The support-weighted joint arm scores 23.019640%. Using the same crop embeddings and source scores with projection-only weighting improves it by 1.702228 pp. Thus this particular support/purity reliability formula suppresses useful evidence overall. Sparse visible support is not an independently validated foreground mask, and overlap among many valid source hypotheses can penalize the correct candidate. This result does not prove all reliability estimation is harmful or all projection bindings correct.

Tight crop direct use scores 23.516475%; full-sentence joint use scores 22.897217%. Attribute-prefix parsing is literal and not a certified subject-role parser. The source grounder always retains full text. CLIP is not claimed to solve spatial relations; no new hand-written position/reference score is added.

The zero-output residual score adapter (64D grounder query + 512D CLIP region + 512D CLIP text + reliability; bottleneck 16) produces nonzero parameter changes in all 336 inputs. It learns a stopped tight-crop CLIP/source distribution and evaluates context-crop features, resetting per input. Three steps score 22.019122%; 16 steps score 23.060563%. The latter is only +0.040923 pp over support-weighted joint direct use, is below the same-information tight-crop direct arm by 0.455913 pp, and below projection-only joint direct use by 1.661305 pp. Do not adopt this adapter as an added-value TTA method.

No second 3D network, Mask R-CNN, Grounding DINO, Qwen, or their output caches enter branch A. CLIP remains frozen; its pretrained knowledge is extra information and cannot be credited to parameter adaptation. Mapped visual evidence is centered before propagation; invisible/unknown candidates do not receive a fabricated CLIP negative label. Complete-link duplicate grouping requires both box overlap and actual point-support overlap, and one group evidence is propagated once per query rather than accumulated duplicate votes.

## Branch B: alignment loss falls, useful RGB learning not established

The independent random CNN has 10,512 parameters and no pretrained RGB encoder. Only source pre-cross-encoder point feature directions supervise it. At most 256 visible seeds fit each per-input module; mean 428.16 disjoint visible seeds test it. RGB residuals modify only those held seeds, with a 0.2 relative-norm cap. Source weights, language encoder, all decoder/matching parameters, query anchors and box choices remain frozen. The full cross-encoder and six decoders are replayed to test the changed evidence.

Held-point alignment loss falls from 1.305857 to 1.148822; wrong-binding training reaches 1.149565. Task performance at 32 steps is 22.065351%, +0.235020 pp over source, but **-0.053285 pp relative to the unfitted random module**. The trained model is also slightly below wrong-binding readout (-0.017808 pp) and wrong-binding fitting (-0.017793 pp). Thus source-relative gain cannot be attributed to learned useful RGB semantics.

The no-training RGB-neighbor interpolation arm scores 21.912236% and beats its permuted-binding control by +0.408532 pp, but only +0.081905 pp over source. Preserve that small correspondence signal without presenting it as successful learned RGB adaptation. Current B results do not justify additional architecture/step search. They do not rule out source-prepared or longer-term RGB learning under a different protocol.

The wrong-RGB controls permute patch-to-point binding within the same input, not images between scenes. The held observation here is a disjoint point/patch subset of the same frame, not a new physical viewpoint. Neither branch claims cross-frame or unseen-scene transfer.

## Scene results (full DEV, no scene selection)

| Platform / anonymous scene | n | Source | A projection-only joint | A adapter16 | B step32 | R59 |
|---|---:|---:|---:|---:|---:|---:|
| drone S1 | 106 | 10.5850 | 13.0738 | 11.2866 | 10.5151 | 11.7616 |
| drone S2 | 62 | 24.0112 | 28.1447 | 25.7553 | 24.1153 | 24.2910 |
| quad S1 | 62 | 10.1175 | 11.9167 | 10.5186 | 10.0718 | 9.6321 |
| quad S2 | 49 | 32.9989 | 36.5471 | 36.1994 | 35.3601 | 36.0676 |
| quad S3 | 57 | 43.5098 | 46.4229 | 44.3722 | 42.9318 | 43.5098 |

No sample-independent significance claim is made from 336 correlated observations. Scene-level values are retained; there are only two Drone and three Quad clusters, and these data have prior method-development exposure.

## Measurement and resources

All 6,048 task entries and 6,304 ordered negative comparisons are preserved, including the stronger zero-update arms. Independent CPU audit checks all 336 box selections, candidate mappings and actual saved A weights; independent GPU audit replays 20 stratified inputs from all five scenes (160 B suffix replays), reconstructs their raw calibrated RGB/point inputs, and reencodes 16 CLIP crops. B GPU replay probabilities are exactly equal; maximum independently reencoded CLIP feature error is 6.97e-7. Every B source zero replay was bit-exact and all fitting/held seed sets are disjoint. Existing saved IoU lookup and all metric/negative arithmetic are independently checked.

Preparation: 71.30s. Frozen CLIP: 295.91s for 149,362 crops. A fitting: 14.45s, 5,376 gradient steps. B fitting and 3,024 frozen suffix replays: 90.90s, 21,504 CNN steps. Peak allocated GPU memory is 0.663 GiB for CLIP and 0.636 GiB for B. These marginal cached-experiment timings exclude prior source full forward/cache production and are not complete deployment latency.

All actual weights, embeddings, RGB/point data, descriptions, scene/sample identities, camera coordinates and private caches remain local. Public export includes original code, fixed configuration, aggregate and anonymized per-observation scalars, all negatives, failure/repair history and audits.

## Literature and implementation scope

CLIP2Scene transfers pretrained image-language semantics through 3D training and semantic/spatial-temporal constraints; it is not this fixed-checkpoint TTA algorithm. [Official CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_CLIP2Scene_Towards_Label-Efficient_3D_Scene_Understanding_by_CLIP_CVPR_2023_paper.html). OpenScene aggregates pixel-aligned visual features and uses depth/projection visibility; our sparse LiDAR z-buffer and rectangle crops do not reproduce its dense instance or visibility evidence. [Official implementation](https://github.com/pengsongyou/openscene). CLIP-Adapter freezes CLIP and uses a supervised few-shot residual adapter; our target-only A/B supervision and modules are different experimental proposals. [Official implementation](https://github.com/gaopengcuhk/CLIP-Adapter).

## Adoption boundary

Keep A projection-only direct CLIP evidence as the strongest new DEV candidate. Do not claim its pretrained RGB gain as TTA, and do not claim independent validation. Do not adopt A adapter3/16 or B random-RGB training as added-value adaptation. Stop these exact fitting recipes; do not expand steps, loss terms or model size based on the same results. A later fixed larger evaluation may test whether the direct CLIP advantage survives other observations/recordings; it is not launched by this study.
