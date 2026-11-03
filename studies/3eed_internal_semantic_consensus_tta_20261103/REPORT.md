# R56: internal stage/entity semantic consensus and real matching-MLP TTA

Decision: retain a weak one-step development signal, but do not adopt a stable TTA method or replace the source/R50 reference. Source balanced mIoU21.830331%; initial semantic consensus21.747631%; one-step TTA21.997478% (source+0.167147pp, same-information direct+0.249847pp); three-step20.987558% (source-0.842773pp). The positive one-step result is real under this fixed protocol, but its scene means improve1/decline2/unchanged2, and it has not been independently confirmed.

## Fixed scope and semantic sources

Exposed DEV336,168Drone/168Quad,twoDrone+threeQuad clusters; Waymo-onlyepoch100, original256final boxes unchanged. No FINAL384/full5912, source retraining, extra visual/teacher model, RGB rule target or geometric position/relationship score is used. RGB is not an input to this round's supervision. Geometry only associates separate-text hypotheses. Earlier query slots trace native decoder evolution; different texts never share slot IDs. All19 arms are registered before label evaluation; no observation/platform-specific layer choice or weight search.

R53 already saves six decoder query states. Proposal is reconstructed through the original gsample_module/decoder_query_proj from the frozen fused seed bank; allsix decoder layers are replayed on that bank and must match their saved query tensors bitexact on all336. These are component replays, not new observations. Four independent complete-description source forwards verify proposal+six decoder projected query and entity-logodds outputs(28stage checks), all bitexact. A query slot need not keep the same physical identity during refinement; the evaluation follows its path and always selects the corresponding final box.

Each stage is projected by the actual trained shared matching MLP and matched to the frozen fused text-token projections with the original0.07scale and ALL-token denominator. The source T firstnoun token mask remains fixed. Reference masks use existing literal spans in the complete sentence, with unsupported spans unavailable rather than negative. Stage-native raw scores and final original source score are separately retained.

## Three fixed semantic readout families

Early raw consensus: take candidate-softmax of native T logodds at proposal+decoder0..4, then equal average of the six distributions; final decoder5 is excluded. Individual earlier-stage outputs are diagnostic arms only.

Entity-role readout: subtract log positive-token count from each entity's native logodds to reduce span-length bias; use target minus logmeanexp of available reference roles, then candidate-softmax. Report final-role and equal-weight six-stage early-role consensus. With no available reference, this reduces to the corresponding target-only distribution. These scores are not calibrated instance identities.

Subject phrase: run the same3EED on each initial parsed literal subject phrase plus the existing sentinel. Positive tokens include all phrase content before the sentinel, explicitly different from the full-description source firstnoun mask.336 new subject-phrase forwards; no new reference forwards.573 old R51 reference-headspan outputs are reused with checked source fingerprint/file hashes and exactly identical sampled raw point IDs. Reference phrase inputs keep their own learned features and independent256 boxes.

Cross-text transport: use unique raw sampled point IDs and source/final/phrase boxes. A_ij=sqrt(shared-point-Jaccard*axis-aligned-box-IoU), requiring>=4 shared points. W_ij=A_ij/(1+sum_i A_ij) is column-substochastic; semantic mass not assigned to a final query remains unassigned. No nearest-box forced mapping, top1 reference truth or same-index shortcut. Duplicate query slots remain hypotheses, so multiplicity/support purity may still bias transport.

Subject mapped mass is mixed with its unassigned mass times the source target distribution. Reference unassigned mass is distributed uniformly, giving no relative reference evidence. Phrase-role distribution is proportional to mapped subject probability divided by mean mapped reference probability, then normalized. A subject-only phrase arm is separately retained. Source-prior backfill is explicit uncertainty handling, not a rollback selected by GT.

Main initial consensus is exactly the equal mean of early raw consensus, early-role consensus and mapped phrase-role distribution. All are computed at source before any update and detached. There is no confidence gate, per-sample choice of which signal to trust, or GT-generated positive label. A single hash-fixed candidate permutation of the entire consensus preserves its probability multiset/entropy while breaking final candidate correspondence.

## Actual parameter adaptation

Only the existing full query matching MLP is updated:184960parameters. All decoder/backbone/texthead/regression parameters are frozen. Source weights initialize each observation and each real/shuffled signal; optimizers are fresh. Adam lr5e-4,betas.9/.999,epsilon1e-8,weightdecay0,gradientcap1,fixed1/3steps. Objective is only KL(frozen initial candidate consensus || current final target candidate distribution). It is self-distillation from one source model, without an additional teacher network; no new entropy/geometry/color/keeping term is inserted.

Native final target matching probabilities are recomputed after every update and choose one of the original final256boxes. Allfour TTA endpoints update parameters on all336; source weights are restored exactly. R55 used a narrow LoRA and different supervision, so this is not an isolated causal comparison of supervision alone.

## All registered results

| Arm | Drone mIoU % | Quad mIoU % | Balanced mIoU % | Source delta pp | Direct-consensus delta pp | Acc25 % | Acc50 % | Improve/worsen | Query changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | +0.000000 | +0.082700 | 40.773810 | 21.130952 | 0/0 | 0 |
| zero_step | 15.539892 | 28.120770 | 21.830331 | +0.000000 | +0.082700 | 40.773810 | 21.130952 | 0/0 | 0 |
| proposal | 15.358494 | 28.159150 | 21.758822 | -0.071509 | +0.011192 | 40.773810 | 21.130952 | 10/12 | 60 |
| decoder0 | 14.956935 | 28.366084 | 21.661510 | -0.168821 | -0.086121 | 40.476190 | 20.833333 | 7/13 | 52 |
| decoder1 | 15.442626 | 28.480090 | 21.961358 | +0.131027 | +0.213727 | 41.071429 | 20.833333 | 12/14 | 53 |
| decoder2 | 15.156297 | 28.461379 | 21.808838 | -0.021493 | +0.061208 | 40.773810 | 21.130952 | 9/11 | 41 |
| decoder3 | 15.292771 | 28.472876 | 21.882824 | +0.052493 | +0.135193 | 41.071429 | 21.130952 | 5/10 | 34 |
| decoder4 | 15.331095 | 28.494739 | 21.912917 | +0.082586 | +0.165286 | 41.369048 | 21.428571 | 4/4 | 23 |
| early_consensus | 14.935362 | 28.729296 | 21.832329 | +0.001998 | +0.084699 | 41.071429 | 21.130952 | 5/10 | 36 |
| final_role | 12.882300 | 27.205547 | 20.043924 | -1.786407 | -1.703707 | 37.797619 | 20.833333 | 5/30 | 89 |
| early_role_consensus | 12.975408 | 27.573534 | 20.274471 | -1.555860 | -1.473160 | 38.095238 | 21.130952 | 8/35 | 100 |
| phrase_subject | 13.926915 | 27.214722 | 20.570818 | -1.259513 | -1.176813 | 38.392857 | 20.535714 | 15/27 | 102 |
| phrase_role | 13.416863 | 25.516855 | 19.466859 | -2.363472 | -2.280772 | 36.309524 | 19.345238 | 18/43 | 124 |
| semantic_consensus | 15.370284 | 28.124978 | 21.747631 | -0.082700 | +0.000000 | 40.476190 | 21.428571 | 8/15 | 59 |
| shuffled_consensus | 2.517253 | 0.572121 | 1.544687 | -20.285644 | -20.202944 | 2.083333 | 1.190476 | 16/168 | 335 |
| tta1 | 15.908954 | 28.086002 | 21.997478 | +0.167147 | +0.249847 | 41.369048 | 21.130952 | 5/5 | 32 |
| tta3 | 13.996244 | 27.978873 | 20.987558 | -0.842773 | -0.760073 | 39.583333 | 20.833333 | 9/26 | 70 |
| shuffle_tta1 | 15.745436 | 27.918749 | 21.832093 | +0.001762 | +0.084462 | 41.071429 | 21.428571 | 5/5 | 24 |
| shuffle_tta3 | 10.664354 | 26.208466 | 18.436410 | -3.393921 | -3.311221 | 33.333333 | 20.238095 | 9/53 | 137 |

Retention of the one-step research candidate is based on the prespecified development results; it is not label-free or independently validated step selection. No prediction, consensus weight or update rule changes after labels. One step improves5IoUs and worsens5, changes32queries. Acc25 increases40.773810->41.369048%;Acc50 remains21.130952%. It exceeds source, initial direct consensus and the near-source shuffled one-step update (+.001762pp vs source). These are descriptive exposed-development differences, not a fresh statistical confirmation. Three steps lower task performance despite stronger target fitting.

Every earlier individual stage is below source on Drone and above source on Quad. The fixed early average is +.001998pp overall:Drone decreases about.604530pp while Quad gains about.608527pp. A unified claim that later fusion destroys correct earlier semantics is therefore unsupported. Decoder1/3/4 are weakly positive on balanced development means, but picking any of them after seeing this table would be development-label layer selection, not independent evidence.

Final-role readout and early-role consensus decrease balanced mIoU1.786407/1.555860pp. Subject phrase and phrase-role readouts decrease1.259513/2.363472pp. Retaining separate role/phrase outputs exposes their harm; the primary mixture was not retroactively repaired after seeing them.

## Is there correct earlier ranking? Offline diagnostic only

| Platform | Source mIoU | GT-best among six early Top1s | GT-best among source+six early | Source failed25 and some early Top1 succeeds | Source IoU improvable by early Top1 | Ref role score dominates sourceTop1 |
|---|---:|---:|---:|---:|---:|---:|
| drone | 15.539892 | 16.444560 | 16.573464 | 3 | 15 | 53 |
| quad | 28.120770 | 28.968971 | 28.968971 | 3 | 9 | 21 |
| balanced_all | 21.830331 | 22.706765 | 22.771217 | 6 | 24 | 74 |

A better early Top1 exists on24 observations; only6 can turn source failure at.25 into success. Source+early oracle is22.771217%, about.940887pp above source. This is a GT-selected offline upper diagnosis over only the prespecified Top1 alternatives, never an inference selector. It does not prove an unlabeled rule can find the right stage. The much larger full candidate pool has268/336 hit25;131 sourcefailures are recoverable somewhere in256, so most available selection headroom does not already appear as an earlier-stage Top1.

On74sourceTop1s, at least one length-adjusted reference role score exceeds target. These are score overlaps, not certified reference-instance mistakes. Direct role subtraction harms results, so the existence of role overlap alone does not justify suppressing those candidates.

## Optimization diagnostics

| Arm | Nonzero updates | KL lower | Median initial KL | Median final KL | Median parameter delta L2 |
|---|---:|---:|---:|---:|---:|
| tta1 | 336/336 | 282 | 0.596259 | 0.250143 | 0.195343 |
| tta3 | 336/336 | 325 | 0.596259 | 0.166703 | 0.393125 |
| shuffle_tta1 | 336/336 | 336 | 8.749527 | 5.895204 | 0.195346 |
| shuffle_tta3 | 336/336 | 336 | 8.749527 | 2.956526 | 0.505040 |

Real median KL falls.596259->.250143 after1step and.166703 after3steps; mIoU nevertheless goes21.997478->20.987558%. Stronger agreement with the current consensus is not stronger grounding. Partial fitting can preserve useful source decisions while copying some semantic information, but the experiment does not isolate implicit regularization, target quality and optimizer effects as a unique cause. No additional steps/LR/loss/layer sweep followed evaluation.

## Scene boundary

| Scene | n | Source | Early average | Main direct | TTA1 | TTA3 | Shuffled TTA1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D_S01 | 106 | 10.584991 | 10.146857 | 10.406311 | 11.313344 | 8.810942 | 10.645839 |
| D_S02 | 62 | 24.011174 | 23.122161 | 23.857076 | 23.765964 | 22.861436 | 24.464103 |
| Q_S01 | 62 | 10.117522 | 10.128883 | 9.147066 | 10.023311 | 8.751168 | 9.570112 |
| Q_S02 | 49 | 32.998900 | 35.070901 | 34.241251 | 32.998900 | 34.241251 | 32.998900 |
| Q_S03 | 57 | 43.509770 | 43.509770 | 43.509770 | 43.509770 | 43.509770 | 43.509770 |

One-step gain comes from D_S01;D_S02 andQ_S01 decline, twoQuad scenes unchanged. Five historically exposed clusters do not provide a new independent confirmation or frame-independent uncertainty estimate. Retaining one-step as a research candidate does not mean selecting it as a validated best step.

## Measurement validity and resource accounting

Predictions and all initial signals sealed before task-label access. Since final boxes/cohort/indices exactly match R55, the evaluation reuses its sealed256candidate IoUs, with exact original-box/source query/identity checks.6384method/sample values are derived from those old scalars; no new raw GT/metadata/geometry scoring call occurs. This remains label-assisted development evaluation. All8638ordered negative comparisons, all336observations, unavailable-phrase status and fiveclusters remain in denominator. Independent scalar reaggregation maxerror1.43e-14pp.

Independent all-row audit recomputes all consensuses, candidate permutations, reference transports and1344fitted endpoints;0Top1 disagreements, maximum native score error1.37e-06. Four hash-selected raw association cases verify9full affinity matrices exactly. Eight first-step gradient reconstructions maxerror3.81e-06. All original final boxes/source-zero outputs/source restoration pass.

One audit-only technical failure is retained: its first implementation divided Jaccard counts in float64 while the measurement uses float32. Four-case probe shows at most5.96e-8affinity difference with identical nonzero support. Attempt002 explicitly honors the original division dtype, with no tolerance or producer/prediction changes; it passes exact matrix checks. No model/inference run failed, no task labels were read before the repair, and no negative outcomes were filtered. Generic RoBERTa pooler initialization and deprecated point-operator tensor-constructor warnings are retained; strict trained source loading/fingerprint/output bridges pass.

New subject-phrase336forwards plus4full-description audit forwards; six-decoder component replays on336=2016;573old reference phrase outputs reused. Main capture continuation38.566s plus5.954s two-case engineering process; correspondence/consensus CPU139.363s; TTA continuation46.483s plus5.827s two-case engineering process.2016optimizersteps total, peak captureGPU692491264bytes, TTA601328128bytes. These are cached multi-arm program costs, not per-method end-to-end deployment latency. Existing R53 extraction and R51 reference-inference costs are dependencies, not free deployment work.

## Disposition and literature boundary

Do not adopt the primary fixed consensus or claim robust TTA. Preserve the measured one-step result as a weak development candidate and retain every negative arm. Internal semantics are not absent: earlier alternatives sometimes help, and1step gives a small net gain. The evidence does not support claiming an effective shared semantic feedback rule or that matching projection is a validated best layer.

End ad-hoc rule-loss exploration on the unchanged epoch100 checkpoint. A separately designed source-stage task-aligned adaptation preparation is a reasonable next research setting; no such training is started or scheduled here. Source-only preparation would change the checkpoint-setting and must be reported separately, without external visual teachers. A new fixed confirmation protocol would be needed before any one-step configuration replaces the current reference.

Original stage projection/query generation was checked in [official3EEDmodel](https://github.com/worldbench/3EED/blob/main/models/bdetr.py). The idea of preparing auxiliary updates to help a primary task has precedent in [Point-TTA,ICCV2023](https://arxiv.org/abs/2308.16481), which uses source meta-auxiliary training for registration. It is not a grounding result or a guarantee for arbitrary frozen checkpoints; this round neither reproduces Point-TTA nor claims novel self-distillation.

Only own code/configs and anonymous scalar task/fit/role/oracle/scene tables are public. Raw captions/media/coordinates/pointIDs/queryIDs/latent vectors/MLP weights and third-party source snapshots remain private.
