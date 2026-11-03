# R54: direct correspondence optimization versus last-decoder Q/K TTA

Decision: do not adopt the tested parameter TTA or direct correspondence solver. This round actually updates Q/K on all336 inputs; it is not another frozen-G test. Source balanced DEV mIoU is 21.830331%, Q/K3 is 21.795440% (-0.034891 pp), and the stronger20step direct solver is 21.830331%. True-RGB Q/K3 changes one prediction and worsens its IoU. The RGB-shuffled Q/K3 changes a different prediction and gives only +0.000428 pp. Every arm has unchanged Acc25 and Acc50.

The negative result applies to this objective, reset protocol, last-layer Q/K interface and bounded optimization budget. It does not establish that all trainable latent binding or all 3EED TTA is ineffective. No source-G improvement gate was imposed before running TTA, and there is no source fallback/filter masking failed adaptations.

## Fixed protocol and source lineage

Waymo-only epoch100, original256 candidate boxes selectable, exposed DEV336 with168 Drone and168 Quad observations from two Drone and three Quad scene clusters. One checkpoint is used, with source fingerprint0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9. Model and optimizer reset per input. No teachers, EMA, external visual model, source retraining, FINAL384 access, full5912 scoring or causal point-deletion experiments were used.

R53 frozen source point/text/query banks are reused with their corrected, coordinate-verified seed provenance. The trained parameters are only decoder[5].cross_v packed Q and K rank4/alpha4 LoRA:4608 parameters. V, all biases, attention output projection, matching MLP, regression heads and all other modules remain frozen. The current query representations and native token-matching scores are recomputed after each update; final boxes always come from the original256.

The direct no-network-update control optimizes256x1024 additive attention-logit residuals shared across heads. These are262144 current-input correspondence variables, not trained network parameters. It uses the same native decoder value aggregation, FFN and matching head as Q/K TTA. It is a specified correspondence optimizer, not a fully unrestricted per-head oracle. Candidate-to-token scores are recomputed by both methods; the text roles used to construct the observation cost are held fixed.

Two768-key subsets plus the full1024-key bank are used. They refer to the same verified original raw point IDs, and the two subsets share approximately576 keys. Earlier-layer features and queries remain those of the full source forward. This is decoder-key subsampling, not fresh backbone sampling, a new scan, or validation on an independent observation.

## Raw observation and soft-assignment objective

Actual seed raw IDs are looked up in the existing calibrated/depth-visible pixel table. There is no nearest-color imputation or rectangular crop statistic. The median number of observed RGB seeds is878.5/1024, with776 minimum and1011 maximum. Three spatially nearest diverse box hypotheses join each expanded source box to create a local competition domain; these are hypotheses, not certified instances. R51 group membership and its old association matrix are not used.

The native trained64D query/token projections supply soft responsibilities for the unchanged literal subject/reference phrases. Raw HSV/Lab color evidence and local spatial/color graph affinities constrain query-point assignments. Unmeasurable complex attributes remain UNKNOWN; RGB supplies observable color/continuity rather than a complex semantic teacher.

For key subset v, a candidate/point assignment is constructed as S(q,k) proportional to n_v*A(q,k)*pi_source(q)*admissible(q,k), with one additional UNKNOWN term0.2 per point and column normalization. pi_source has total mass0.8 and is obtained from the initial native candidate scores. S and its UNKNOWN probability change during optimization. The UNKNOWN number is a constructed model assumption, not a calibrated 3EED confidence.

The common objective is RGB +0.25 geometry +0.1 graph +source-assignment KL +source-mass KL +0.25 shared-raw-ID view JS +relative-source-logodds MSE +0.01 centered realized attention-logit perturbation MSE. The mass term retains the source proportions, including UNKNOWN; it does not force equal instance sizes. Source graph normalization and RGB normalization scales are fixed for the RGB permutation control.

Adam uses lr0.01, betas0.9/0.999, epsilon1e-8, zero weight decay and gradient-norm cap1 for both interfaces. The initially tested checkpoints are1 and3 steps. Source probabilities, coordinates, priors, masks and observed costs are detached; only Q/K LoRA or the free current-input attention variables receive optimization. This is observation-driven correspondence adaptation, not entropy minimization or a V/P/R score target.

A fixed hash permutation among visible seed RGB values is applied to both the color unary and RGB graph affinity; visibility, geometry, text, point IDs and source initialization remain fixed. A Q/K3 no-observation arm removes RGB/geometry/graph terms, retaining source, view and preservation terms.

Before any task scoring, engineering loss diagnostics showed that3 free-variable steps reduced the objective less than3 Q/K steps. A single stronger20step direct control was therefore registered, using exactly the same objective/lr and the same RGB permutation. Its first3 optimizer steps reconstruct the existing saved3step variables exactly, including Adam history reconstruction. There is no target-GT selection or optimizer grid.20 steps are a bounded control, not a proof of global convergence.

LAME supplies the relevant precedent for output-side optimization with source-unary/graph regularization, but its original task is classification and its official solver is a fixed-point/concave-convex procedure. The present attention/UNKNOWN objective and Adam control are task-specific prototypes, not a LAME baseline reproduction: [paper](https://arxiv.org/abs/2201.05718), [official implementation](https://github.com/fiveai/LAME/blob/master/src/adaptation/lame.py).

## Overall task results

| Arm | Drone mIoU % | Quad mIoU % | Balanced mIoU % | Delta source pp | Acc25 % | Acc50 % | Improve / worsen vs source | Top1 changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| zero_step | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| free_1 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| free_3 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| qk_1 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| qk_3 | 15.470110 | 28.120770 | 21.795440 | -0.034891 | 40.773810 | 21.130952 | 0 / 1 | 1 |
| free_RGBshuffle_1 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| free_RGBshuffle_3 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| qk_RGBshuffle_1 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| qk_RGBshuffle_3 | 15.540747 | 28.120770 | 21.830758 | +0.000428 | 40.773810 | 21.130952 | 1 / 0 | 1 |
| qk_noobs_3 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| free_20 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |
| free_RGBshuffle_20 | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0 / 0 | 0 |

Improvement/deterioration uses continuous IoU delta tolerance1e-10. Accuracy uses the inherited official strict IoU>0.25/>0.50 thresholds. Both changed cases remain in their original accuracy-threshold categories. These are localization outcomes, not instance-identity gold. The tiny RGB-shuffled improvement is not positive evidence for correct RGB binding.

All free-variable arms, Q/K1 and no-observation Q/K3 select exactly the source candidate on all336 inputs. True-RGB Q/K3 selects a different candidate on one Drone observation; that observation changes from IoU0.442329 to0.325096. The shuffled Q/K3 changes a separate Drone observation from0 to0.001437. No case-specific repair, image selection or rerun followed these scores.

## Did optimization really happen?

Yes. Every Q/K arm has nonzero network-parameter changes on all336 observations. The source model is restored byteexact after the episodic resets, and V/other frozen weights have no gradients or value changes. The independent whole-case audit checks 3696 saved fitted states; all zero-step outputs match source bitexact, and no fitting arm uses a fallback.

The independent replay uses the official full last-decoder forward, rather than the training split-decoder helper. Four hash-selected observations, two/platform, replay all 52 endpoint scores bitexact. Independently recomputed objective values have max error 9.31e-09. Only Q/K LoRA receives network gradients; zero-initial B gives the expected zero A gradient at step1. True/shuffled RGB source gradients differ on all four checks, so raw RGB actually reaches the update. This does not certify useful semantic feedback.

| Arm | Final loss lower than initial | Median loss decrease | Median assignment KL | Median UNKNOWN before | Median UNKNOWN after | Median non-UNKNOWN retention | Median competitive-gap fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| free_3 | 336/336 | 0.00213350 | 0.00002642 | 0.955068 | 0.955414 | 0.991052 | 0.000245 |
| qk_3 | 320/336 | 0.00768863 | 0.00323347 | 0.955068 | 0.962314 | 0.894559 | 0.004450 |
| free_20 | 336/336 | 0.01001483 | 0.00082040 | 0.955068 | 0.957458 | 0.950994 | 0.001573 |
| qk_RGBshuffle_3 | 319/336 | 0.00520174 | 0.00251838 | 0.955068 | 0.963537 | 0.888098 | 0.003583 |
| qk_noobs_3 | 0/336 | -0.00047164 | 0.00002120 | 0.955068 | 0.954714 | 1.000422 | 0.000953 |

The competitive-gap diagnostic is max over challengers j of ((s_after(j)-s_source(j))-(s_after(q0)-s_source(q0)))/(s_source(q0)-s_source(j)). Reaching1 is required to overtake the original first query. It is post-seal descriptive analysis, not a stopping rule or feedback filter.

The20step direct solver lowers the objective on all336 observations but changes no final selection. Q/K3 lowers it on320/336 observations and has median competitive-gap fraction0.004450: the greatest relative challenger movement usually covers only a small fraction of the existing decision gap. Thus parameter updates and assignment changes are real, while the final grounding decision is nearly unchanged.

## What the current failure does and does not identify

1. It is not a missing-gradient, frozen-student-cache or zero-update result. Updated attention is passed through the actual value aggregation, decoder FFN and unchanged matching head. Independent official-decoder replay matches all audited updated scores.

2. The optimized observation energy is not established as a task-aligned supervision signal. Both measured RGB and graph/geometry terms can improve without identifying the described instance. The direct solver also exhibits this disconnect, so the issue cannot be assigned solely to the Q/K parameterization.

3. The constructed initial assignment has median UNKNOWN mass0.955068, and Q/K3 raises it to0.962314; median non-UNKNOWN mass retention is0.894559. Geometry and smoothness costs can partly improve by reducing candidate assignment mass. Source KL/mass penalties limit complete collapse but do not prove that the surviving mass belongs to the correct target. These values describe this constructed S, not the fraction of true background points or native model uncertainty. No point-instance truth was used to verify purity.

4. The current loss-to-output coupling remains weak. The final ranking-gap diagnostic is small even after nonzero attention changes. Preservation, source priors, the matching readout and the identity specificity of the RGB/geometry objective remain competing explanations; there is no isolated ablation proving a unique cause. A small preservation-loss value alone does not prove that its gradient influence is small.

5. Existing readout pitfalls were checked rather than assumed away: all336 source probability Top1 choices agree with stable logodds; no source maximum equals1 and all source maxima are unique. In post-score token overlap diagnostics, all336 positive readout masks lie inside the existing parsed subject span and never overlap a reference span. This checks parser/readout consistency, not correctness against annotated spans or identity gold. No method or predictions changed after this check.

## Mean weighted objective reductions

| Arm | RGB | Geometry | Graph | Source KL | Mass KL | View JS | Output keep | Logit keep | Total loss decrease |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| free_3 | +0.00119394 | +0.00023276 | +0.00117861 | -0.00003414 | -0.00001407 | +0.00000003 | -0.00000019 | -0.00000060 | +0.00255634 |
| qk_3 | +0.01667630 | +0.00149889 | +0.01105059 | -0.00474732 | -0.00139807 | +0.00000037 | -0.00116310 | -0.00123464 | +0.02068302 |
| free_20 | +0.00783630 | +0.00129085 | +0.00638272 | -0.00095898 | -0.00032575 | +0.00000009 | -0.00000266 | -0.00001837 | +0.01420419 |
| qk_RGBshuffle_3 | +0.01647855 | +0.00154396 | +0.00742441 | -0.00483618 | -0.00157993 | -0.00000004 | -0.00107060 | -0.00097336 | +0.01698683 |
| qk_noobs_3 | +0.00000000 | +0.00000000 | +0.00000000 | -0.00007494 | -0.00002746 | +0.00000016 | -0.00135083 | -0.00006199 | -0.00151505 |

Component reductions are initial-minus-final with the actual fixed objective coefficients. The RGB term is signed; median RGB energy near zero is not absence of RGB use. The no-observation arm assigns zero objective coefficients to RGB/geometry/graph even though those diagnostics can still be computed. One-step fit diagnostics retain pre-step loss terms; final post-step objective terms are available for3/20step endpoints.

## Scene boundary

| Scene | n | Source | Free3 | Q/K3 | Free20 | Q/K3 RGB-shuffle | No-observation Q/K3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D_S01 | 106 | 10.584991 | 10.584991 | 10.474393 | 10.584991 | 10.586346 | 10.584991 |
| D_S02 | 62 | 24.011174 | 24.011174 | 24.011174 | 24.011174 | 24.011174 | 24.011174 |
| Q_S01 | 62 | 10.117522 | 10.117522 | 10.117522 | 10.117522 | 10.117522 | 10.117522 |
| Q_S02 | 49 | 32.998900 | 32.998900 | 32.998900 | 32.998900 | 32.998900 | 32.998900 |
| Q_S03 | 57 | 43.509770 | 43.509770 | 43.509770 | 43.509770 | 43.509770 | 43.509770 |

These are five historically exposed clusters, with repeated scenes/observations; they do not support frame-independent confidence intervals or a new independent-confirmation claim.

## Execution, costs and technical history

All 4368 task cells are retained: 3696 new scored cells and 672 exact reused source/zero cells. All 24 directed negative comparisons, all336 observations, UNKNOWN and no-update rows are exported. Single GT scoring follows prediction seals, observation/method/scope audits. Independent saved-scalar reaggregation max error is 3.55e-15pp.

There are18480 actual local optimizer steps across all arms/observations:3024 network Q/K steps and15456 free-variable steps. The main program continuation takes 144.963s, in addition to the initial two-case engineering process; the20step direct-control process takes 212.366s. Main/control Torch GPU peaks are 749302784/690169344 bytes. These are whole multi-arm program costs on cached source features, not isolated per-method end-to-end deployment timings. There are no new full grounding/backbone forwards; source-bank extraction costs from R53 are an upstream dependency, not free deployment work.

The observation audit verifies 295811 real seed/pixel samples and 86016 local query domains over all336 inputs. Target/reference color-unary availability is limited:82 observations have an exactly zero semantic color unary, while RGB-aware graph affinities remain available. Learned RGB semantic encoding is not present.

Two technical failures are retained. First, Torch removal of packed-weight parametrization changes state_dict insertion order although all tensors are unchanged; the first engineering process therefore failed an order-sensitive final fingerprint check after two valid cases. A standalone lifecycle check verifies the cause. The fixed check hashes tensors in the original source-name order with exact keyset/value identity, and the two valid fitted cases are reused without refitting. Neither the objective nor predictions were changed. Second, the first observation-audit process completes its checks but fails to write its receipt because a helper is missing; attempt002 fixes only that writer, keeping the equations and tolerances. Both prior attempts remain preserved.

Generic RoBERTa pooler-initialization warnings occur before loading the complete trained source checkpoint. Strict state loading, source fingerprint and byteexact source output bridges pass; no randomly initialized fusion component is treated as pretrained.

## Disposition

Stop this fixed objective/QK1-3step configuration and do not promote the20step direct solver as an effective method. Do not spend FINAL384 or a new full-target evaluation on it. The experiment answers the user request with real parameter adaptation and a same-information direct optimizer, rather than withholding TTA until a G gate passes.

The next useful mechanism test is whether a support region has entity-specific influence on grounding relative to a matched neighboring-point intervention, with subject/reference roles kept separate and the same source model reused. This was explicitly deferred here. Such a test could distinguish generic context/smoothness from discriminative target support before another binding objective or update layer is introduced. No additional experiment or loss is scheduled in this round.

All private fitted Q/K tensors, free correspondences, raw media, descriptions, coordinates, original IDs, source features and checkpoints remain excluded from the public package. Public per-case fit diagnostics use new scalar-only row aliases, and all negative task outcomes are retained. Earlier interface study: [R53 immutable report](https://github.com/Zonglin-He/A/blob/ab3a4c8dfcc44682063eb135eae4a70ee4c2b706/studies/3eed_native_latent_point_rgb_alignment_20261103/REPORT.md).
