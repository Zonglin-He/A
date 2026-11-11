# R72 — Expert-free geometry-verified candidate rejection and matching TTA

**Decision: do not adopt this fixed recipe.** Real raw RGB/LiDAR evidence, geometric support construction, native matching-MLP updates and same-information direct controls were executed. No new method improves task mIoU over source in any of the three cohorts. The bottleneck in this implementation is decision-relevant supervision coverage, not an unexecuted optimizer. This does not establish that every expert-free TTA route is ineffective.

## Fixed scope and actual execution

Waymo-only epoch100; original A256 boxes and source token readout; 2,304 existing R69 inputs (VAL1280 and TRAIN1024) plus exposed DEV336. The two platforms are equally weighted. All cohorts were previously exposed; this is a fixed-implementation experiment, not a fresh blind test. FINAL data/manifests, source retraining, additional full 3D networks, external image detectors, CLIP/SAM/Qwen teacher supervision and new source auxiliary training are excluded. A/B/C and R59 remain cached references. The R71 frozen CLIP reader is an external-knowledge comparator only: its vectors are first joined after the new evidence and head fits are sealed and never enter feedback.

## Overall task results

| Cohort | Method | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % |
|---|---|---:|---:|---:|---:|---:|
| VAL1280 | source | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | frozen_ABC | 11.2556 | 25.9656 | 18.6106 | 34.3750 | 17.0312 |
| VAL1280 | R59 | 11.4835 | 26.5171 | 19.0003 | 34.6875 | 17.5000 |
| VAL1280 | CLIP_reference | 13.3208 | 27.6951 | 20.5080 | 36.4844 | 17.6562 |
| VAL1280 | graph_direct_mu1 | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | graph_direct_opt | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | graph_direct_hard | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | graph_tta1 | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | graph_tta3 | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | RGBshuffle_direct_opt | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | RGBshuffle_tta3 | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | Nshuffle_direct_opt | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | Nshuffle_tta3 | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | box_direct_opt | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | color_only_direct_opt | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | relation_only_direct_opt | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| TRAIN1024 | source | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | frozen_ABC | 16.6737 | 19.4682 | 18.0709 | 35.4492 | 16.8945 |
| TRAIN1024 | R59 | 16.8304 | 19.1693 | 17.9998 | 34.8633 | 16.6992 |
| TRAIN1024 | CLIP_reference | 17.8041 | 23.1767 | 20.4904 | 38.4766 | 17.8711 |
| TRAIN1024 | graph_direct_mu1 | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | graph_direct_opt | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | graph_direct_hard | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | graph_tta1 | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | graph_tta3 | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | RGBshuffle_direct_opt | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | RGBshuffle_tta3 | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | Nshuffle_direct_opt | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | Nshuffle_tta3 | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | box_direct_opt | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | color_only_direct_opt | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | relation_only_direct_opt | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| DEV336_DIAGNOSTIC | source | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | frozen_ABC | 15.7363 | 28.8686 | 22.3025 | 42.2619 | 21.4286 |
| DEV336_DIAGNOSTIC | R59 | 16.3856 | 28.8367 | 22.6111 | 43.1548 | 21.7262 |
| DEV336_DIAGNOSTIC | CLIP_reference | 18.6357 | 30.8080 | 24.7219 | 44.0476 | 22.0238 |
| DEV336_DIAGNOSTIC | graph_direct_mu1 | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | graph_direct_opt | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | graph_direct_hard | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | graph_tta1 | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | graph_tta3 | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | RGBshuffle_direct_opt | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | RGBshuffle_tta3 | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | Nshuffle_direct_opt | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | Nshuffle_tta3 | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | box_direct_opt | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | color_only_direct_opt | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | relation_only_direct_opt | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |

Direct rejection (fixed penalty, exact distribution optimum and hard exclusion), the true-RGB TTA1/TTA3, wrong-RGB control, negative-set shuffle and color/relation ablations all have exactly source task scores. One TRAIN query changes under matching TTA, with unchanged IoU. Simple box-support direct reading changes one VAL and one TRAIN query, also with unchanged IoU. Therefore absence of an IoU change must not be described as absence of parameter or probability changes. All frame/scene rows and all ordered negative method comparisons are retained, including losses against ABC, R59 and CLIP.

## Evidence coverage and the real bottleneck

| Cohort | Target color parsed | Any graph-supported candidate | Supported color request | Nonempty rejection set | Rejected groups / queries | Source Top1 rejected | Relation rejection |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAL1280 | 681 | 797 | 435 | 1 | 1 / 1 | 0 | 0 |
| TRAIN1024 | 482 | 526 | 280 | 4 | 5 / 7 | 0 | 0 |
| DEV336_DIAGNOSTIC | 172 | 193 | 110 | 0 | 0 / 0 | 0 | 0 |

Overall 1,335 / 2,640 expressions have one literal target color; 1,516 inputs have a supported surface hypothesis, but only five (0.1894%) produce an accepted contradiction. Those five reject six groups / eight queries, none the source Top1. Their normalized source rejected probability mass ranges 0.00001355–0.00693676. The raw cached source scores are positive per-query entity scores and are not a candidate distribution; all rejection-mass and KL calculations normalize over the 256 candidates. Source Top1 is unchanged by normalization.

Hard removal of every accepted negative still leaves the original winner. This is stronger evidence of a coverage bottleneck than a small-gradient argument. Matching TTA can propagate changes outside the rejected candidates, but that produced no task improvement. There are 586 VAL, 439 TRAIN and 131 DEV source Acc@25 failures with an Acc@25-capable candidate in the pool; none of their source winners is covered by this rejection signal. These are postseal localization diagnostics, not labels used to select supervision.

No rejected query has target IoU > 0.25 (a localization proxy). This tiny eight-query observation does not certify correct semantic identity, color annotation or independent instance purity. In particular, a correct physical target can have a poor predicted box. We did not obtain a human/GT instance-mask certification of the surface components and do not claim high-precision semantic rejection.

## Global surface support, not another whole-box color mean

Raw calibrated LiDAR points are projected to RGB with positive-depth/FOV checks. Nearest depth per 4-pixel cell approximates visibility; uniform image-cell sampling caps the global graph at 4,096 points. Before any candidate membership, 12-neighbor 3D local normals and distance, depth-jump, image-gap and RGB line-boundary constraints form two fixed normal/RGB scales. Components represent local surfaces, not certified complete objects. Nearest-depth cells are not a full dense occlusion solution.

Existing candidate group representatives associate to these pre-existing surfaces using 0.85 box cores, 1.15 outer support, component containment >= 0.8, core coverage >= 0.7, >=12 support points and image-span coverage >=0.35 on both scales. Full-box projection must be >=0.99 inside the view. These checks support consistent local measurements; they do not prove that a predicted box contains only one true instance. Mixed groups remain a risk. Visible-point centers/extents never become full-object centers/boundaries.

Color contradictions require all retained components to support a sufficiently distant alternative hue while all admissible supports have low requested-color evidence. Whole-object black/white/gray absence is UNKNOWN because windows/shadows and partial visibility can mislead. Explicit upper/lower clothing phrases permit a restricted black/white contrast; unscoped person colors are UNKNOWN. Complex attributes, texture semantics, negation and multiple colors are not guessed. Fixed 0.55 opposite / 0.15 requested / 0.35 margin tests and normal/boundary scales are pre-GT choices, not task-label optimized thresholds. No threshold sweep follows this negative result.

## Official view convention repair and remaining parser coverage

The initial implementation required literal image/camera words for left/right, incorrectly ignoring the dataset-wide camera-view convention. Before any R72 task lookup, this was checked against the [3EED primary paper, annotation alignment appendix](https://arxiv.org/html/2511.01755v1). The original sealed extraction is preserved; a separate protocol repair enables the documented observer-camera frame and rejects chained-role syntax rather than guessing its subject. No task score chose this repair.

There are 1821 literal left/right-of matches, but most are absolute image-region expressions or contain syntax outside the simple target-to-named-anchor parser. Only 5 have an accepted simple entity relation, only 2 have a usable single-color reference phrase, and zero pass the complete anchor-witness/relation rejection. Thus this recipe has not demonstrated a useful language-relation path. It also has not tested every spatial fact present in the descriptions: absolute image-region cues and richer role/compositional parsing are unimplemented, not physically absent.

Reference hypotheses come from source query/token scores on the literal reference span within the same full-description forward, aggregated into deduplicated groups. They are not oracle anchors or a separate trained reference detector. All groups covering 95% source reference mass must be independently supported and color-compatible; otherwise the relation is UNKNOWN. Opposite relation requires disjoint full projected-box intervals with a 2% image-width margin and low 3D overlap for every retained anchor. Candidate incompleteness, source semantic bias and the color requirement remain strong coverage limitations. Missing references never make the target a negative.

## Genuine parameter adaptation and strong same-information controls

The actual frozen-prefix matching MLP is 288→288→288→64 (184,960 parameters). Each input restores source weights and Adam state. The accepted negative set is fixed before fitting. Adam lr=0.0005, gradient clipping=1 and fixed 1/3 steps optimize -log(1-P(N))+KL(P0||P). No GT stop rule, layer search, pseudo-positive target or external teacher is used. The readout is softmax(log P0 + native log-odds change relative to the initial CPU head), giving an exact zero-step source bridge. Boxes and all feature/backbone/decoder/token parameters are fixed. Parameters can change predictions of unsupported queries: that propagation is measured, not claimed to be independently supervised.

Optimization limitation: the fixed Adam budget is not monotone. In the main graph branch, 10/15 individual steps increase the same objective; the two shuffled branches each have 8/15 such steps. Parameter changes have positive L2 norm, so the optimizer executed, but this is not a demonstration of sufficient fitting. The exact no-update optimum below separates that optimization limitation from the decision-coverage problem: even an optimal distribution solution and complete hard removal cannot replace the source winner with these accepted negatives. No post-GT learning-rate/backtracking grid was used.

Direct controls use the identical negative set: fixed log-score penalty mu=1; hard removal; and the exact optimum of that same reject+KL objective. With initial negative mass a and lambda=1, the latter has negative mass a/2, preserving source proportions within both strata. It is a stronger no-update comparator than an arbitrarily weak penalty. It supplies no positive identity labels.

Exactly 5 main inputs and five inputs in each control branch execute updates: 45 actual optimizer steps; 30 saved 1/3-step head states independently replayed. Every inactive input is retained as a source fallback. The graph branch has five active inputs out of 2,640, not 2,640 successful adaptations. Wrong RGB permutes component color measurements among supported groups while retaining true geometry/support; it tests color binding only, not complete RGB removal, and does not preserve negative-query probability mass. Negative-set shuffling preserves rejected group count among supported groups, not necessarily query count or probability mass. No method separates from these controls in task metrics, so the round does not establish true correspondence dependence.

## Literature fidelity and limits

The design is inspired by mechanisms, not a reproduction of those papers. [Geometric Set Consistency](https://arxiv.org/html/2203.15361v1) uses reconstructed geometric sets and set-level representation pretraining; our fixed single-frame surface graph does not inherit its learned instance discrimination. [DARTH](https://arxiv.org/html/2310.01926v1) adapts a source-trained tracking detector/embedding system with detection and patch-association objectives; this 3EED has neither that source task preparation nor verified temporal identity. [Test-Time Adaptation for Depth Completion](https://arxiv.org/html/2402.03312v1) relies on source-trained adaptation/proxy machinery; no random auxiliary head was added here. Object/relation decoupling is a useful modeling distinction, but supervised grounding results do not guarantee an unlabeled relation witness. Consequently the initial assertion that useful expert-free supervision certainly exists is not established by these references or this experiment.

## Measurement, costs and final decision

All 2,640×16 = 42,240 readouts were sealed before the initial join to previously saved candidate IoUs; the status-only repair repeated this deterministic saved-value join once with byte-identical prediction arrays. New raw GT/3D-IoU scoring passes: zero. Independent CPU replay checks every output, fixed box and control; the 28 hash-selected raw-input graph/support replays span all cohort/recording strata. Independent NumPy matching replay maximum probability error is 8.63e-07. Separate audits verify 43,904 scalar/scene checks, 17,071 ordered negative comparisons and 38,770 method/frame/coverage checks. A single developer authored these independent programs; this is independent numerical execution, not a separately blinded human audit.

Marginal CPU raw-evidence construction 146.906s; camera-contract re-evaluation 14.581s; matching fits/cache replay 13.507s. No new full 3EED or CLIP forward is needed because the exact frozen-prefix/baseline captures already exist. These cached marginal times are not end-to-end deployment latency or a comparison of complete inference costs. No peak RSS measurement was collected; GPU usage for new experiment: zero. Data parsing/projection inherits the declared zero-distortion assumption when metadata omits coefficients.

Two pre-score implementation failures are preserved: the first prediction attempt incorrectly assumed raw source per-query scores sum to one; only the validation/normalization of inactive TTA readouts was repaired, while source baseline scores and all Top1 decisions were preserved. A first postscore descriptive Counter accumulation used NumPy booleans and saturated some coverage totals; the independent coverage check triggered integer accumulation and regeneration of descriptive tables. Neither repair alters evidence thresholds, weights, final predictions, saved task values or adoption criteria.

Stop this exact rule-negative recipe rather than adding optimizer steps, weakening gates after GT inspection or describing fallback as stable TTA benefit. The fixed graph supplies some local measurements but no evidence capable of rejecting the current wrong winners. This is not a clean precision-versus-recall tradeoff estimate: accepted semantic identity remains uncertified. Frozen CLIP remains a stronger external-knowledge system reference from R71, not proof that external models are necessary for every TTA method. A distinct future experiment would first need an independently checkable increase in decision-relevant identity/role evidence or source-trained auxiliary-task alignment; neither is started here.

Public export contains owned code, fixed configuration and hashes, anonymous per-input/scene/frame task scalars, every negative comparison, coverage and optimizer statistics and audit receipts. Raw RGB/LiDAR/descriptions, camera coordinates, original IDs, point/query memberships, feature vectors, actual weights, credentials and third-party source snapshots are excluded. Full reproduction requires official data, the source checkpoint and referenced earlier owned-code dependencies.

Status-only audit repair: per-arm fallback flags initially followed the main graph gate even for the box-support ablation. The correction changes 856 status records, reuses byte-identical prediction NPZ files and verifies every query/task IoU is unchanged; original scores and audits are preserved. Source/ABC/R59/CLIP reference statuses now inherit their original R71 records. Public export attempt 1 stopped at a copied old guard allowlist; attempt 2 was cancelled locally for the status correction before any network publication. The final package includes the corrected metadata and complete scans of all 35,042 saved privacy tokens. Multi-pattern matching uses the already-installed Aho-Corasick engine with exact original boundary semantics, checked against the original regex/substring definitions.
