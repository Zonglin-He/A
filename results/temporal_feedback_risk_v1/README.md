# Temporal feedback risk: two full domains, fixed nonvisual/simple controls

The user asks whether a Source-conditioned appearance prototype can reinforce an initially wrong event, and requests action-centric versus appearance-centric time gains plus length-normalized, temperature and smoothing comparisons. This study keeps the current main method unchanged and adds one low-priority CUDA-hidden CPU consumer on all 1902 HC / 10303 Vid queries, using already audited actual native-grid E ROI pooled features and the same main spatial candidates. No new Source/expert/model/optimizer inference is needed. Each temporal arm selects its own legal event and runs the same final trajectory DP with unchanged parameters. Whole-domain prediction seals precede separate official CPU scoring and separate full cold-array/parent/input/SHA/mean auditing.

The feedback is a single within-video self-consistency update, with self-inclusion. It is not an iterative process or independent semantic likelihood. A severe Source-to-current failure alone cannot identify its cause, because native per-offset MAP/hull and joint-MAP are different decoders. Source native, Source joint-MAP, and no-visual-update joint-MAP with the same final spatial DP are separate controls.

Fixed settings, declared before any new scores:

|Arm|Temporal rule|Purpose|
|---|---|---|
|Source|Original per-offset legal MAP physical hull, original boxes|Native baseline|
|SourceJointMAP|Original interval prior joint MAP, original boxes|Isolate Source decoder change|
|NoVisualUpdate|Original interval prior joint MAP, current spatial candidates/final DP|Isolate visual posterior update|
|FullMAP|Original prior times exponential of summed local evidence|Unchanged current method, exact checked|
|LengthMeanEvidence|Original prior times exponential of interval **mean** local evidence|Check evidence accumulation/length sensitivity|
|TemperatureEvidence2|Original prior times exponential of summed evidence / 2|Fixed weaker evidence, no parameter sweep|
|SmoothPrior3|Fixed `[1,2,1]/4` smoothing separately on prior start/end indices, original legal support masked, renormalized|No-appearance local temporal smoothness baseline|
|TemperaturePrior2|Square root of the prior, renormalized|Calibration-only invariant control: exact same MAP and final trajectory as NoVisualUpdate|

Smoothing uses the existing sampled endpoint grid, not a fixed physical-second window; sample spacing and support are recorded in the original binding. Length is the inclusive sampled-node count. These are explicitly specified simple controls, not claims of reproducing a named published algorithm. No setting is selected using full-set accuracy, and no current registry/manuscript/Fig1 is changed.

The text-only rubric is frozen before prediction/scoring and separates explicit action cues, visual attribute cues, both, and uncertain/static-action ambiguity. It is a reproducible operational stratification, **not independent gold semantic annotation**. All queries remain in overall results, including mixed and uncertain descriptions. HC has inadequate pure-appearance coverage; that contrast must be marked unsupported instead of filling the group with colored-clothing action queries. Text-only rubric development and inspected examples remain private with their hashes; no captions are public. Full semantic conclusions require the additional blinded annotation/sensitivity review, rather than treating a keyword label as ground truth.

The primary mechanistic contrast is FullMAP minus NoVisualUpdate tIoU by query stratum. Report parent-cluster 10000-bootstrap pointwise 95% CIs, the difference of action/appearance group deltas with joint parent resampling, counts/coverage, all signed t/v/R.3/R.5/boundary errors, short/long events, severe >5pp t/v losses and their union counted once, rescues, initially good/damaged cases and complete failure tails. Small/absent groups do not support a semantic contrast. Initial joint-tIoU quality bins and prior-MAP mass increases are strictly offline associations, never online GT gates or causal proof of self-reinforcement.

Four actual retained full-chain cases passed repeated exact outputs for all eight arms; current FullMAP/native/joint Source controls match audited originals and pure-prior temperature MAP invariance passed. Full-domain prediction is actually launched; official scores, independent full audit, root signed/group/tail/cost analysis, any figure inspection and verified two-ref publication remain pending. Cached conditional wall time is not process CPU or deployment latency; inherited Source/expert acquisition remains nonzero.

Existing EdgeTAM then ReferFormer GPU priority continues. CG-STVG, full gamma sensitivity and all corruption remain user-paused, and the additional full-spatial CG-HC control remains cancelled. This new temporal-risk study does not resume any of them.

## HC complete scored/audited/root analysis

All1902 HC queries have complete8-arm official scoring and independent all-array/parent/input/mean audit. [Full findings](hc2/ROOT_FINDINGS_v1.md) retain all means, pairwise CI, tails, prior-fixed negative, short/long and cost. Full minus same-joint/no-visual-update t+3.071255pp CI[2.341218,3.800142], v+1.501279pp CI[1.059534,1.941792], alongside460 severe t losses and465 severe t/v union losses. Fixed evidence temperature2 reduces union to356 with lower mean t; prior-fixed failure is0 under Full,65.189873% under mean/T2 and66.309013% under smoothing. This distinguishes readout-only explanations, but does not prove prototype self-reinforcement causality. The current method remains unchanged.

Blind text review found all six keyword HC appearance examples were actions. Independent POS annotation also had15/62 fine-label disagreements; all original labels and failures are retained. A separate conservative query-only sensitivity abstains on unknown vocabulary, with HC1778 event/124unknown/zero static-visual proxy and Vid2129event/2713static-visual proxy/5461unknown. Its38 retained reviewed examples agree with one AI review, not independent human gold. Static layout is not pure appearance; HC action-versus-appearance difference is unsupported. Scalar sensitivity PNG was actually inspected before exporting SVG. Vid numerical prediction continues; no Vid whole scoring/audit or final semantic contrast is claimed here.

## Vid complete: official scoring, independent audit, root analysis and figure

All10303 Vid queries/732 parents, eight arms,82424 official checks and20606 prediction/receipt SHA checks plus every cold array/input/mean passed. Two-process official scoring455.488651418s and independent audit420.308667335s are complete; original numeric code/predictions unchanged. Root numerical, conservative semantic, complete signed tails and actual cost analyses completed. [Complete Vid findings](vidstg/ROOT_FINDINGS_v1.md), [PNG](vidstg/figures/VidSTG_temporal_feedback_risk_v1.png) and [scalar SVG](vidstg/semantic_diagnostics_v1.svg) are available. PNG was actually viewed and directly authorized for publication; no raw RGB/caption/GT-coordinate/feature/mask/weight is disclosed.

Versus the same joint-MAP/no-visual-update spatial branch, Full gains t+1.467386pp CI[1.057285,1.888033] and v+.686752pp CI[.481305,.895982], while1947 severe t losses/750 v losses/1957 union losses and2536 rescues coexist. Full also exceeds fixed mean-evidence/T2/smoothing averages, but T2 lowers severe union to1438 with lower average gain; mean evidence180 union nearly removes average gain. This is an average-benefit/risk tradeoff, not stability for every query or a causal proof of prototype self-reinforcement. No target-informed parameter change.

Conservative text-only groups2129 event/2713 static-visual proxy/5461 unknown are not gold semantics. Event-minus-static t-gain difference-.673423pp CI[-1.508570,.176488] crosses0; HC has no supported static counterpart. No action-specialization or equivalence claim follows. Pointwise parent-bootstrap intervals use exposed full targets, not independent unknown confirmation or familywise error control.
