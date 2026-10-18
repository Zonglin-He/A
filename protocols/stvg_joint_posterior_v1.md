# Source-trained joint event–trajectory posterior: finite pilot v1

New source-training route authorized 2026-10-18. Separate from original DINO P7-A and the failed source event critic. No production promotion, Qwen/full-data restart or parameter search. Existing exposed 32 target queries remain exposed, including the previous GT-oracle diagnosis.

## Data and locks

Reuse the exact source96 roster, parent disjointness, 36 train/12 source validation per domain and binding002 of joint_event_critic_v1. Transfer a source-HC model to target-Vid and vice versa. Full Query and fixed frozen Student visual features only. Target annotations are inaccessible to model stages; source supervision is packed separately on CPU. Reuse original all-admitted DINO anchors, including event-exterior observations, not the GT-supported oracle subset. Zero new DINO/Qwen calls. Recompute 128 frozen visual feature captures; verify Source boxes/time exactly against saved captures. Frozen maps are spatially average pooled8x8 and linearly resampled to32 physical full-context times; detail loss is a limitation shared by controls.

State: two legal interval logits plus32 event-relative box knots in logit normalized cxcywh. Source spatial targets use linear interpolation between actually annotated event-interior coordinates, never exterior GT. An unknown knot contributes no loss. Final event-exterior output retains Source boxes, without claiming a learned exterior identity track. Source event endpoints outside the observed clip are clipped for representability and counted, while official target evaluation retains original labels.

## Models and controls

Two source directions, each joint VE denoiser, same-size deterministic joint corrector and factorized temporal/spatial VE denoiser. Width64, two temporal transformer layers, full-Query cross attention, dynamic four-point spatiotemporal ROI plus full-context features. Factorized ablation blocks cross-state attention and uses fixed Native times for spatial ROI; it retains full video/query context. Architecture parameter counts match. The deterministic model uses the same inputs, architecture, epochs and optimizer steps, but learns Source/Direct-corruption correction rather than GT-centered denoising. Thus this is a budget-matched feasibility control, not a perfect single-loss-variable proof of diffusion superiority.

60 epochs, batch6, AdamW .0003/weight decay .0001, gradient norm1; checkpoint selected solely by fixed-noise source validation loss, seed20261018. VE sigma log-uniform .03..2. Spatial and temporal reconstruction blocks receive equal loss weight. No target tuning or retries based on effect.

A small source-only association head sees ROI and Query, never distance to Source boxes. Supervision is source geometric compatibility IoU>=.5 at actually annotated anchors. Other-instance and unidentifiable/no-match components are marginalized together. This is not a trained three-class identity discriminator. Outside-event source observations lack target GT and remain unlabeled. Gaussian variance is estimated from source-train compatible residuals, floor .0025. The observation likelihood is an inlier Gaussian / uniform-outlier mixture, marginalized at inference; event-exterior observations have an uninformative event likelihood. This initial model does not exploit exterior observations for identity propagation.

Four fixed-seed latent samples,16 prior evaluations each from sigma2 to.03. Joint score approximated by VE conditional denoising; observation gradient added with fixed .02 sigma² factor and coordinate cap .2. This bounded approximation is not an exact Bayesian posterior sampler or a direct implementation of SDA. Final latent mean is prelocked; multimodal identity hypotheses may be blurred. Deterministic corrector receives identical initial perturbations and number of evaluations. No oracle selection of samples. No external Actionness projection; original backbone internal action gates remain.

## Gate and parameter adaptation

Measure Source, all-anchor episodic Direct, joint diffusion, deterministic correction and separated diffusion on all32, sealed together before target scoring. Report v/t/sIoU, R.3/.5, full paired deltas/intervals, fixed-Source-time geometry and observed/unobserved-native positions. Actual learned cross-state derivatives checked numerically; nonzero coupling alone does not establish value.

The frozen joint diffusion posterior must exceed Direct mean vIoU AND tIoU in each target domain. This gate is a resource decision, not proof of significance, and is never used to select hyperparameters. If it fails, stop before target parameter adaptation, retaining the third control as not triggered. If it passes, source engineering qualification of actual Decoder Jacobian/512 temporal and1792 spatial interface must precede32 fixed posterior-score-guided Student fits. No external tube optimization is called parameter adaptation. Source and Direct are preserved as strong baselines. Eight-hour deadline, one existing GPU lease,16–24GiB RAM,1GiB swap, free disk>=30GiB. No automatic expansion of the small source training cohort.

## Primary references and interpretation

- Rozet & Louppe, Score-based Data Assimilation, NeurIPS2023: https://arxiv.org/abs/2306.10574 ; official implementation https://github.com/francois-rozet/sda . Transfer principle: learn a trajectory prior separately from inference-time observations. Our conditional finite STVG approximation does not inherit their exact dynamical assumptions.
- Janner et al., Diffuser, ICML2022: https://diffusion-planning.github.io/ ; official code https://github.com/jannerm/diffuser . Transfer principle: whole-trajectory denoising and guided conditions, not per-frame independent Gaussian perturbations.
- Sünderhauf & Protzel, Switchable Constraints, IROS2012: https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf . Transfer principle: uncertain association belongs in inference; residual rejection alone cannot establish query identity.

Small source data and the previous cross-domain critic failure remain material risks. This pilot cannot establish novelty, global impossibility or robust identity association. All complete positive/negative results, actual costs, missing conditions and stop decisions must be published anonymously with implementation and archive receipts. No raw Query/GT coordinates/media/tensors/weights.
