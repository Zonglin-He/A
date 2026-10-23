# Native temporal prior + Native2D + BoxFixed

## Production algorithm

The frozen original Source runs its unchanged native224 two-offset inference. Its actual final-layer FP32 start/end logits define a legal interval distribution for each offset; the original physical-time hull combines the two. This native posterior q0 is used BOTH to seed event/background prototypes and as the prior in the final update. There is no six-layer posterior aggregation, additional six-layer head evaluation or Global6 input.

The original projected appearance spatial grid BEFORE textfusion is observed without replacing any Source argument/output. An auxiliary complete BoxFixed trajectory covering the full video defines a fixed1.5 context per frame. Fractional area pooling yields native local content features. The same single ReferDINO/eight observations, shared initialization, complete expert-box propagation and exact path DP are unchanged.

For each offset, center/L2-normalize using the other offset, K=(1+cos)/2. Native q0 soft membership seeds two kernel prototypes, shrunk toward the fitting-fold unconditional mean with kappa64. Apply ONE update qnew(I) proportional q0(I) exp(beta SUM inclusive L_t), beta1, then original physical MBR and unchanged spatial selection at the resulting interval. Cross-offset fitting is not independent validation. Compatibility scores are not calibrated event probabilities.

No extra RGB encoding for feedback, separate motion/word-class head, network update, quality rejection, teacher or observation is introduced. Original Source still contains its original architecture; removing our Global6 does not remove native decoder layers or original internal signals.

## Selection and evidence

Latest user explicitly selects this native-prior variant AFTER its fixed cached ablation. Kappa64/beta1/ONEround/context1.5 are inherited unchanged, not reselected. Existing128parents/direction (dev96+fixed32) are exposed. NativeFeedback-Source pooled t/fullv: HC+1.2744/+3.3935pp, Vid+1.8090/+4.1882pp. Relative previous Global6+Native2D, pooled HC-.4958/-.2723 and Vid-.4294/-.0871; HCfixed32-2.9464t/-1.7407v with pairedCIs below0. All positives/negatives remain in results/native_prior_v1. This is a user-authorized engineering choice, not an equivalence or fresh-validation claim. Earlier kappa mean-protection failures also remain unchanged.

Final acceptance remains EACHdirection queryequal m_tIoU >= Source+.9pp. Complete official full is required; no guarantee, no full-result parameter reselection.

## Scope and interfaces

The shared feedback requires spatially supported native local representations and actual native temporal evidence, not a motion branch, word vocabulary, six decoder layers or fixed feature dimension. Only TA-STVG has been tested. Other backbones/MLLMs require their own validated adapters.

Global6 is retained only as a sealed original evaluation baseline. Source/SharedInit/Global6 and eight observation positions must reproduce archived official values. Full and its two spatial ablations are NEW native-prior outputs; no previousFull alias is allowed.
