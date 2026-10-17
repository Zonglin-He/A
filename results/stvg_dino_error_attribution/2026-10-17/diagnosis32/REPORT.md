# DINO reference versus trajectory-propagation error attribution — exposed32

## Scope

CPU-only reanalysis of the same HC16/Vid16 previously exposed sources. All original predictions and negative results remain intact. No model inference, new Teacher call, parameter update, optimization search, critic retraining, or queue restart. The restored production DINO P7-A registration is unchanged.

Direct compares episodic Source boxes to final Direct boxes under the same Native interval. OPD-current compares inherited Before boxes to final P7-A boxes under the same frozen projection interval. These are valid within-method paired contrasts; Direct versus OPD is not an objective-only ablation because inherited state, objective and time readout differ.

## Independent critic recomputation

| Target | Best-time oracle gain on fixed Direct (pp) | Extra spatial-state oracle gain (pp) | Conditional time selection ΔtIoU (pp) | Strict same-time spatial pairs | Beneficial updates falsely rejected | Harmful updates detected |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 2.711423 | 4.509771 | 0.021128 | 45 | 3 | 9 |
| vidstg | 2.065703 | 0.055885 | -0.743727 | 30 | 18 | 0 |

Vid has 27 beneficial and 3 harmful strict Direct-versus-Source comparisons; critic accuracy is 9/30 and it rejects18 beneficial comparisons while detecting0 of3 harmful ones. Shared-source comparisons are not independent samples. The six-candidate oracle is a post-GT diagnostic, not an executable selection rule. The fixed critic remains stopped; these results do not evaluate the untriggered matched OPD extension.

## Fixed-time spatial attribution

| Target | Contrast | Mean ΔvIoU (pp) | Mean dense ΔsIoU (pp) | Negative vIoU sources /16 | Observed contribution to Δv (pp) | Unobserved native contribution (pp) | Benchmark-interpolated contribution (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| hc2 | Direct | -0.384918 | -3.475913 | 4 | 0.032485 | 0.043272 | -0.460675 |
| hc2 | OPD_current | -0.999466 | -3.591836 | 4 | 0.001156 | -0.018970 | -0.981653 |
| vidstg | Direct | 4.456877 | 11.100746 | 1 | 0.100110 | 0.673983 | 3.682784 |
| vidstg | OPD_current | 3.033090 | 5.490783 | 2 | 0.049014 | 0.463539 | 2.520537 |

The three contributions sum exactly to each per-source fixed-time delta. Observed means admitted supervision frames with exact GT. Unobserved native means actual Decoder frame-grid outputs with GT, excluding admitted anchors. Benchmark-interpolated frames are evaluated by the official interpolation rule; they are not new Decoder outputs or independent observations. Dense sIoU holds all annotated event frames fixed, including positions outside the chosen final interval. Missing exact anchor GT is unknown, distinguishing outside-event support from sparse annotation holes in SUPPORT_AND_PATH_DIAGNOSTICS.

Roughly97–99% of the negative sources' net vIoU loss is outside the few directly observed anchors in both methods. This is an accounting location, not proof that the references were correct or that interpolation is the causal failure. Incorrect references can also damage the rest of the trajectory.

## What actual pixels show

All32 private case panels were actually inspected: four planned anchors and up to two largest-absolute-change unobserved native positions per method. Original video hashes/domain-specific frame clocks and original DINO RGB SHA are verified. Review uses full original Query plus actual pixels and target annotation overlays; raw media, Query, coordinates and target identifiers are excluded here. These are model-assisted judgments, not independent human labels. Selection of changed frames is diagnostic and not representative random visual sampling.

The128 planned slots contain106 admitted references and22 abstentions. The admitted visual labels are61 CI (identity consistent and visible extent reasonable),8 CP (same identity but partial/uncertain extent),28 WI (visibly different annotated target/participant),6 MX (merged instances or person-plus-other-object extent),3 UN (uncertain). CI does not certify precise geometry, especially when exact anchor GT is missing. Role or annotation uncertainty is retained in the per-source notes.

| Source alias | Direct ΔvIoU pp | OPD-current ΔvIoU pp | Evidence relevant to harm |
|---|---:|---:|---|
| H002 | -22.797050 | -33.828243 | References switch between interaction participants; both methods damage previously correct unobserved positions. Mixed reference conflict and propagation cannot be separated causally here. |
| H003 | -10.170579 | -12.987795 | Repeated wrong-participant references; both final trajectories move to that participant. |
| H005 | -18.887042 | -22.810564 | Later references follow the wrong participant; both methods leave a previously well localized target. |
| H006 | 0.106291 | -0.938080 | Admitted references enclose another clothed participant. OPD harms, Direct is slightly positive; crowded annotation extent also uncertain. |
| H010 | -15.178174 | 0.492247 | Three early references follow counterpart. Direct harms while current OPD slightly improves; conflicting references do not force identical optimizer outcomes. |
| V014 | 1.261994 | -1.635867 | Same queried adult, but later references under-cover interaction/limb extent. OPD damages both observed and unobserved frames; Direct improves. Not a wrong-person case. |
| V015 | -0.869316 | -1.456028 | Different annotated participant is visibly enclosed. Query role phrasing remains ambiguous, so semantic-role certainty is unknown; both methods harm. |

All four HC Direct-negative cases and all four HC OPD-current-negative cases contain visibly wrong-participant reference evidence. This supports prioritizing reference/role reliability in this HC prefix. It does not prove that identity errors are the dominant failure on the full benchmark, that every bad anchor causes harm, or that an identity-particle model would fix it. Positive cases with mixed references (e.g.H009, V002) remain in the denominator.

V014 is a counterexample to explaining every harm as wrong identity: the one exact-GT anchor has DINO IoU0.518614; inherited-before IoU0.555106 falls to0.469163 after OPD. DINO agreement rises0.384628→0.839828 while dense sIoU falls11.525423pp and fixed-time vIoU falls1.635867pp. The available reference is not geometrically better than the inherited prediction at that anchor. Other anchors lack exact GT and have extent uncertainty. This is not clean evidence of correct references followed by purely unobserved-only damage.

There is real evidence of conflicting spatial effects during saved parameter updates. Across11 sources and50 saved OPD rounds, the exact-GT observed subset with reference IoU≥0.5 improves while unobserved native positions worsen. The threshold is a disclosed geometry diagnostic, not a semantic identity label or online acceptance rule. For example,V003 step3 improves observed IoU13.113629pp but reduces unobserved-native mean2.277655pp; V013 step6 changes them+0.035538/−0.759229pp. Both sources have identity-consistent references and retain positive final results. These witnesses establish nonlocal interference, not that it dominates final losses.

Dense-only losses are retained even when vIoU is unchanged: V006/V008/V011 include native-time misses and unobserved-trajectory degradation. In those cases exact anchor GT is absent; reasonable-looking reference identity alone cannot establish geometrically correct supervision. H004/H012/H013 also show local Query/pixel/annotation inconsistencies, retained as unresolved rather than silently changing labels or evaluation.

## Decision and remaining distinction

Keep the fixed joint critic stopped and do not launch a new identity model, critic, diffusion module or optimizer sweep. In this exposed HC prefix, unreliable role/object references explain the observed failure pattern more directly than a universal correct-anchor propagation failure. Sparse-update interference is nevertheless observed, and the Vid extent example prevents an identity-only account.

A causal distinction would require a separately authorized, matched reference-only intervention (same observation support and optimizer) and retention checks at unobserved native frames. No such intervention was run here. No GT-assisted reference filtering, online routing, best-step selection or revised method score is presented. The old512-window blind panel remains44 model-assisted/468 pending and is not closed by this separate diagnosis.

## Verification

Initial metric readback checks:704; independent one-frame-at-a-time support checks:576; actual DINO-input RGB hashes matched:124; inspected case panels:32. Full signed per-source numbers, zero updates, missing annotations, mixed/unknown visual judgments and positive controls are published. No private media or coordinates are exported.
