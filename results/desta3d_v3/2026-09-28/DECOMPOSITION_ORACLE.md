# PANEL16 Decomposition Oracle

Same16 exposed Vid training parents; one analytic GT-gradient correction, no optimizer/GTprefix/target. Query fixed, full shared F gradient; source-derived locked large radii. Joint union and unit-gradient balance; both single-tensor and two-pass energy budgets. Descriptive unadjusted CI; no claim of universal decomposition need, latent sufficiency, learned-reader or OPD success.

|condition|tIoU %|sIoU %|vIoU %|format failures|
|---|---:|---:|---:|---:|
|original|54.003027|61.929737|41.113359|0|
|T_only|52.424211|55.166345|40.698360|0|
|S_only|54.003027|64.256412|41.282710|0|
|decomposed|52.424211|57.999232|42.867907|0|
|joint|56.779628|58.817998|47.269088|1|
|joint_pass_matched|56.402833|66.662761|48.808980|0|

## Native comparisons

|contrast|delta vIoU pp|descriptive95%CI pp|positive/negative/zero parents|>5pp harm parents|
|---|---:|---|---|---:|
|decomposed_minus_original|+1.754549|[-11.646097, +15.422771]|8/7/1|6|
|decomposed_minus_joint|-4.401181|[-10.411040, -0.404300]|3/11/2|4|
|decomposed_minus_joint_pass_matched|-5.941072|[-12.342213, -1.730258]|4/11/1|5|
|T_only_minus_original|-0.414999|[-10.606067, +9.246022]|5/3/8|3|
|S_only_minus_original|+0.169351|[-6.476635, +6.658244]|6/9/1|5|
|joint_minus_original|+6.155730|[-9.292606, +21.858446]|9/6/1|4|
|joint_pass_matched_minus_original|+7.695621|[-6.063835, +21.050715]|10/5/1|3|

## Local geometry and finite objectives

Gradient cosine: mean **−0.004501**, median **+0.045818**, range **[−0.368121, +0.147946]**. Negative: **7/16**; positive: **9/16**; absolute cosine ≤0.1: **11/16**. Histogram bins [-1, −0.1, 0, 0.1, 1] have counts [3, 4, 7, 2].

|objective/direction|local descent positive/negative/zero|finite CE decreased/increased/unchanged|
|---|---|---|
|T / T|16/0/0|13/3/0|
|T / S|10/6/0|5/11/0|
|T / J|16/0/0|14/2/0|
|T / J_pass|16/0/0|14/2/0|
|S / T|10/6/0|6/10/0|
|S / S|16/0/0|8/8/0|
|S / J|16/0/0|6/10/0|
|S / J_pass|16/0/0|9/7/0|

## Scope and decision

Registered practical opportunity gate: **False**. A low cosine is not itself a conflict, necessity result, or native improvement.
Shared F is before frozen B1, with caption features fixed. Each gradient includes both identity and reader paths. This is not a pixel derivative and is distinct from the old post-adapter actuation controls.
The correction is one normalized projected initial-gradient step. T/S spans have128 columns each; Joint uses their256-dimensional union and equally normalized branch gradients. Frozen PTD/B1, complete native conditioning, no parameter optimizer, no best-state selection.
The user budget equality counts one shared Joint tensor once. It is injected twice. Joint-pass-matched divides it by sqrt2 to match actual two-pass squared exposure. Both comparisons are reported; neither implies globally optimal Joint or Decomposed adaptation.
Radii .087687 and .170316 times stock-F norm are inherited from previous source-selected controls. This single finite magnitude does not resolve other step lengths or optimization sufficiency. We do not add a grid.
All16 are exposed Vid training parents, not new independent examples. SourceGT supplies native action targets, never a GT-generated prefix. Spatial finite CE is on baseline student reference/interval/anchors; final native support may differ. Model query features/parameters remain frozen.
Full-vocabulary fixed-anchor coordinate readouts, native failures, good-case retention and every signed parent effect are retained locally; anonymous complete scalar aggregates are in DECOMPOSITION_ORACLE.json. No videos, captions, GT records, weights, or raw tensors are published.
Validated96 predictions,32 shared-F backward calls,128 fixed-support finite reads,0optimizer. Scalar/tensor288 geometry error 2.22e-16; independent summary 134 reductions error 2.274e-13. GPU allocation 614.718683s; cumulative 43107.875212573s, cap=null.
No directional module, external teacher, OPD, target or64 expansion starts from this result. The cancelled full-source run and scalar-mask STOP remain unchanged.

## Interpretation of the completed experiment

**The finite result opposes the proposed Decomposed-over-Joint claim in this locked configuration.** Decomposed minus Joint is -4.401181pp vIoU, descriptive95%CI [-10.411040,-0.404300]; against pass-energy-matched Joint it is -5.941072pp [-12.342213,-1.730258]. Decomposed's +1.754549pp versus Base has CI [-11.646097,+15.422771]. Thus local task difference/partial conflict is supported as a diagnosis, but it did not establish that these corrections should be deployed separately.

Seven gradient pairs have negative cosine, nine positive, and eleven are within absolute0.1. Both cross-direction local dots are adverse for6/16, favorable for10/16. At the finite locked radius, T correction increases S CE10/16; S correction increases T CE11/16. Self temporal CE decreases13/16 yet native temporal mean falls1.578816pp. Local direction, finite CE, and free native readout are separate evidence.

All96 final predictions remain included. Decomposed has6 parents with >5pp vIoU loss to Base, compared with4 for Joint and3 for pass-matched Joint; native-good v retention is5/7,5/7,6/7 respectively. P09 gains+69.921335pp under Decomposed and P13 loses53.627939pp; both are retained. Joint's one P03 native box-grammar failure is retained at official zero. Direct-endpoint-only tIoU for Joint is59.404628%, compared with official whole-grammar56.779628%; this diagnostic does not replace the primary metric. S-only event logits/completion equal Base16/16, a structural property.

This does not prove Joint is universally better or that decomposed adaptation is impossible. The comparison uses one large inherited radius allocation, unit-gradient Joint balancing, a256-dimensional shared union versus two128-dimensional branch spans, fixed query features, and one analytic update on16 repeatedly developed training sources. It does not establish small-step behavior, optimal budget allocation, or learnability. No follow-on grid or OPD is launched.
