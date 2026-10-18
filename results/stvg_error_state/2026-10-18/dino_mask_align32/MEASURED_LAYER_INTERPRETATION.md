# Measured mask intervention and six-layer response

Paired A/B differ only in the native-attention exclusion. Same high content and exact same reference. New admission: HC166, Vid509; lost admission0 in both domains. This isolates an overall mask-interface effect in this original32 configuration; it does not prove that a particular self-attention head or one specific layer is the independent cause.

Layer1 A/B averages are very close. From layer2 onward A scores decline while aligned B scores recover. High native content alone did not preserve native confidence; matched exclusion changes that behavior. A/B mask intervention is applied at ALL6 layers, so these trajectories locate where observed differences grow, not separate per-layer causal contributions.

| Domain | Source | Layer1 | Layer2 | Layer3 | Layer4 | Layer5 | Layer6 |
|---|---|---|---|---|---|---|
| hc2 | native_high_score | 0.566205 | 0.567499 | 0.567828 | 0.573569 | 0.575514 | 0.586171 |
| hc2 | A_score | 0.059198 | 0.029652 | 0.021604 | 0.018679 | 0.017208 | 0.017514 |
| hc2 | B_score | 0.059641 | 0.062862 | 0.074655 | 0.097433 | 0.117084 | 0.130807 |
| vidstg | native_high_score | 0.605056 | 0.620532 | 0.623423 | 0.618476 | 0.617272 | 0.616901 |
| vidstg | A_score | 0.287730 | 0.163884 | 0.105693 | 0.090872 | 0.088010 | 0.087407 |
| vidstg | B_score | 0.289481 | 0.346830 | 0.364229 | 0.367120 | 0.375480 | 0.377138 |

## Predetermined previous low-score examples

The original H010/position20/sample5 and V010/position36/sample11 references are the same ones diagnosed previously, not selected by new GT gain. Their full layer sequences and actual student-to-native INITIAL reference IoU are in MEASURED_LAYER_DIAGNOSIS.json. H010 final A .007210 vs B .746272 (native .742227); V010 final A .038384 vs B .387763 (native .387945). Final geometry was already close under A, so score recovery is not explained by merely selecting a closer input. No GT identity claim follows.

## What the actual once update says

The interface hypothesis is supported: aligned exclusion restores substantial feedback. The method-increment requirement is not met. HC B-once vs E full vIoU -.218821pp, Vid -.047250pp; no >5pp vIoU-loss query. Both domains retain all32 roster/noop/unknown cases. No dynamic run because the prelocked two-domain E improvement criterion failed.

The strongest available accounting distinction is Vid: local accepted-policy teacher minus E +.668880pp [+.079536,+1.370920], but compressing the SAME feedback to a per-sample shared4 target gives -.022052pp versus E. Compression minus local teacher -.690932pp [-1.354750,-.173425]. This is on matched accepted anchor support with query means and zero contributions elsewhere. It is not complete tube vIoU. HC local teacher versus E is only +.018568pp with CI crossing0.

The final once update additionally averages the24 correction vectors and applies one shared correction to the full trajectory. Vid observed-frame sIoU changes +.080599pp (CI crosses0), unobserved actual Decoder-frame sIoU -.250204pp [-.422344,-.100760]. This retains a propagation-scope issue alongside compression; it does not prove that adding a local residual alone will fix deployment. HC unobserved delta -.288805pp (CI crosses0).

Decision: stop at the authorized paired mask experiment and actual one-update evaluation. Do not keep diagnosing score collapse as if mask alignment had no effect; do not equate restored confidence with new useful supervision. No new threshold, best layer, local state, propagation rule, training, data expansion or dynamic OPD has been run.
