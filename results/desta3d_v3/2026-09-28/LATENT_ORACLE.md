# Source GT privileged 3D branch-latent oracle

16 exposed source-training parents; GT creates latent masks; no optimizer/target/OPD. Not unlabeled TTA or held-out generalization.

Frozen official PTD4B+B1; alpha .25 after reader/LN/SiLU before output projection; RGB and gates unchanged

|Latent|tIoU %|sIoU %|vIoU %|
|---|---:|---:|---:|
|original|54.003027|61.929737|41.113359|
|temporal|52.217313|61.342371|40.819676|
|spatial|54.003027|62.147850|41.261938|
|dual|52.217313|61.708303|41.042165|
|wrong_temporal|52.173759|59.211256|40.025967|
|wrong_spatial|54.003027|62.294824|41.292610|

|Versus original|metric|delta pp|95% CI|positive / negative|>5pp losses|
|---|---|---:|---|---|---:|
|temporal|vIoU|-0.293683|[-0.8810489838537802, 0.0]|0 / 1|0|
|temporal|sIoU|-0.587366|[-1.7620979677075603, 0.0]|0 / 1|1|
|temporal|tIoU|-1.785714|[-5.357142857142858, 0.0]|0 / 1|1|
|spatial|vIoU|+0.148579|[-0.1309963328122239, 0.5094055967829741]|3 / 7|0|
|spatial|sIoU|+0.218113|[-0.2766225868062201, 0.8608940628097229]|3 / 7|0|
|spatial|tIoU|+0.000000|[0.0, 0.0]|0 / 0|0|
|dual|vIoU|-0.071194|[-0.75178684414475, 0.45846913171132697]|3 / 7|0|
|dual|sIoU|-0.221434|[-1.5677950823497477, 0.7746758923433673]|3 / 7|1|
|dual|tIoU|-1.785714|[-5.357142857142858, 0.0]|0 / 1|1|
|wrong_temporal|vIoU|-1.087392|[-3.2621766960060112, 0.0]|0 / 1|1|
|wrong_temporal|sIoU|-2.718481|[-8.155441740015029, 0.0]|0 / 1|1|
|wrong_temporal|tIoU|-1.829268|[-5.48780487804878, 0.0]|0 / 1|1|
|wrong_spatial|vIoU|+0.179251|[-0.18094716751448314, 0.6471595866797116]|4 / 6|0|
|wrong_spatial|sIoU|+0.365087|[-0.2768330763805438, 1.272683987419397]|4 / 6|0|
|wrong_spatial|tIoU|+0.000000|[0.0, 0.0]|0 / 0|0|

|Correct minus matched wrong (eligible)|metric|parents|delta pp|95% CI|
|---|---|---:|---:|---|
|temporal_minus_wrong_temporal|vIoU|13|+0.976873|[-1.0843679801277293, 4.014986702776628]|
|temporal_minus_wrong_temporal|sIoU|13|+2.622910|[-2.1687359602554586, 10.037466756941575]|
|temporal_minus_wrong_temporal|tIoU|13|+0.053605|[-6.593406593406594, 6.7542213883677285]|
|spatial_minus_wrong_spatial|vIoU|16|-0.030672|[-0.36518444111549225, 0.26879983917764805]|
|spatial_minus_wrong_spatial|sIoU|16|-0.146974|[-1.1105505228416197, 0.662680778768145]|
|spatial_minus_wrong_spatial|tIoU|16|+0.000000|[0.0, 0.0]|

S and wrong-S native comparisons share original reference, interval and frame support by structural contract. T/dual may change those conditions; their sIoU is final-tube evidence, not fixed-support spatial gain.

Full per-parent positives, negatives, failures, original-good retention and oracle support remain in REPORT.json.
This one fixed actuation interface does not establish all 3D information sufficiency, an external teacher, OPD benefit or deployment readiness.

Independent root aggregation/mask/sparse-merger audits passed. Original and all-one native controls reproduced exactly. Full local raw remains private; this public file contains aggregate results only.
