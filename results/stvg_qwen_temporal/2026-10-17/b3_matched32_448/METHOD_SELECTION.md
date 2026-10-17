# Native heads and frozen Teacher choice: fixed development32

Same exposed HC16/Vid16, not independent confirmation. No production/PAPER change. Each question below has its own matched comparator; rows from different temporal supports are not merged into a winner score.

## Parameter interface

Native Temporal256x2 is the previous B3 residual expressed at the actual native head input, exactly verified rather than treated as a separate fit. Native Spatial256x4 changes the parameter interface, freezing the old1792 Query/LN parameters. Same references, Gaussian/optimizer/acceptance. Fixed-T2-clock comparison:

|Adapter|HC vIoU %|Vid vIoU %|
|---|---:|---:|
|Original Query+LN1792|28.580257|14.572580|
|Native box-head residual1024|25.658712|13.862596|

Shared nonanchor influence/gradient/Source restoration and actual full-frame decoding are established engineering properties. This one fixed no-retuning configuration has lower mean geometry/localization than the old interface; it is not a demonstrated quality upgrade, nor proof all head adapters fail. Simplicity versus observed loss is an architecture tradeoff. Retain the original interface as the quality reference; no automatic LR/adapter search or promotion.

## Spatial Teacher, original Student and same T2 clock

|Teacher/use|HC vIoU %|Vid vIoU %|
|---|---:|---:|
|No spatial adaptation|27.487135|12.239461|
|Revised R1 Direct + original OPD|28.580257|14.572580|
|Qwen Direct + original OPD|28.840534|15.343443|
|R1 same-reference parameter-free D2|25.756999|13.222003|
|Qwen same-reference parameter-free D2|29.770524|15.043710|

Qwen Direct has higher OPD point means than R1 on this32, but paired intervals cross zero and source tails remain. Qwen D2 has a higher HC mean than Qwen OPD and lower Vid mean; no per-domain splicing or claim that parameter adaptation is already indispensable. Keep both uses as matched evidence, Qwen as a candidate rather than a confirmed method winner.

## Temporal Teacher, same B3 hypotheses and Student

|Temporal readout|HC vIoU %|Vid vIoU %|
|---|---:|---:|
|Original Frozen decoder|28.116723|12.185607|
|Source marginal MAP, no update|24.576445|11.914150|
|R1 B3|24.713955|11.925210|
|Qwen B3, matched eight448|24.390435|11.914150|
|Qwen tilted target, no-parameter hard readout|24.338231|11.914150|

This isolates the whole caption+alignment Teacher family under the same hypotheses/raw pixels/448 sampling resolution, reward task and Student. Native patch grids/tokenizers differ between model architectures; actual vision and text costs are reported, not assumed equal. The earlier Qwen128-resolution pilot is preserved as an input-budget error and excluded from the matched Teacher conclusion.

The correct adaptation comparison is each fitted Teacher version versus Source marginal MAP. A change from original Frozen also includes decoding. Full256 published marginal readout results and this development32 have different cohort means; never combine them or select a decoder from target GT. Query Consistency remains the intended mechanism but requires an incremental effect beyond the same-decoder control.

## Whole system and cost boundary

R1/R1, Qwen/Qwen, R1/Qwen and Qwen/R1 freshly linked Full are NOT all measured here. Spatial evidence is fixed to the previous T2 observation clock; new Temporal output would change acquisition times. Combining stored boxes with a new interval is a fixed-evidence diagnostic, not proof a freshly acquired single-Teacher pipeline works. Do not report a complete2x2 Teacher Full table or sum branch means into one. This round does not auto-launch mixed Full, cross-backbone, official fullset or new tuning.

COST.json files include actual calls/tokens/stage walls and failed engineering loads. Reused evidence is not zero historical cost; cold/hot and different inference backends are not a controlled latency benchmark. Source checkpoint and Teacher training-domain overlap remain separate from inference-time no-GT access. The semantic512 panel remains44 model-assisted/468pending; displayed input/output engineering panels add no independent identity/role annotations.

Completed prior public groups: [head32](https://github.com/Zonglin-He/A/commit/820c6a584c2ee949e08dc8fceb74b13d4b267c8e), [Qwen Direct32](https://github.com/Zonglin-He/A/commit/f49d9ab52d8d8fb90e1b4368fea557c3b9fd4c1d).


Measured Temporal decision: Qwen B3 minus same-decoder no-update is -0.186010pp HC (paired95 [-0.733094,0.361074]) and exactly0pp Vid on this16+16; versus R1 it is -0.323520/-0.011060pp.30 queries actually optimized,1500 accepted steps; equality of Vid task metrics does not mean parameters never updated. The Qwen uncertain posterior is0.680686/0.682228 versus R1 0.629674/0.593869; this fixed task did not reduce uncertainty. No evidence here to replace R1 Temporal by Qwen or to claim the semantic-distillation increment is solved. Keep B3 as the no-external-Actionness research interface with its honest same-decoder no-update control; Qwen Direct is a separate promising Spatial candidate. No full mixed-Teacher rerun or new module search is triggered.
