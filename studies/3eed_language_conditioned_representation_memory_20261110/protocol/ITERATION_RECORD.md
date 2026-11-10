# R64: language-conditioned representation correction memory

Hypothesis: frozen-source-keyed R59 representation corrections transfer across later inputs with compatible subject/context, exceeding same-information direct R59 readout. Memory is not identity labels or original-weight editing.

Locked: DEV336 only; three independently initialized orders, platform resets; predict before writing; 1024 FIFO entries, 16 per input, top-8, alpha .5, beta 5; no capacity/coefficient/order rescue selection. Fixed source, original 256 final boxes, R57 transport/UNKNOWN, R59 fusion. No FINAL384, source training, GPU, new losses or raw-media access.

Controls: source, ABC, R59, clear-every-input, past-value derangement with keys fixed, no explicit subject/context condition; R58 original endpoints reused and identical old online EMA/backtracking replayed on both new orders. Report every order and R58 pre/post separately.

TDA CVPR2024 official implementation stores class-keyed feature/pseudo-label caches, using current item before readout; our residual memory deliberately predicts before write and uses no pseudo-label. It is an inspired adaptation, not TDA reproduction. Sources: https://kdiaaa.github.io/tda/ and https://github.com/kdiAAA/TDA/blob/main/tda_runner.py .

Preparation: all 336 cached R59 score vectors reproduced bit-exactly; 72,758 pre-dedup supported candidate residuals. No task labels used in runtime/selection.

Decision fixed before scores: memory must exceed R59 and shuffled values on all three orders; otherwise stop this recipe without retuning. No fresh-generalization claim on repeatedly exposed five-scene DEV336.
