# P1 pre-fit lock: fixed evidence loss and label-free rejection

The saved HC01 ordinary IoU-OPD path was read without new labels: mean teacher-anchor IoU improved from approximately .39457 to .46772, .50552 and .55193, then fell to .15276 on the fourth Adam proposal. Three anchors became zero-IoU. This is a concrete failure to retain agreement with the same reference, sufficient to test rejection of destructive proposals. It does not prove reference correctness.

Retain original checkpoints, eight exposed historical sources, source grid, QA intervals, Gaussian keys, original per-domain sigma/tau/learning rate, five Adam proposals maximum and 1792 query/LN parameters. Each arm resets to the identical source state. No online carry or hyperparameter search.

Controls: exact prior strict IoU arm; raw WHO/WHEN-decoupled unit-IoU; presence-supported unit-IoU. The latter isolates admission from weighting. Reuse any exactly identical fixed fit rather than rerunning it.

New loss: original IoU reward and softmax weights remain unchanged. Compute original per-anchor forward-KL control-variate loss, then mean(r_t * loss_t), where r_t is the same-ID mask presence fraction in the already fixed five-frame local window. All admitted centers have masks, hence 0 < r_t <= 1; a center and at least one adjacent same-ID pair are required. Raw inter-frame mask IoU is not used as a correctness probability. Reliability is uncalibrated. Common loss scaling is partly canceled by Adam; do not describe this as a linear step-size reduction.

Matched update validation: evaluate the weighted mean teacher-reference IoU after each actual Adam proposal using the current cached decoder. Accept if change is at least -1e-7 and finite. Otherwise restore both parameters and Adam state, retain the rejected proposal record, and stop remaining updates. No retry, line search, smaller learning rate, or GT-selected step. Execute full-model final re-decode and require equality to saved cached output. The loss-weighted arm without rejection isolates the validation effect.

Checkpoint every attempted update and preserve no-information paths. Report proposed, accepted and rejected counts separately. The rule measures agreement with an imperfect reference, not semantic correctness or validation on held-out frames.

Seal every prediction and method source before final GT evaluation. These eight sources are previously exposed development sources; no new independent-source confirmation or production promotion follows automatically.
