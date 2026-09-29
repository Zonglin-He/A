# A0.4 Fixed Shared-R16 Factorized Predictor — completed / independently audited

The original local H128 predictor did not learn the R16 coefficient field on Train128. The predeclared direction gate failed at200 and after continuation to2000 actual Adam steps. No PTD/native evaluation was run.

Only output128→256 changed to128→16. Frozen source features128, state33, evidence8, union256, Train-only B16, seed, AdamW.001/wd0/clip1, batch4 and query-global cosine remained fixed. 76688 trainable parameters. Source-GT cached targets; Train128/95parents and repeatedly exposed Dev64/16parents. Fresh31/388 untouched.

| Endpoint | Split | R16 cosine mean | Median | >.1 | >.3 | Full-oracle cosine mean | Full median |
|---|---|---:|---:|---:|---:|---:|---:|
| S200 | train | 0.027539446 | 0.006040455 | 12/128 | 0/128 | 0.024072848 | 0.005253112 |
| S200 | dev | 0.020124081 | 0.002461687 | 4/64 | 0/64 | 0.017776135 | 0.002219129 |
| S2000 | train | 0.036952286 | 0.006738156 | 16/128 | 1/128 | 0.032287899 | 0.006114088 |
| S2000 | dev | 0.017331723 | 0.002974831 | 3/64 | 0/64 | 0.015391630 | 0.002655709 |

Gate: Train median>=.3 AND Dev median>=.1. S200 Train<.3 authorized restoring the same model/Adam/RNG/sample-order trajectory for1800 additional steps. No new seed, best-step or fresh restart. At2000 the gate still fails; stop offline predictor architecture tuning. No rank32/width/step/attention search.

S200 terminal training batch loss .963999152; train/dev mean direction loss .972460554/.979875919. S2000 terminal batch loss1.000358611; train/dev mean direction loss .963047714/.982668277. A batch loss is not the whole training set loss. S200 gradient norm [.0170942,.868863], clip0/200; cumulative S2000 norm [.00621910,24.1491184], final2.000422, clipping503/2000. This optimization behavior is retained; the result is not proof that every offline function class is incapable.

All384 complete terminal coefficient fields were independently recomputed with NumPy. Maximum reported cosine/summary error2.77556e-15; full-cosine = R16-cosine × sqrt(projected-energy) error1.94289e-16 per query. FP64-projected target to FP32 rounding relative L2<=3.54255e-8. B16 equals A0.3 columns1–16 exactly; no basis refit, sign change, or Dev fitting. Hidden initialization equals original A0; all8 trainable tensors changed and all3 buffers remained exact. Integer Adam keys/live-Parameter state and actual counters1–2000, entire sample-order RNG and continuation passed. Three synthetic CPU controls are distinct from this real GPU execution.

S200 science worker finished and sealed normally. Its original wrapper then attempted to overwrite mutable ACTIVE using a write-once helper and raised FileExistsError. The original code/pins/status/failure remain. An isolated v2 launcher changed only mutable status handling and continued the saved trajectory; no GPU replay. The original receipt is child completion accounting, not proof of wrapper process exit0; the later traceback tail has no captured duration.

Measured worker and nonoverlap waiting/wrapper times: S200 5.210060369+1.084509022s, continuation 26.181450574+1.065882223s; total 33.541902188s. Cumulative ledger 72806.29816828508s, cap=null. Target preparation and independent audits are CPU-only and separately recorded. No research artifacts deleted.

A0.3 Oracle-R16 still retains its measured +10.938378pp Dev vIoU and83.3888% full-oracle gain; those are privileged source oracle results, not A0.4 learned results. A useful shared action subspace does not ensure that this local offline predictor can infer the required query-dependent THW coefficients.

Next candidate, not implemented/registered/run: R16 coefficient-field per-query optimization feasibility. A source supervised controllability check and a genuinely label-free TTA objective are different questions. Existing oracle targets cannot be described as available at test time. No trust/no-op gate is entered because direction learnability did not pass. No native/full447/full618/fresh/expert/OPD/target expansion.
