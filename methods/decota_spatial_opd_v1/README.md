# DeCoTA Spatial OPD

User-selected main method from 2026-10-06. Native temporal output is frozen.
Original Uniform4 observations and one frozen Grounding DINO provide admitted
Top1 feedback. A Gaussian policy on the logit coordinates of the native merged
spatial output draws 32 antithetic actions each round. Detached IoU feedback
weights a true Gaussian-likelihood update of the 1792 query/LN parameters.
The configured final round is the output; no best-step or best-sample readout.
Query residual and Adam reset per query; only the configured LN delta persists.

`configs.json` records whether each dataset configuration is provisional or
actually selected by the finite development search. User method selection is
separate from evidence of task improvement. The first locked configuration
had positive clean cross-domain confirmation on VidSTG, inconclusive total
gain on HC2, and harmful mean current-query correction on HC2. These findings
remain in the research archive. Search does not turn exposed validation data
into an untouched test set.
