# User-requested restoration of the historical DINO P7-A method

The 2026-10-17 human attachment d5af5269 selects the historical dual-support
method, not Qwen Query–Caption adaptation and not the old direct L1/GIoU method.
The active Qwen official-clean service and its heartbeat are user-paused. All
Qwen partial files, claims, logs, science locks and completed development results
are retained. No old controller or official full-population queue is restarted.

The unchanged implementation is
`methods.stvg_opd_p7a_v1.predictor.P7AOPDPredictor`. Source-native I0 supplies
Uniform4 observations to the original frozen Grounding DINO tiny revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`. The original target/context token
scores, admission and per-frame Top1 apply. Source-frozen actionness projection
provides final I1; no Temporal parameter training occurs. Median/MAD center
fractions are HC1/Vid0.5, prior weight0.1, epsilon1e-6; the two physical offset
intervals form an envelope. Native observation and projected output supports
remain separate.

Spatial OPD retains1792 coordinates,32 antithetic Gaussian logit-cxcywh actions,
detached IoU reweighting, the original likelihood/control-variate implementation,
per-Query Adam and Query-residual reset, final configured round, and fractional
LN carry. No MLLM, DINO replacement, response solver, new rollback gate, new
optimizer, Prompt search or retuning is added. Fixed historical settings:

| Target | LR | sigma | tau | rounds | LN carry |
|---|---:|---:|---:|---:|---:|
| HC2 | .01 | .025 | .05 | 40 | 1/16 |
| VidSTG | .03 | .1 | .25 | 10 | 1/8 |

Validation fixes the first16 arrivals of original locked P0 order1 per target,
32 historical exposed sources total. Each target starts from Source and its
original online stream is maintained. This is implementation regression and a
finite requested validation, not independent confirmation or official rerun.
The original predictor/fitter bytes remain unchanged. Checks cover all1792
parameters, Source/expert freeze, QR and Adam reset, exact LN writeback and next
initial state, all Gaussian/IoU/Adam arithmetic, independent direct enumeration
of the actionness score and physical envelope, and complete two-decoder readout.
Both targets and all32 complete predictions seal before any CPU GT evaluation.
Frozen-native, frozen-projection, online-before, OPD-native and Full-P7A remain
in the same report, with all positive/negative/neutral/unknown rows and tails.

Entrypoint: `.conda/tubedetr/bin/python -B scripts/verify_stvg_dino_rollback_v1.py`
with explicit stages `prepare`, `worker`, `score`. Preparation and immutable
runtime/roster pins precede model work. Single GPU,20/24GiB host guard,Swap1GiB,
host6GiB/disk30GiB floors and at most1h for the32-case model verification. A
failure preserves its files and is not retried by changing scientific settings.
Root independently checks results, actual fixed-source pixel outputs, anonymized
public export, GitHub bytes/SHA/blob/mode/tree/main+research and archive closure.
Production and paper pointers are changed only after this bounded verification,
under the explicit human rollback instruction; historical registrations are
retained byte-for-byte in recovery. The request does not restart formal13,785,
robustness, budget sweeps, Qwen or unrelated paused services.
