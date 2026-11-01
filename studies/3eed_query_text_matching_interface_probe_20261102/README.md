# R33 — bounded query/text matching-interface comparison

Read REPORT.md. Same Waymo-only epoch100, frozen source feature caches, all22
R29 preferences from19 observations, originalU96/93 keep rows, beta1, exact
noun log-odds and sourceq0. query/text288-288-288-64 MLP x lr1e-4/5e-4,
AdamW weight_decay0,16steps, platform-independent source reset:8states128steps.
The text_h input is point-conditioned cross-encoder output; no RGB is added.

Only the original R30 query5e-4 satisfies the prespecified feedback feasibility;
it also wins balanced exposedDEV mIoU25.090904 vs best text23.712580. Even
without feasibility the text branch does not improve DEV. No new U128 check,
no extra sweep, no general claim text-side adaptation cannot work.

All1088 A/DEV scalar task outcomes and88 preference states, including negatives
and zeros, are exported. RawIDs, geometry, captions, media, projection states,
features and gradient vectors stay private. The CPU audit does not retrain;
saved-query state bitexactly equals R30 on both platforms. Unselected text
heads have no new full-model GPU closure, only isolated-module CPU replay.

Gradient source/endstate diagnostics find norm-weighted negative-dot energy
<=0.013696percent despite roughly half negative pair angles. This does not
exclude intermediate trajectory conflicts and does not justify a PCGrad trial.
No feedback was filtered from these diagnostic values.

Code composes with prior R24/R29/R30/R32 studies and licensed local 3EED
assets/environment. Rebuild the original private feature schema before running:
interface33.py -> sealed predictions -> select33.py -> independent CPU audit.
No third-party code snapshots, assets or model weights are redistributed.
TPT/RLCF/PCGrad are audited mechanism references, not reproduced baselines.
