# O1.1: Residual Scale Readout Test

User attachment 815909fb authorizes only this CPU readout diagnostic. O1 had zero
non-expert task gain, only1/24 changed selections, despite eight valid SGD writes.
Reuse the SAME32-source exposed VidSTG stream, SAME conditions/candidates and
SAME saved arrival states. Primary scope is the original24 non-expert arrivals.
Vid-source TA-STVG and UniversalVTG source evidence is exactly O1; no new model
or expert calls, no backbone, optimizer, updates, trajectory, data or GPU work.

For each arrival compute R=phi@w from saved FP64 raw768D phi and ARRIVAL w, not
post-update/final-stream w. Assert saved bias=0. Four fixed alphas [1,8,16,32]:
S(alpha)=ell_native+alpha*R. Candidate order and native-first ties unchanged.
Alpha1 must reproduce O1 score/selection exactly. Independent scalar dot-product
readback checks all24 residuals and96 selections. Immutable input hashes before
and after confirm original state/evidence unchanged.

Sequence: (1) select/seal all96 outcomes without teacher or GT-derived metrics;
(2) read already-cached Full expert choices for top1 agreement, then seal;
(3) read O1's sealed per-candidate task metrics, previously dual-implementation
verified, and gather selected tIoU/vIoU. No fresh GT file read or model scoring.
Report changed choices versus native, exact top1 agreement with Full Rerank,
task means and paired changes versus alpha1, harm tails and all positive/negative
individual changes. No outcome-driven alpha addition, deployment selection or
normalization rerun. Alpha is post-hoc readout amplification, NOT LR*alpha or a
new online state trajectory. Expert-position fast outputs would stay unchanged;
they do not enter the24-arrival primary readout test.

Means and descriptive paired bootstrap10000 seed20260929 in original stream
order. Intervals are conditional on the realized frozen trajectory, not stream
robustness estimates; these24 arrivals share states and are historically exposed.
A positive result supports scale sensitivity of this learned signal, not a fully
validated normalization method. If selections change without useful direction or
utility, simple amplification is not sufficient in this scope; no universal
Slow-Fast impossibility claim. No gates/prototypes/memory/LR search or GPU rerun.

Complete the archive and sanitized GitHub export, including negative findings.
Production and old experimental outputs stay unchanged. Runtime bound600s CPU,
new artifacts<=50MiB. Existing raw phi,w/media/annotations remain local.
