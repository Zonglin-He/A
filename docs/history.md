# Method history

2026-10-22 — User selected the original BoxFixed + Global6 + original physical MBR
as the main method. 320 was rejected; 448 remains an accuracy/cost reference, with
no offset1 upgrade. Resume preserves the completed HC1902 and saved Vid6088 outputs
under the same frozen full plan. This cleanup does not change that plan or method.

Bounded HC64/Vid64 tuning and fixed32-parent development rechecks did not justify
changing tau1/observed.5/guard.1/link.25. Final and Untuned are aliases. The original
Source checkpoint, ERef strength/calibration and single ReferDINO/eight observations
remain unchanged. Existing full results are not used to select parameters.

2026-10-22 — User requested local and GitHub cleanup while full evaluation runs.
An isolated maintained tree was prepared. It exposes only the selected method;
Global6 moved from research scripts into the package. Legacy OPD, reward studies,
conditional time replay, alternate backbones and resolution trials are archived,
not deleted from history. Required original TA-STVG and ReferDINO sources remain.
CPU tests, saved-capture numerical compatibility and original-code checks precede
publication. No GPU/model/GT scoring is performed for cleanup. Active runtime files
and saved predictions are preserved; its source identity is not migrated or relaxed.

Historical details, negative findings and exact versions:
[pre-cleanup research archive](https://github.com/Zonglin-He/A/tree/archive/research-pre-mainline-20261022). Current verification:
[receipt](../results/verification.json). Publication is recorded separately after
remote tree and file bytes are verified.

Cleanup implementation published at `1448faaa4d88d1abb4c460fd68a66d80410c161a`
to main and research/stvg-opd-paper-hc2-revision-v2. Remote tree and all 250 files
(1,585,327 bytes) verified. Archive branch retains the original 852-file tree and
commit ancestry. Publication receipt: [docs/publication.json](publication.json).
The active full queue retained all 434 original source hashes and continued advancing.

2026-10-22 — Complete official full evaluation finished: HC2 1902/227 and VidSTG10303/732
sealed before separate CPU scoring. Both jobs scored; controller/worker/teacher/scorer
exited and shared GPU lock released. Full vIoU Source→Final is24.9948→26.7597
(HC,+1.7649pp, parent-cluster CI[1.2804,2.2751]) and11.7777→14.2685
(Vid,+2.4908pp,CI[2.1351,2.8504]). Unobserved sIoU gains1.7881/2.3067pp;
tIoU changes−.0666/+.9617pp. All severe declines266/1149 and every signed row
retained. This is ERef+BoxFixed+Global6, not OPD or spatial-to-temporal feedback.
No full-result selection or parameter changes. Independent publication audit verified
12205 prediction/receipt hashes, all raw Final/Untuned aliases, signed means/thresholds/
counts and 10000-resample parent-cluster intervals with zero aggregation error.
Original434 sources and all seals unchanged;24,410 official/dense scorer checks pass.
Actual costs split original/resumed worker counters; original imported capture cost is
excluded. Corrected generic scorer wording: Final aliases Untuned, so only one posterior/DP
is computed. [Complete report](../results/boxfixed_full_v1/REPORT.md).

Full results published at `5d8b2c09e0d2e0d62f3a01dec03de53684bcf3e6` to main and
research/stvg-opd-paper-hc2-revision-v2; remote tree and all274 files/6031045bytes
verified, retaining clean single-method code and archived ancestry. Exact238 public
frozen sources match c310af58;196 private installed runtime dependency sources remain
hash-bound, not exported. Publication adds no GPU/model/GT scoring.

2026-10-22 — User explicitly authorized mechanism-led HC2 temporal tuning followed
by a new full HC evaluation. Old published v1 remains immutable. New protocol in
outputs/temporal_hc_v2 locks a SHA-ranked official96-parent panel (64tune/32validation),
18 temporal profiles maximum: endpoint temperature, layer slope and duration tilt only.
Spatial weights/teacher/observations/ERef/precision/old hull/physical MBR fixed.
Old HC64 had only31 exact query IDs in HC1902 (62 parent IDs overlap); old temperature
1.5 improved old development but harmed prior32 recheck, so no blind temperature repeat.
New96 captures use unmodified old provider with exact old expert runtime fingerprint
and explicit evidence reuse; Source/ERef/native intervals bitwise checked against v1
before panel labels. No old resume hash exemption. Fresh temporal outputs have new
plan/code/receipt bindings. All official full outcomes were already exposed, so this
is disclosed target development and post-selection full rerun, not independent test.
Root supervises sequential tuning; one locked validation, no validation retuning.

2026-10-22 — HC temporal v2 bounded search completed:96 official captures sealed
(6 complete reuse/90 native Source captures/0 new expert forwards),64 tuning parents,
14 profiles, then one fixed32-parent validation. Selected tau1.5 improved tuning
t+.8584/v+.2973pp but validation failed t−1.5605/v−1.0859pp versus Global6.
Fixed-space control localizes the loss to time; short/already-good events suffer
interval overexpansion. No validation retuning, fallback selection, gate relaxation
or new full launch. Default and old complete published full remain unchanged.
32 CPU engineering checks passed. All signed positive/negative evidence retained.
[Temporal tuning report](../results/temporal_hc_v2/REPORT.md).
