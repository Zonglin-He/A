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
