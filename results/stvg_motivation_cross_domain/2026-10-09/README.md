# Sealed source-only cross-domain native error composition

TA-STVG and TubeDETR use two directions: VidSTG-source → HC2 validation and
HC2-source → VidSTG test. Each direction has128 historically exposed parent
sources, one query per parent and identical uniform-at-most64 RGB frames.
Eight real GPU qualification cells passed before512 formal native outputs;
all512 sealed before GT scoring. No PTD joint-SFT output is relabeled source-only.
Native preprocessing and final outputs are retained, with no test-time updates.

The predeclared T+/S− dominance hypothesis is NOT supported. Counts are
[31,19,44,34], [22,17,58,31], [17,21,29,61], [30,16,31,51] in order
T+S+, T+S−, T−S+, T−S−. Strict tIoU>.5 and fixed-GT-frame meanIoU>.5 remain.
All categories, registered contrasts,10000 paired-parent intervals and every
anonymous scalar row are retained. Missing support scores zero over all GT
frames. This is a descriptive historical panel, not fresh/full-benchmark evidence.

Root independently checked source/checkpoint/RGB/receipt integrity and3072
metric components over126272 frame/model observations, maximum error
1.2212453270876722e-15. Independent scalar audit reproduced104 values.
Code, locked protocol, anonymous scalars and the statistics-only panel are
public. Dataset pixels/query/GT overlays/raw predictions/weights remain local.

The human's subsequent filmstrip request is implemented in v7 using the same
data. Its expressly post-hoc conditional panel observes spatial failures even
after temporal success; it does not replace this negative primary result.
See docs/STVG_MOTIVATION_FILMSTRIP_V7.md and
results/stvg_motivation_filmstrip/2026-10-09. Reproduce all144 primary and
conditional scalar checks without models, labels, media or prediction payload:

```bash
python -B scripts/audit_stvg_motivation_filmstrip_public_v7.py
```

This closes only the finite figure evidence task. P1 GPU prediction production
is globally sealed, but its CPU scoring/root review and P2–P6 remain separate.
No overall paper completion is claimed; EATA and historical queues stay paused.
