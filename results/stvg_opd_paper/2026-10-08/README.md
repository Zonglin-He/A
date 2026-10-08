# Fixed STVG-OPD paper suite: actual P0 confirmation

[P0 report](P0_REPORT.md) contains actual 128-parent, two-order, three-arm confirmation results. Frozen is the shared matching source readout. [Root statistics](P0_ROOT_STATISTICS.json), every anonymous compressed row and all negative outcomes are retained. This cohort is historically exposed and excluded from the current parameter search; it is not a fresh unseen test.

Reproduce the scalar statistics and matched contrasts without private research assets:

```bash
python -B scripts/audit_stvg_opd_p0_public_v1.py results/stvg_opd_paper/2026-10-08
```

Python/NumPy suffice for this public scalar audit. GPU replay and raw official evaluation require separately obtained source checkpoints, DINO and official/private inputs. Dense readout and saved Gaussian/Adam/state arithmetic were audited postseal. The decoder Jacobian was not independently reimplemented.

No video, query caption, annotation, model weight, optimizer state, parameter vector or raw tensor cache is distributed. P1–P6 are distinct later stages; prepared code and rosters are not evidence that those stages ran. EATA's unavailable direction/source-media/Fisher stays user paused.
