# A0.1 cached fitability triage — completed / independently audited

**Decision: stop width/step tuning; next candidate is global-context conditioning.** Neither full-set scientific arm passed. The fixed hash-selected single-query engineering control reached cosine **0.9557962994**, exceeding the preregistered 0.9 threshold. No native inference or further experiment ran.

## Frozen scope and single-factor controls

Exactly the sealed A0 Train128 / 95 source parents and Dev64 / 16 exposed diagnosis parents; source-GT-derived privilege and directions. Fresh31/388 untouched. Frozen PTD was not loaded; no pixel decoding, new oracle, new labels or native inference. Feature_dim128/state33/evidence8/union256/radius .13545580427763146 unchanged. Single seed20260928, batch4, AdamW .001/wd0/clip1, sole query-global coefficient 1-cos, fixed terminal states. H256 only widens internal layers, input425→256. Q1 repeats the same hash-selected cached query four times per step; engineering control only.

## Terminal direction readout

| Arm | Parameters | Steps | Train mean | Train median | Dev mean | Dev median |
|---|---:|---:|---:|---:|---:|---:|
| A0 H128 (historical) | 107648 | 200 | .0298659168 | .0076936024 | .0242224497 | .0024013038 |
| W256-S200 | 247808 | 200 | 0.0298977158 | 0.0071399619 | 0.0248327495 | 0.0022607661 |
| W128-S2000 | 107648 | 2000 | 0.0412345896 | 0.0073095771 | 0.0200930473 | 0.0018424985 |
| Q1-overfit | 107648 | 2000 | 0.0102799611 | 0.0045341887 | -0.0025612945 | -0.0071937664 |

Q1 full-set columns are diagnostic transfer readouts, not a method score. Its fitted query terminal cosine is **0.9557962994**, terminal loss **0.0442037006**. Every arm retains all 128/64 defined directions; no query replacement or outcome selection.

## Loss and actual gradients

| Arm | Terminal Train128 mean loss | Terminal Dev64 mean loss | Last training-batch loss (before final update) | Last preclip gradient norm | Preclip norm min–max | Clipped steps |
|---|---:|---:|---:|---:|---:|---:|
| W256-S200 | 0.9701022842 | 0.9751672505 | 0.9661125243 | 0.100918010 | 0.008908179–0.380574346 | 0 |
| W128-S2000 | 0.9587654104 | 0.9799069527 | 1.0019062907 | 0.114157088 | 0.005258143–2.613516092 | 17 |
| Q1-overfit | 0.9897200389 | 1.0025612945 | 0.0451592207 | 0.917549193 | 0.084912680–9.257630348 | 1675 |

Gradient norms are worker-recorded preclip measurements, not an independent raw-gradient reconstruction. Q1 clips frequently (1675/2000), so describe bounded successful fitting, not effortless optimization. No loss/checkpoint selection.

## Validation and interpretation

CPU tests passed H128 seeded initialization/three Adam steps exact equivalence, width256 input/gradient scope, fixed hash and routing. Real long-run H128 step200 parameters **and Adam state** exactly match sealed A0. All 8 trainable tensors change in each arm; frozen basis unchanged, integer Adam IDs rebound to live Parameters and counters200/2000/2000 verified. Independent NumPy recomputed all576 saved terminal coefficient cosines and summaries; maximum error 2.0872192863e-14. Old caches/pins remain unchanged.

The registered decision favors global-context conditioning as the next discriminating change: the present local network can approximate this one oracle field but the bounded width/step rescues do not establish multi-query fitting. This does not uniquely isolate the cause of amortization failure, prove that every oracle field is representable, or establish native utility. No width512, more steps, architecture modification, trust gate, expert, OPD, full618/full447, fresh388 or target experiment was started.

## Measured execution

Three serial workers plus nonoverlapping wrappers: **107.401691 seconds**. Independent CPU audit: 7.536875 seconds. Historical cumulative GPU allocation/wrapper accounting: 72513.02124451008 seconds, cap=null. All workers exited normally; no GPU failure/replay. Research GPU idle. Disk free 130579345408 bytes, above8GiB; no research artifact removed. No new recurring automation remains for this completed short stage.
