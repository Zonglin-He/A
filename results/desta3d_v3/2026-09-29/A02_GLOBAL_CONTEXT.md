# A0.2 Global-THW Mean Screen and conditional Oracle Field Structure Audit

**Completed / independently audited. Neither global-mean arm passes the direction gate. Stop width/step/global-context architecture tuning. The preregistered CPU structure audit has also completed on all192 cached fields.**

## Scope and exact comparison

Same sealed Train128/95 source parents and Dev64/16 exposed diagnosis parents; source-GT-derived cached privilege/oracle supervision. Feature128/state33/evidence8/union256/radius .13545580427763146 unchanged. h0=silu(input(x)); h=(h0+silu(depthwise_conv(h0)))+mean_THW(h0); output256(silu(mix(h))). Zero new parameters,107648 total; initialization exact to original H128. Seed20260928,batch4,AdamW .001/wd0/clip1,only query-global1-cos; fixed200 or2000 terminal state. No PTD load, pixels/labels/fresh31/388/native/experts/OPD/target. No best-step or grid.

## Complete terminal direction readout

| Arm | Split | Mean cosine | Median | >.1 | >.3 | Count |
|---|---|---:|---:|---:|---:|---:|
| Local-S200 | train | 0.0298659168 | 0.0076936024 | 15 | 0 | 128 |
| Local-S200 | dev | 0.0242224497 | 0.0024013038 | 5 | 0 | 64 |
| GMean-S200 | train | 0.0280504686 | 0.0070424838 | 12 | 0 | 128 |
| GMean-S200 | dev | 0.0242027242 | 0.0018272851 | 5 | 0 | 64 |
| Local-S2000 | train | 0.0412345896 | 0.0073095771 | 16 | 1 | 128 |
| Local-S2000 | dev | 0.0200930473 | 0.0018424985 | 2 | 0 | 64 |
| GMean-S2000 | train | 0.0371665219 | 0.0077183527 | 17 | 1 | 128 |
| GMean-S2000 | dev | 0.0215934015 | 0.0018763005 | 5 | 0 | 64 |

Both required medians (train>=.3 AND dev>=.1) fail at both budgets. Dev >.1 at2000 increases2→5, but only5/64 align above.1 and0/64 above.3; this does not establish a broad rightward alignment shift. Existing baselines reused without training.

| Arm | Final Train mean loss | Final Dev mean loss | Last training-batch loss before final update | Preclip gradient min / max / last | Clipped steps |
|---|---:|---:|---:|---|---:|
| GMean-S200 | 0.9719495314 | 0.9757972758 | 0.9705231041 | 0.015318344 / 0.419778019 / 0.119412668 | 0 |
| GMean-S2000 | 0.9628334781 | 0.9784065985 | 1.0005146414 | 0.006018679 / 6.488626957 / 0.076266997 | 53 |

Preclip gradient norms are worker records, not raw-gradient independent reconstructions. All8 parameter tensors changed; basis frozen; integer Adam IDs/live counters200/2000; long-run step200 parameters AND optimizer state exact to the short run. NumPy independently reconstructed384 terminal cosines, maximum error 6.02295990859e-15.3 CPU model controls passed separately from actual GPU fits.

## Oracle field structure: full192, no selection

Each complete FP64 field A[THW,256] uses uncentered channel SVD energy. Report equal-query mean within each split. Global projection is constant overTHW; temporal projection retains one vector perT and is constant overHW. The latter includes the former, so their energies must not be added. Ranks are per-query optimal channel subspaces, NOT one shared cross-query basis.

| Field projection / channel rank | Train128 energy % | Dev64 energy % |
|---|---:|---:|
| Global constant vector | 0.029656 | 0.032596 |
| Per-time spatial mean | 0.655224 | 0.636620 |
| Rank1 | 49.377111 | 51.824762 |
| Rank4 | 76.700170 | 78.766523 |
| Rank8 | 84.606552 | 86.018153 |
| Rank16 | 90.437770 | 91.370581 |
| Rank32 | 94.636446 | 95.182519 |

| Projection / grid adjacency | Train128 mean cosine | Dev64 mean cosine |
|---|---:|---:|
| Global mean projection | 0.015993600 | 0.015545651 |
| Per-time spatial-mean projection | 0.080182605 | 0.078905159 |
| T neighbor | 0.111606342 | 0.108556020 |
| H neighbor | 0.097105003 | 0.100712576 |
| W neighbor | 0.110581688 | 0.114241243 |

All1,635,406 adjacent pairs have nonzero support; no pair was dropped as undefined. Adjacency is in the observed grid, not motion-aligned physical correspondence. Per-query spectra, full mean/median/range summaries, finite-difference energies and edge counts are retained locally; no top-k query selection.

## What this supports

The oracle targets are not well approximated by one global vector or one spatially constant vector perframe. They have concentrated channel energy while neighboring THW directions vary substantially. This motivates testing low-rank channel factors with THW-dependent coefficients as a future parameterization, rather than another wider/deeper local mixer or global/temporal-only output. It does not establish that the per-query factors are predictable from cached inputs, common across queries, or native-effective after truncation. Low mean energy of the target also does not prove that a global input descriptor can never help; the failed intervention is specifically this additive hidden mean at the locked budgets. No new parameterization was implemented or trained.

Raw NumPy versus independent Torch FP64 full-field projections/eigenspectra/adjacency maximum metric difference 1.30867539028e-14, relative spectrum difference 4.77566385694e-16. Synthetic controls compare direct full SVD and zero-support cases. Root sealed-spectrum/projection identities/counts/aggregates rechecked all192, max difference 1.11022302463e-15.

## Measured execution

Two serial fit workers plus nonoverlapping wrappers **56.404836s**; independent cosine CPU5.149292s; structure CPU11.214402s; root structure readback0.019725s. Historical GPU allocation/wrapper total72569.42608079306s cap=null. All workers exited normally, no GPU failure/replay. Research GPU idle; free disk129337237504bytes above8GiB. No data deletion or new recurring timer.
