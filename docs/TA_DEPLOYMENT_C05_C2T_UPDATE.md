# Research update: transient deployment corruption and temporal critic

Completed C0.5 and C2-T, no adaptation or production change. Main corruption setting
is now query/GT-independent random bursts on the full physical video timeline.
Original GT-centered v1 was superseded after3 unscored smoke cells, excluded.

C0.5 uses32 historically exposed VidSTG parents, VidSTG-source frozen TA-STVG,
5 families x3 fixed durations (1/5/10%): drop black placeholders, frame freeze,
motion blur, temporary occlusion, exposure. All480 cells completed. Primary
within-parent macro delta-tIoU -0.115994 pp CI[-1.998042,2.021439], delta-vIoU
-0.460574 pp CI[-1.371009,0.460466]; benchmark-development gate did not pass.
60 no-observation-hit cells retain exact clean predictions. Actual local harms
are preserved, with no severity changes or result-selected burst locations.
Post-hoc >10%-event-overlap group has delta-tIoU -3.9718 pp and delta-vIoU
-2.1568 pp; group composition differs, so this is association, not a causal
estimate or replacement primary population.

C2-T uses the original sealed first16 x clean/old noise-medium/defocus-medium/
JPEG-medium candidates, not newly generated transient candidates. Fixed
UniversalVTG proposal-confidence times interval-overlap ranks only student
candidates, no teacher-generated final interval. Corrupted parent macro:
Native/Expert/Oracle/Uniform tIoU=35.5731/39.4350/50.0039/34.7470%; expert gain
+3.86190 pp CI[.93620,7.16057], vIoU+2.00381 pp CI[.07766,4.33135]. High-margin
pair accuracy80.752% CI[62.196,96.528],14 contributing parents; all-pair74.570%
CI[61.832,86.123],16 parents. Fixed next-OPD resource gate passes. Clean t gain
+7.56271 pp; corrupted-minus-clean excess -3.70082 pp CI[-11.11177,1.31282],
so no established corruption-specific advantage. Four corrupted cells lose
>5pp tIoU; two lose>5pp vIoU. One of3 actual temporal corruption-loss cells
recovers clean-level accuracy. Q05/noise recovers16.04->0->20.71%; Q12/defocus
87.30->58.29->53.14%, candidate oracle58.29%. Support and critic failures remain
separate. The current bridge can proceed to a later minimal rerank vs hard
pseudo-target vs OPD comparison; none is implemented here.

C0.5 and C2 are development diagnostics, not fresh confirmation. Public anonymous
rows reconstruct means/CIs, candidate selections and pairwise statistics.
Bootstrap order is original roster ordinal in this registration; earlier C1
used query-key order, so finite-seed oracle CIs differ slightly despite exact
per-cell equality. No old C1 measurement was changed.

See [complete report](../results/tastvg_deployment_c05_c2t/2026-09-29/REPORT.md)
and [registered corrected protocol](../protocols/tastvg_deployment_c05_c2t_v2.md).
