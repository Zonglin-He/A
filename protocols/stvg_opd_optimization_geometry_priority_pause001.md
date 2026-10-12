# Explicit priority pause addendum — 2026-10-12

The human explicitly chose `当前 query 保存后暂停基线，先跑新对照`.
BASELINE_PAUSE_RECEIPT binds the original live controller/worker, process start
ticks, committed prefix and unchanged source/configuration bytes. SIGSTOP is
reversible: no process kill/restart or optimizer reconstruction. The paused
worker retains its CUDA allocation and original shared flock, so other ordinary
workers remain excluded. The authorized diagnostic acquires its own exclusive
priority reservation ONLY while both original process identities are still
stopped. It installs the same data-read guard and deterministic settings as the
original GPU entry point. This is a bounded external scheduling exception, not
a modification of the baseline, its lease file, runtime or science. The normal
shared lease cannot be obtained while the preserved worker is stopped. No
concurrent active experimental GPU worker is allowed.

If the preserved allocation prevents qualification, retain the actual failure
and ask for a different handoff; do not kill the paused process or lower the
scientific memory/shape/step budget. ToDesk/desktop contexts are not experiments.

Run real Full replays to collect parameter gradients at each common OPD state;
verify their scientific fields against the exact original complete Full fit.
These 1792 GPU replays are necessary mechanism measurements, not replacement
baseline predictions. Formal comparisons retain the hash-bound original Full
reference and add 3584 new Direct/control predictions. There are 5376 actual
full GPU fits in this mechanism run, plus 168 qualification fits (28 cells x
three arms x two real repeats). Qualification is separate from formal results.

Restore the original worker and then controller with SIGCONT only after the new
GPU worker actually exits and releases its priority reservation. Check prefix
bytes and process identity again, and record resumption. New CPU scoring/root/
view/publication can continue separately. Any preserved actual new numerical
failure requires bounded root handling; a user pause is not a failure.
