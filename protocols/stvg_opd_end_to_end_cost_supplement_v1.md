# Fixed cold-decode Full-method timing

This is a finite timing supplement to the human-authorized STVG supplement.
It does not select parameters, cases, reference gates, or a deployment method.

For each original cross-clean direction, take the first 16 original parent
queries ordered by SHA256 of `stvg-cost|dataset|2026101104|query_ordinal`.
Use its saved original Full-fit initial 1792 state, original source weights,
all original frames (up to 200), native event, Uniform4, frozen DINO top1,
and original HC/Vid configuration. No GT is opened. All 32 complete fits
must match the corresponding original scientific fit, with fresh expert
predictions independently checked against the original hash-bound input.

Weights remain resident. Each query has an empty application frame cache,
fresh video decoding, a native student forward including its encoder prefix,
and fresh expert detection. OS filesystem pages are not forcibly evicted;
this is cold application decoding, not an OS-page-cache benchmark. Saved
query parsing is used, so an external language-model parse is not included.
Weight loading and CUDA initialization are separately reported.

Measure wall time with CUDA synchronization: decoding/preprocessing, native
student/prefix, fresh expert, adaptation, and final spatial readout. The
actual last replay call inside the original fit is the final readout. Its
time is subtracted from total fit time to prevent counting it twice.
All replay calls are synchronized for measurement; this overhead and the
qualified engineering recorder are part of this audited implementation.
Hash checks, independent CPU arithmetic and serialization follow the timed
path and are separately reported. Total timed query time must equal the
five-stage sum plus recorded dispatcher overhead.

Report all 32 query timings, per-direction median and p95, CUDA peak allocated
and reserved memory, actual expert calls, admitted frames and backward count.
Keep all slow and no-admitted queries. These historical fixed queries do not
estimate deployment throughput on a fresh population. Run only after the
baseline selected-formal GPU barrier, under the same exclusive GPU lease.
Do not alter any old code, saved fit, receipt, scientific lock or runtime.
