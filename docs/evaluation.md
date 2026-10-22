# Evaluation protocol

Inference manifests contain only id/dataset/key/metadata. Per-query receipts bind input,
configuration, runtime fingerprint and prediction bytes. A complete seal binds the
manifest and all predictions; the scorer verifies every prediction before opening
labels. The CPU scorer retains the original HC2/VidSTG endpoint and dense-frame rules.

The official evaluated lists contain HC2 1,902 queries/227 parents and VidSTG
10,303 queries/732 parents. These corrected lists replace the historical raw3482 queue.
Never recover that superseded queue. Official full evaluation is not a tuning set.

The primary overall comparison is Final versus unadapted Source. ERef is an internal
initialization reference. Strong/448 is a historical precision/cost reference, not a
selection gate. Final and Untuned are aliases after bounded tuning candidates failed
the fixed development recheck; do not compute or count them twice.

Report full vIoU and threshold success, tIoU/boundaries, observed and unobserved space,
all signed changes and serious regressions, and parent-cluster uncertainty. Isolate
time by holding the spatial trajectory fixed where needed. Development panels are
previously exposed; do not label them independent confirmation. Historical metric
corrections and all old signed results are retained in the research archive.

A cached numeric compatibility test does not measure deployment speed. Count decode,
transfers, native source, expert, geometry and temporal/DP readout in actual end-to-end
cost. Segment cache counters by worker identity and startticks; do not subtract across
process resets or report imported old capture cost as current execution time.

Both full sets are sealed and scored; private predictions remain local. The complete
[public report](../results/boxfixed_full_v1/REPORT.md) includes all signed/severe changes,
thresholds, temporal errors, parent-cluster CIs and segmented receipt costs. Publication
reads saved scores and verifies hashes/aggregation; it performs no new GT scoring or inference.
