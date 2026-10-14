# Matched-support spatial reference diagnosis

Authorized 2026-10-14: retain Temporal Caption–Query + Adam and Spatial OPD;
isolate reference construction before one conditional new candidate mechanism.
The eight previously exposed short-window sources remain fixed. No new teacher,
tracker, prompt, learning rate, temperature, model, or source selection is used.

The support of each source consists of every old QA center where all displayed
candidate center boxes and the Source box are valid, an actual GT annotation
exists, and the sealed base R1 response is legal, not truncated and non-abstaining.
This support is selected by availability, not the magnitude or sign of IoU.
All excluded centers and reasons are retained. Support and reference identities
are frozen before new fitting. Cases with no support remain exact NoUpdate.

Four references use precisely the same support: Source native box; sealed R1
selected candidate (including Source selections); maximum-IoU displayed candidate
(Candidate-Pool Oracle, smallest candidate ID breaks an exact tie); actual sparse
GT box (GT Box Oracle). The last two are explicitly label-assisted diagnostics.
Source and R1 selections are not skipped by a non-Source gate in this matched
diagnosis. No reference is inserted into the deployable method or CURRENT.

Every nonempty case/role starts from the same source checkpoint and initial
1792 Query/LN state, with fresh Adam, five original configured rounds, unchanged
domain configuration and stochastic key `caption_query_fixed_space|case_id`.
All rounds/final output are retained; no GT-based step or amplitude selection.
Full model final redecoding is mandatory. F and the old T interval are exact
controls; S and T+S are evaluated for each reference. This is not a comparison
of the old one-anchor recipe to a new many-anchor recipe: all four new roles
share their new matched support. It does not directly adjudicate V01/V03's old
GT-unknown supervised anchors.

Candidate/reference IoU at shared centers is separate from full-Tube vIoU.
Report all case and domain paired changes, supervised and unsupervised frames,
reference-fitting progress and actual parameter displacement. Oracle outcomes
are not method performance. Labels are explicitly exposed for Oracle construction
before fitting; the worker receives only locked references, and final performance
scoring starts after every prediction is sealed. The diagnostic cohort cannot be
relabelled as independent confirmation.

Interpretation is conditional: GT succeeds but pool does not -> candidate quality
or sparse-reference geometry/transfer is limiting in this interface; pool succeeds
but R1 does not -> selection loses available benefit; GT also fails -> optimization,
reachability or sparse-to-full propagation remains unresolved. These cases may
coexist. Failure of pool adaptation alone does not prove poor candidate geometry.

Only if this comparison supports candidate insufficiency relative to the GT
control may one new, separately locked, local visual-evidence candidate mechanism
be tested. No parameter/prompt sweep. Its coverage and identity must be assessed
before R1+OPD. Independent >=32 unused sources per direction remains conditional
on a frozen GT-free rule; F/T/S/TS remain mandatory. A further unsuccessful fixed
R1-Tracklet mechanism ends that concrete recipe, not the Spatial branch itself.

Use one own guarded GPU unit, RAM High16GiB/Max20GiB/SwapMax1GiB; stop own unit at
RAM18GiB/anon14GiB/host-available6GiB/GPU-free4GiB/disk-free30GiB. Never resume old
queues. Preserve all original evidence and engineering recoveries. Publish only
anonymous numerical evidence/source/protocol; no raw media, text, GT boxes,
weights or parameter caches. Root independent math/metric/readback, actual visual
review, verified remote bytes/refs and archive check-snapshot-check close the task.
