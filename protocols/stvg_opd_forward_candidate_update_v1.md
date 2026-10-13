# Fixed real-forward candidate feasibility screen

This is a new bounded screen after the failed observation-response projection
qualification. It does not repair or reopen that numerical result.

Use its same eight historical P9/P0 previous-P6 queries (HC indices 0,1,2,3;
Vid indices 32,33,34,36), Inherited-order1, original pixels/frame grid/source
checkpoint/Native WHEN/Omega0/admission/Top1/configuration/noise. At each saved
original OPD boundary k=0,floor(K/2),K-1, reuse the exact sealed Adam result.
Two fresh frozen captures per query give 48 boundary comparisons. Four actual
decoder candidate readouts per comparison give 192 readouts, NOT 192 fits.
No new backward, optimizer step, training, expert call or GT access is needed.
The original full proposal copies its saved after-state directly: never
subtract/add round-trip it. Other candidates mask that same displacement to LN
or Query, or take one half in the original float32 parameter representation.

Candidates, in fixed tie order: A, LN, Query, Half. Query has 256 coordinates;
the original six LN vectors have 1536. Before is the current saved state.
The drift reference is query-entering Inherited-order1, NOT source LN or a
rolling reference. Omega contains admitted original Top1 expert positions.
U is the complete real decoder grid excluding ALL physical Omega0 visited
positions, including visited positions without admitted evidence. No GT event
or annotation restricts U. Metrics use float64 arithmetic on original float32
outputs; aligned center/width/height box IoU has the original 1e-7 union floor.

G(d)=mean_Omega[IoU(B_after(d),expert)-IoU(B_current,expert)].
D(d)=sum_U ||B_after(d)-B_entering||^2 / |U|, without the old F objective's 1/2.
Fix rho=0.9 before any new measurements. If G(A)>0 and U is nonempty, choose
the minimum D among candidates with G>=rho*G(A), exact fixed tie order.
Otherwise choose A, and explicitly retain nonpositive/empty/undefined paths.
No zero/no-update candidate is added. Empty evidence and original uninformative
updates retain exact original no-op. Adam moments/noise remain opaque reused
scientific witnesses; hypothetical candidates do not write any online state.

The prelocked useful-space criterion is at least half of positive-G(A)
boundaries IN EACH domain having a non-A feasible candidate with
D(A)-D(candidate)>1e-10+64*eps_float32*D(A). Evaluate fractions on repeat0;
repeat1 tests exact all-state/all-box/all-metric/selection reproduction.
All selected positive contexts must preserve at least rho of actual gain.
Report query coverage, the full candidate mix and negatives, not just winners.
The criterion is a resource decision for this fixed historical screen, not
statistical proof of generality. Missing positive or U denominators are unknown.

Report synchronized repeat1 complete four-candidate screen cost, including
parameter construction, four real decoder forwards, CPU output transfer and
decision arithmetic. Frozen capture/current-state validation, source hashes,
serialization and full independent audit are outside that timer. Also report
per-candidate measured costs. Compare descriptive medians with old F local
costs on exactly the same informative repeat1 contexts; acknowledge different
instrumentation. Never subtract these from deployment latency or extrapolate
an endpoint speedup. No performance GT or fixed64 fit may start automatically:
a task-validation protocol is a separate necessary next decision if this
screen establishes useful space and deterministic replay.

Independent CPU readback verifies source/pixel/state/frame/proposal/Adam/noise
custody, all original endpoint reproduction, masking/half arithmetic, every
candidate's actual box metrics, choices, fractions/cost statistics and repeats.
Save all anonymous positives, negatives, zeros, unknowns and no-ops, actual
plots with pixel view, source/public scalar verification, remote publication
and archive receipts before closing this scope. Preserve all earlier scopes,
P7-A, CURRENT and the same cancelled baseline T standby identities.
