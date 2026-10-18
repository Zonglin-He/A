# Query-only E64: matched actual decoder response and bounded control

Fixed original HC32/Vid32 development cohort, Source checkpoint, E, final-MBR,
QR256 interface and all original Gaussian/loss/domain parameters. No new expert,
network training, Temporal update, cohort expansion or production promotion.

The main local study uses the nominal Dense OPD pre-rounds 0/19/39 (HC) and
0/4/9 (Vid), all 64 queries including inactive/flat-reward no-ops. At each SAME
historical state and frozen rollout, evaluate all eight binary combinations:
original versus half QR norm; Adam historical first moment versus current gradient;
native intermediate-reference detach versus its full chain-rule gradient. All
directions use the SAME original second-moment diagonal metric and are normalized
to the original proposal norm before the amplitude factor. This isolates direction
and magnitude; it is not eight new optimizer trajectories or a temperature sweep.
Reconstruct the original Adam proposal and exact old next decoder output. The
full-chain intervention changes the backward graph in one isolated decoder
instance only; its forward values must be bit-identical and vendor source stays
unchanged. Central finite differences at fixed QR-unit displacements .001/.0005
compare both gradients with the actual forward function. Report numerical
instability rather than tune the FD scale to pass. H008 Sparse rounds2/5 are a
predeclared additional case check, excluded from the primary all64 summaries.

Separately run Sparse_control and Dense_control from Source/zero QR/fresh Adam,
original 40/10 scheduled rounds, same E and random key/round seeds. Each proposal
has at most four predetermined QR scales 1,.5,.25,.125. Accept the FIRST actual
native-decoder recomputation whose same frozen-rollout FP64 surrogate is no larger
than before plus fixed numerical tolerance 1e-7*(1+abs(before)). Record actual
output displacement, not a GT/output-quality gate. If none passes, restore QR AND
all Adam state and continue to the next scheduled round; no extra rounds. If a
scaled step passes, retain the original proposed Adam moments (a deliberate
algorithm definition). Rewards/candidates/weights remain frozen during retries;
every next scheduled round resamples around the actual accepted Student state.
Full-chain and momentum interventions are LOCAL DIAGNOSTICS, not silently inserted
into these native-gradient control arms. No GT-selected checkpoints or attempts.

All local and full-path outputs are sealed for all64 before a separate CPU scorer
opens target GT. Report full vIoU, tIoU invariance, exact known unobserved decoder
frame sIoU, paired parent CIs, severe negative tails, actual QR/output movement,
accepted scales and restored rejections. Main local effects average three slots
within query then query within domain, not treating steps as independent parents.
Local factor effects are not end-to-end method scores. Control arms compare with
the original same-arm results, Source/E, fixed-rollout and same-E Direct; never
claim surrogate descent equals task improvement. Include H008/H128/H119/V038/V016
and check preservation of positive changes versus merely suppressing updates.

Hard limits:128 new query-only fit slots,194 local slots,45 GPU minutes,2h overall;
shared GPU lease,>=4GiB GPU free before capture,>=6GiB host and>=30GiB disk,
16/20GiB memory high/max and1GiB swap in the execution unit. Recapture only the
64 Source prefixes; no historical Before stream. Both full two-pass output and
frozen-weight/source-logit identity must hold. Existing CURRENT/MAIN/PAPER and
all prior predictions/negative results/pause decisions remain unchanged. Publish
only anonymous scalar results/code/protocol with actual remote content checks;
private coordinates, text, raw states, media, annotations and weights excluded.
