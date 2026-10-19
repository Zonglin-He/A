# GT-assisted reference selection and propagation capacity32

2026-10-19, user-authorized finite causal diagnosis, NOT deployable TTA or new OPD.
Old E global four-vector correction is not a proximity-to-student rule: its
joint truncated residual objective plus relative DINO-score cost can prefer a
wrong but consistent assignment. Old multi-trajectory OPD and shared-state
negatives remain valid in their scope. New question: how much complete-tube
capacity exists if reference choice is selected by full vIoU, or references are
perfect, and can the existing shared-state interface use those perfect labels?

## Fixed common sources

All original NH16/NV16 exposed development parents, same32 with existing E,
original up-to3 admitted-frame DINO alternatives, finalMBR, Source and exact
cached spatial decoder inputs. CH/CV excluded from this common three-arm
comparison because no equivalent shared-state replay is used; no new Source
forward is allowed. FH/FV formal64 excluded and hash-protected. One query/parent,
all32/no-ref/no-time-overlap/no-known-anchor samples retained, unknowns explicit.
No new experts, DINO calls, backbone, head, source training, parameter search or
local-OPD change. All network weights frozen. Model/checkpoint as sparse_state32.

## Interventions and support controls, locked before GT input extraction

Baseline E+same finalMBR, plus inherited Fine and old DINO Direct for context.
1. Candidate_combo_oracle: enumerate original E admitted frames, up to3 retained
   alternatives each, plus existing outlier/no-reference choice (-1). For each
   assignment, compute mean residual in (cx,cy,logw,logh) FROM SOURCE and apply
   the same E global propagation to SOURCE (never apply E twice). Select by
   complete official fixed-time vIoU, not local anchor IoU. Include current E
   exact as tie-preferred feasible incumbent. No new boxes, score gates or
   frame support. This is oracle choice among E's existing enumerated LS means,
   NOT a new continuous-coordinate optimum. Report full selection landscape,
   candidate counts and best locally-chosen-anchor mean as a diagnostic control.
2. GT_E_known4: at original four planned positions with exact known GT, replace
   candidates by the true box singleton, score1 (no identity probability), then
   run SAME E estimator, inherited source scale/cutoff and beta1/16. Correct
   references can still be rejected by its consistency objective; record that.
   No GT interpolation to missing anchor labels. Also report GT_mean_known4
   and GT_subset_oracle_known4 (full-v-selected subset means, at most16)
   to avoid confusing estimator selection with shared4 representation capacity.
3. GT_state_known4: same true known planned boxes, existing final native spatial
   ca_qcontent_proj transient bias256, native RMS scale, Direct_anchor box loss
   5*mean L1sum4+2*mean(1-GIoU), fresh Adam HC .01/Vid .03, betas.9/.999,eps1e-8,
   exactly2 steps, final step2 (no GT step selection). E-retaining map unchanged;
   zero state exact Source/E, previous5 layers invariant, no temporal updates.
   This is achieved performance of one fixed finite optimizer, not an oracle
   over every possible shared state. It cannot prove all state capacity absent.

Matched support: original E-selected positions intersect exact GT-known slots.
GT_E_matched and GT_state_matched use those slots only. DINO_state_matched uses
same slots, original E-selected raw labels, same state/loss/2steps/optimizer.
Reuse old Direct exactly only when its support/targets/config all match;
otherwise actual fixed2 replay, no label-based changes. This isolates wrong
labels from additional labeled positions. GT_known4 may supply perfect evidence
where original DINO was missing or rejected; disclose this coverage intervention.
No known reference => keep E in GT interventions, no invented targets.

## GT custody and interpretation

Explicit GT oracle selection and adaptation are authorized OFFLINE diagnostic
interventions. Lock roster/config before GT extraction. GT coordinates remain
private in input32; all final oracle/state outputs sealed before summary scoring,
although full-v oracle candidate selection itself necessarily uses GT. Never
claim those outputs are GT-free or deployable. Same time/metric denominators and
all known unobserved actual nodes; never discard poor samples or choose per-domain
winning interventions as a deployment rule.

Screen complete vIoU mean delta E at roughly5pp in each domain as a research
capacity threshold, with paired intervals and influence analysis, not a promised
method gain or statistical null test. Small conditional success is not a large
all-parent effect. Even a large oracle gain does not prove DINO can rank it,
that OPD is needed, or that a fixed-feedback/direct method cannot use it.

## Engineering and resources

CPU enumeration/score/audit <=900s per phase. One frozen decoder-only bounded
GPU worker <=900seconds including loading/pilot, wall3600s, same gpu.lock,
freeGPU>=4GiB, original memory/disk guard, owned MemoryHigh16GiB/Max20GiB,
SwapMax1GiB. No signals to other services. Native shared bias gradient finite
check on fixed NH001/NV001; actual per-query model hashes unchanged, previous5
layers exact, sigma equals old replay. Preserve every failure/partial; no retry
with altered scientific parameters. Full network/backbone capture0, expert0.

Save all32 candidate landscape, chosen assignments, true-reference support,
GT consistency, shared-state actions/gradients/loss paths/final trajectories and
cost. Independent CPU replay of assignments, baseline/official scores, Adam,
fit loss and aggregation. Allowlist publish code/protocol/config/model provenance,
anonymous signed/unknown output metrics/audits/negative findings, never raw
GT/DINO/trajectory coordinates, captions, parentIDs, media, weights or features.
Archive same turn check/snapshot/check; verify GitHub bytes/hash/tree and both refs.
CURRENT/MAIN/PAPER and old general exploration/heartbeat remain paused/unchanged.
No automatic subsequent OPD design or parameter sweep.
