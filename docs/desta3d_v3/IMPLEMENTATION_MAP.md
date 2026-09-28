> Current diagnostic: [free-merger native controls and objective correction](ORACLE_FAILURE_DIAGNOSIS.md). These optimize standalone token deltas with model weights frozen, not the L0–L3 adapter scopes. Source preparation below remains cancelled.

# DESTA-3D v3: implementation and evidence map

**Current route (2026-09-28):** the mainline candidate is [privileged branch-latent adaptation](LATENT_PRIVILEGED_OPD.md). The source six-arm oracle has completed; it has not established correct-evidence advantage. External downloads were paused, then explicitly resumed by the user to finish this pixel-view route as a baseline. This is not a source-fit or OPD restart. Read the current oracle report/decision before interpreting the historical plan below.

This route was authorized by user attachment6e648ae6 on2026-09-28. The full-source fit was cancelled by the user at140 query occurrences /35 committed Adam steps. Its checkpoint is retained for provenance only and is not eligible as a source-prepared model, teacher or OPD state. The active external-evidence route is documented in EXTERNAL_PRIVILEGED_OPD.md; completed v2 controls remain historical evidence. This page distinguishes implemented interfaces from later conditional research stages.

The following diagram describes the **cancelled historical source-preparation route**, not the active execution queue.

```mermaid
flowchart TD
  V[Official Vid train archives] --> I[Media and label contracts]
  H[Official HC1 train; raw-parent split] --> I
  I --> R[Sealed train and internal source-val roster]
  R --> A[Fresh dual128 adapter; PTD4B frozen]
  A --> E[Independent event branch: GT-conditioned MTP endpoint CE]
  A --> S[Independent spatial branch: GT-conditioned MTP coordinate CE]
  E --> L[Evidence BCE + small original PTD CE]
  S --> L
  L --> O[Full balanced epochs; integer-key Adam checkpoints]
  O --> P[Full native source-val and matched Frozen predictions]
  P --> Z[Seal all hashes and physical identities]
  Z --> Q[Independent source-val scorer; parent/domain report]
  Q --> N[Conditional native temporal/spatial positive controls]
  N --> U[Conditional source-only qualification of unlabeled adaptation]
  U --> T[Conditional teacher-view qualification and branch Self-OPD]
```

| Component | Entry | Actual status |
|---|---|---|
| Full metadata / HC original extraction | `scripts/prepare_desta3d_v3_source.py` | Completed;85,184 intended queries/4,500 HC clips |
| Full original media probe | `scripts/audit_desta3d_v3_media.py` |9,936 completed;38 HC failures preserved |
| Same-clip HC repair | `scripts/repair_desta3d_v3_hc_media.py` |36 exact-annotation same-clip repairs;byte differences disclosed |
| Sealed labels and input identities | `scripts/build_desta3d_v3_source_roster.py` |85,181 accepted;3 media-contract quarantines;no outcome selection |
| Exact physical grid and bounded archive extraction | `vg_tta/desta3d_v3_data.py` |CPU contracts and full roster audited;actual pixels verified per use |
| Explicit structured source losses | `vg_tta/desta3d_v3_source.py` |CPU and4 real GPU backwards passed;GT-conditioned prefix, not native-control equivalence |
| Full training / checkpointing | `scripts/desta3d_v3_source_fit.py`, `vg_tta/desta3d_v3_training.py` |Cancelled by user at140 query occurrences /35 committed steps; checkpoint retained for provenance only; not eligible as source-prepared model |
| Native source-val / independent readback | training runner, `scripts/score_desta3d_v3_source_fit.py` |Code and CPU scalar/tensor controls ready;no complete new validation epoch yet |
| Future native branch scopes | `vg_tta/desta3d_v3_native_scopes_v2.py` |CPU-tested nested L0–L3; L2 includes branch query pool and event temporal pointwise; L3 shared projection/conv with norm_stem frozen; real controls pending |
| Shared detached referent pooling | `detached_referent_pool` in source module |CPU utility tested, defaultOFF;not adopted by training |
| Privileged PTD qualification | `scripts/desta3d_v3_privileged_ptd_qualification.py` |Four-view runner and seal-first independent scorer implemented/CPU-tested; actual GPU qualification pending |
| Evidence-view Self-OPD | route protocol |Conditional hypothesis; optimizer not implemented or executed |

Full source is large: train72,454 Vid query/4,106 HC query, 4,892/169 raw parents. Internal source-val8,230/391 query,544/19 parents. A balanced epoch contains144,908 occurrences (HC repeats explicitly),36,227 updates; maximum5 epochs181,135 updates, warmup9,057. An hour allocation is a safety review, not an epoch. Source-val selection can stop after at least3 completed epochs and patience2. Never extrapolate first-window loss into task improvement.

Authoritative entries: `protocols/desta3d_v3_full_source_route_v1.md`, `protocols/desta3d_v3_source_fit_v1.md`, `artifacts/desta3d_v3/full_source_roster_v1/{SUMMARY,ROOT_ROSTER_READBACK,QUARANTINE}.json`, `artifacts/desta3d_v3/full_source_fit_v1/{CONFIG,LOCK,REGISTRATION,CPU_PREFLIGHT}.json`. The root checkpoint audit checks all committed window order, actual per-parameter counters, live optimizer binding, RNG and hashes. It reads no predictions or labels.

Target data are absent from this worker. Historical target development exposure is retained. HC2 validation overlaps HC1 source parents, so old HC2 panels are not independent tests of the new mixed-source model. CURRENT and old queues remain unchanged. The user separately requested a Luna max monitor every30min for the active download/qualification route; it cannot restart cancelled training. Read `docs/RESEARCH_HISTORY.md` for current measured progress and receipt totals.
