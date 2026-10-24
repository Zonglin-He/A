# Paper-review supplemental experiments — completed on available data

The requested suite excluding Fig1 is complete on the two locally available fully annotated datasets. CG-STVG was run last at native420FP32 using cross-source official EMA checkpoints. The optional third complete STVG benchmark was not run: no local complete dataset/path was established. This exception is explicit, not counted as an executed experiment.

The production SimplePath+native prior+all-video Native2D method, parameters and completed full results are unchanged. All official target sets were historically exposed; these new screening/confirmation cohorts are prospectively fixed for the contrasts and parent-disjoint from each other, not historically unseen validation.

| Request | Actual evidence | Status |
|---|---|---|
| Fig1 | Existing work excluded by user | Excluded; no new Fig1 |
| Sparse expert interpolation and distance | [Offline full12205](offline/) | Completed from saved full evidence; native MBR time, edge E fallback |
| Entropy and consistency TTA | [Matched LN baselines384](consistency_v2/ANALYSIS.md) | Complete, real optimizer updates; not canonical Tent/CoTTA |
| Task-aware pseudo-box baseline | [PseudoBox384](pseudobox/) | Complete20Adam steps on four box residuals, no network-weight update; HC stronger than Full retained |
| Native prior/MBR/prototype controls | [Fourteen full-cache arms](cached/) | Full12205, fixed-space conditional tables explicitly labeled |
| Spatial support and two DP decodes | [Coupled supports384](supports/) | Complete repooling/time recomputation and2×2 control |
| Mask/affinity/anchor corruption/spatial sensitivity | [Sixteen controls384](spatial_controls/) | Complete; one-way affinity distinguished from past-only anchor |
| Budget/K/schedules/dense | [Screen](budget_v2/SCREEN_ANALYSIS.md), [fixed confirmation](budget_v2/CONFIRM_ANALYSIS.md) |1660/1664screen valid; four K32 exact-search resource failures retained;10eligiblearms×256confirm complete; dense cost-ineligible by predeclared rule |
| Actual cost/K/T curves | [Budget figures and data](budget_v2/) | Actual joint-context GPU costs; SVG+private PNG/PDF QA complete; no cache deployment-speed claim |
| Anchor quality/native coverage/provenance | [Offline evidence](offline/), [parameter provenance](provenance_clarification.json) | Complete offline diagnostics; partial exposure registry disclosed, no GT inference gate |
| Controlled RGB shifts | [Screen](shifts/SCREEN_ANALYSIS.md), [confirmation](shifts/CONFIRM_ANALYSIS.md) |384+768same-transformed-RGB Source/Full pairs, all scored/reported |
| Second backbone CG-STVG | [Screen](cgstvg_v9/SCREEN_ANALYSIS.md), [confirmation](cgstvg_v9/CONFIRM_ANALYSIS.md) |128screen+256confirm, five arms,1920metric checks; all four real native bridges passed |
| CG resource/readout/input-contract recovery | [All failures and exact recovery](CGSTVG_MEMORY_RECOVERY.md), [cost accounting](cgstvg_v9/COST_ACCOUNTING.json) | Original frozen versions preserved; no precision/resolution/frame/parameter relaxation |
| Third complete STVG benchmark | [Availability audit](THIRD_BENCHMARK_AVAILABILITY.json) | Optional, not run: complete local data unavailable |

[Completion manifest](COMPLETION_MANIFEST.json) binds the saved score receipts and360936official/common/dense comparisons across studies; this count is not a number of independent samples. Raw TSVs retain full precision. All signed outcomes, severe t/v/union-once, parent CIs, duration groups, failures, actual costs and exposure limitations remain public. Media, captions, GT coordinates, raw features/masks, checkpoints and credentials stay private.

Key findings are mixed. The historical local `Source` baseline already includes OUR MBR; official TA-STVG uses per-offset MAP-hull. The stored baseline must be displayed as Source+MBR, with original native Source separately reported ([correction](../source_semantics_v1/README.md)). MAP-to-MBR gains are decoder gains, not new visual evidence. Final DP benefits are supported, whereas first auxiliary DP and trajectory-specific temporal support do not show clear advantages in the fixed comparisons. PseudoBox beats Full on HC. K16 offers a positive HC comparison but does not replace main K8. Matched entropy/consistency results do not reject all TTA. CG confirmation has positive temporal/Full−Space gains but no established Full−Source joint-quality gain and substantial spatial losses. These outcomes do not trigger further tuning or selective omission.

For deployment semantics use docs/method.md; these supplemental studies are experimental evidence, not new production registration. Historical Source+.9 target_not_met remains preserved even though the user accepted the measured current main method.
