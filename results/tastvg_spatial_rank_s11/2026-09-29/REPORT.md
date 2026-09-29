# S1.1: Raw-RKL / Rank-RKL / Rank-RKL + Normalized Step

Completed fixed development experiment. Raw S1 sealed results reused; two new arms each execute six independent16-arrival streams,18 effective updates. No grid or posthoc configuration selection.

## Primary future nonexpert results

Paired delta vs Frozen (and budgeted rerank, identical to Frozen at nonexpert arrivals), in percentage points. Source means over five corruption types, then bootstrap10000 seed20260929 over12sources;60corruption cells are not60independent sources. Fixed stream/order conditional intervals.

| Arm | Corruption ΔsIoU [95% CI] | Corruption ΔvIoU [95% CI] | Clean ΔsIoU [95% CI] | Clean ΔvIoU [95% CI] |
|---|---:|---:|---:|---:|
| Raw-RKL | +0.000228 [+0.000091, +0.000389] | +0.000089 [-0.000026, +0.000214] | +0.000057 [-0.000121, +0.000229] | -0.000004 [-0.000138, +0.000107] |
| Rank-RKL | +0.078608 [+0.022241, +0.143997] | +0.047181 [+0.007487, +0.095654] | +0.084098 [+0.026113, +0.152190] | +0.051306 [+0.010724, +0.101671] |
| Rank-RKL+Norm | +0.057182 [+0.018959, +0.100168] | +0.031166 [+0.005766, +0.061403] | +0.069146 [+0.029699, +0.112531] | +0.036195 [+0.009091, +0.067821] |

| Paired change | Corruption ΔsIoU [95% CI] | Corruption ΔvIoU [95% CI] |
|---|---:|---:|
| rank_minus_raw | +0.078380 [+0.022128, +0.143622] | +0.047092 [+0.007470, +0.095445] |
| norm_minus_raw | +0.056953 [+0.018821, +0.099820] | +0.031077 [+0.005744, +0.061165] |
| norm_minus_rank | -0.021426 [-0.047843, +0.001615] | -0.016015 [-0.034268, -0.001987] |

## Actuation diagnostics

Medians over18effective updates per arm. Fixed source probe radius1.5765801926049214; Norm target step.015765801926049215.

| Arm | q range | p range | actual step norm | step/probe | KL decreases |
|---|---:|---:|---:|---:|---:|
| raw | 0.002532225 | 0.020441491 | 6.68308203e-05 | 4.23897374e-05 | 18/18 |
| rank | 0.631986493 | 0.020357206 | 0.0165622002 | 0.0105051429 | 18/18 |
| norm | 0.631986493 | 0.020356815 | 0.0157657973 | 0.0099999971 | 18/18 |

Full KL before/after, gradient ranges, all six final state norms and step ranges are in DIAGNOSTICS.json; every update in per-arm ROWS.json. Rank uses same SGD.005; Norm one fixed global-norm update. Empty expert at arrival0 in every stream means6noops/arm. Nonempty writes at4/8/12. Neither arm uses a GT gate, clipping, rollback, replay, momentum or extra steps. All equal rewards produce uniform teacher, which does not guarantee zero compatibility-RKL gradient if student p is nonuniform.

## Nonexpert arrivals after the first effective update

Additional diagnostic9source subset,45corruption cells (arrival>4); the main12source endpoint above retains the initial3nonexpert no-update controls.

| Arm | Corruption ΔsIoU [95% CI] | Corruption ΔvIoU [95% CI] |
|---|---:|---:|
| raw | +0.000304 [+0.000146, +0.000492] | +0.000118 [-0.000031, +0.000279] |
| rank | +0.104810 [+0.033873, +0.184160] | +0.062908 [+0.013500, +0.121033] |
| norm | +0.076242 [+0.029420, +0.125954] | +0.041555 [+0.008614, +0.078361] |

## Positive and adverse sources

Source-mean nonexpert corruption Δs/Δv in pp. All sources, including early exact no-op, are retained in SOURCE_EFFECTS.json.

| Arm | source | ΔsIoU | ΔvIoU |
|---|---|---:|---:|
| rank | Q07 | -0.003876 | +0.000047 |
| rank | Q02 | +0.000000 | +0.000000 |
| rank | Q12 | +0.289932 | +0.250440 |
| rank | Q10 | +0.303607 | +0.167066 |
| norm | Q07 | -0.012960 | +0.000050 |
| norm | Q02 | +0.000000 | +0.000000 |
| norm | Q12 | +0.173257 | +0.152033 |
| norm | Q10 | +0.199977 | +0.107945 |

## Protocol and verification

Same TA-STVG Vid-source checkpoint5ab12c86, original16previously exposed VidSTG sources/onequery each, existing clean/5%random-burst frame drop/freeze/motion blur/occlusion/exposure pixels (GT/query-independent). Same Sa2VA-4B revision3fee777d five uniform-frame masks. H and expert caches reused, new expert calls0.1792parameters: second-pass spatial query256 plus final spatial block5 norm1/3/4 weight+bias1536. Four fixed orthogonal antithetic directions plus center, same absolute radius. Current-policy nine candidates regenerated per scheduled expert per arm; same downstream native routing and geometry as S1. Candidate targets/teacher rewards detached. Rank descending0-based average ties1e-12. q=softmax(-rank), student p=softmax(-native L1/GIoU distance), reverse KL(p||q).

Source reset per arm/condition, state inherited across arrivals, report pre-update prediction. Entire192new predictions sealed before old16GT label scoring. Legal GT frame sIoU and correctedvIoU independently dual-scored384times.192state transitions/12resets,36gradient-step/KL reconstructions,64512parameter-coordinate update checks,8learned-state full native reinsertions.48current centers match central predictions; nonexpert frames do not consume expert/candidate data. Exact numerical details in per-arm AUDIT/CURRENT_POLICY_AUDIT files. FP32 KL reconstructed in float64 tolerance1e-6; intended parameter-step reconstruction tolerance1.5e-7.3CPU contract tests plus public scalar reconstruction.

New GPU process total 96.104968s, including loading and validation;432current-policy candidate tubes,36backward steps,0new encoder captures,0new expert calls. Raw reused, no new raw GPU. Public scalar checks: 13086.

## Interpretation boundary

Ranking sharpens teacher q and fixed normalization enforces the requested actuation scale; task utility is assessed separately. Finite-support box compatibility is an output-space surrogate, not a native parameter-policy likelihood. Different inherited parameters produce different later on-policy supports, so this is a comparison of fixed online recipes, not all later gradients on common fixed data. This repeatedly exposed16source development cohort, single stream order and few writes do not establish population equivalence, unseen generalization or universal failure of specialist/OPD methods. No production registration change, temporal OPD rescue or joint evaluation. Parameter-space OPD is only the user-specified possible next mechanism, not executed here.
