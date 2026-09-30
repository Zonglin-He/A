# Paper48 P3_b100: Frozen vs Ours

64 hash-selected sources, one query/source, two orders; availability 100%. Historical project exposure disclosed. Official dense metrics and sampled continuity metrics kept separate.

| Group | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |
|---|---|---:|---:|---:|---:|
| clean | Frozen | 42.9005 ± 0.0000 | 19.6635 ± 0.0000 | 34.3750 ± 0.0000 | 14.0625 ± 0.0000 |
| clean | Ours | 42.3997 ± 0.0000 | 19.2374 ± 0.0261 | 29.6875 ± 0.0000 | 12.5000 ± 0.0000 |
| corruption | Frozen | 42.4434 ± 0.0000 | 19.3138 ± 0.0000 | 31.8750 ± 0.0000 | 13.4375 ± 0.0000 |
| corruption | Ours | 42.9904 ± 0.0000 | 19.5251 ± 0.0132 | 31.2500 ± 0.0000 | 12.5000 ± 0.0000 |

Future nonexpert, source bootstrap, both-order variation, clean/corruption excess, harm tails and quartile drift are in companion JSON. Orders/subsets reuse sources; not independent cohorts. No tuning or method promotion.
