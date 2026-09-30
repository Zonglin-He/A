# Paper48 P3_b0: Frozen vs Ours

64 hash-selected sources, one query/source, two orders; availability 0%. Historical project exposure disclosed. Official dense metrics and sampled continuity metrics kept separate.

| Group | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |
|---|---|---:|---:|---:|---:|
| clean | Frozen | 42.9005 ± 0.0000 | 19.6635 ± 0.0000 | 34.3750 ± 0.0000 | 14.0625 ± 0.0000 |
| clean | Ours | 42.9005 ± 0.0000 | 19.6635 ± 0.0000 | 34.3750 ± 0.0000 | 14.0625 ± 0.0000 |
| corruption | Frozen | 42.4434 ± 0.0000 | 19.3138 ± 0.0000 | 31.8750 ± 0.0000 | 13.4375 ± 0.0000 |
| corruption | Ours | 42.4434 ± 0.0000 | 19.3138 ± 0.0000 | 31.8750 ± 0.0000 | 13.4375 ± 0.0000 |

Future nonexpert, source bootstrap, both-order variation, clean/corruption excess, harm tails and quartile drift are in companion JSON. Orders/subsets reuse sources; not independent cohorts. No tuning or method promotion.
