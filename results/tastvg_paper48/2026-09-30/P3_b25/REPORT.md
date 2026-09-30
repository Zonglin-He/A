# Paper48 P3_b25: Frozen vs Ours

64 hash-selected sources, one query/source, two orders; availability 25%. Historical project exposure disclosed. Official dense metrics and sampled continuity metrics kept separate.

| Group | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |
|---|---|---:|---:|---:|---:|
| clean | Frozen | 42.9005 ± 0.0000 | 19.6635 ± 0.0000 | 34.3750 ± 0.0000 | 14.0625 ± 0.0000 |
| clean | Ours | 42.5941 ± 0.3153 | 19.4168 ± 0.5778 | 33.5938 ± 1.1049 | 13.2812 ± 1.1049 |
| corruption | Frozen | 42.4434 ± 0.0000 | 19.3138 ± 0.0000 | 31.8750 ± 0.0000 | 13.4375 ± 0.0000 |
| corruption | Ours | 42.3394 ± 0.1108 | 19.2661 ± 0.2657 | 31.8750 ± 0.4419 | 13.1250 ± 0.4419 |

Future nonexpert, source bootstrap, both-order variation, clean/corruption excess, harm tails and quartile drift are in companion JSON. Orders/subsets reuse sources; not independent cohorts. No tuning or method promotion.
