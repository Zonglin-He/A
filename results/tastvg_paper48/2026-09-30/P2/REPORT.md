# Paper48 P2: Frozen vs Ours

128 hash-selected sources, one query/source, two orders; availability 25%. Historical project exposure disclosed. Official dense metrics and sampled continuity metrics kept separate.

| Group | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |
|---|---|---:|---:|---:|---:|
| clean | Frozen | 43.1834 ± 0.0000 | 19.8097 ± 0.0000 | 31.2500 ± 0.0000 | 14.0625 ± 0.0000 |
| clean | Ours | 43.6929 ± 0.3442 | 20.2885 ± 0.0262 | 31.6406 ± 0.5524 | 15.2344 ± 0.5524 |
| corruption | Frozen | 42.3028 ± 0.0000 | 19.3181 ± 0.0000 | 29.5833 ± 0.0000 | 13.3854 ± 0.0000 |
| corruption | Ours | 42.8775 ± 0.1806 | 19.7887 ± 0.0340 | 30.1562 ± 0.7366 | 14.4271 ± 0.6629 |

Future nonexpert, source bootstrap, both-order variation, clean/corruption excess, harm tails and quartile drift are in companion JSON. Orders/subsets reuse sources; not independent cohorts. No tuning or method promotion.
