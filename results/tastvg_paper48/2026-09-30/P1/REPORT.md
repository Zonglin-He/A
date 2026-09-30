# Paper48 P1: Frozen vs Ours

670 hash-selected sources, one query/source, two orders; availability 25%. Historical project exposure disclosed. Official dense metrics and sampled continuity metrics kept separate.

| Group | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |
|---|---|---:|---:|---:|---:|
| clean | Frozen | 44.9220 ± 0.0000 | 21.7122 ± 0.0000 | 31.9403 ± 0.0000 | 16.4179 ± 0.0000 |
| clean | Ours | 45.0020 ± 0.6777 | 22.1230 ± 0.3091 | 32.3881 ± 1.0554 | 16.9403 ± 0.1055 |
| corruption | Frozen | 43.8101 ± 0.0000 | 21.0217 ± 0.0000 | 30.5373 ± 0.0000 | 15.1045 ± 0.0000 |
| corruption | Ours | 43.9757 ± 0.6541 | 21.4929 ± 0.2817 | 31.4627 ± 1.1398 | 15.5821 ± 0.1266 |

Future nonexpert, source bootstrap, both-order variation, clean/corruption excess, harm tails and quartile drift are in companion JSON. Orders/subsets reuse sources; not independent cohorts. No tuning or method promotion.
