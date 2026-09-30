# Paper48 P5: Frozen vs Ours

128 hash-selected sources, one query/source, one order; availability 25%. Historical project exposure disclosed. Official HC-STVG dense evaluator with literal inclusive-GT-end input convention; 224px/64-frame project inference setting, not official420px benchmark.

| Group | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |
|---|---|---:|---:|---:|---:|
| clean | Frozen | 58.3682 | 29.9901 | 42.1875 | 13.2812 |
| clean | Ours | 58.3890 | 30.4115 | 42.9688 | 14.8438 |
| corruption | Frozen | 57.1348 | 29.1984 | 42.1875 | 12.1875 |
| corruption | Ours | 57.1732 | 29.6923 | 43.9062 | 14.2188 |

Future nonexpert, source bootstrap, one order (no order SD), clean/corruption excess, harm tails and quartile drift are in companion JSON. Orders/subsets reuse sources; not independent cohorts. No tuning or method promotion.
