# S0.6 — Spatial Critic Qualification

Completed / independently audited. Cached Sa2VA rewards carry directional information about student-only parameter probes. The user-authorized condition for S1 was met; S1 was subsequently executed as a separate experiment.

| Group | All-pair accuracy % | Source95% CI % | Antithetic accuracy % | Source95% CI % | Critic top1 gain pp |
|---|---:|---|---:|---|---:|
| corruption | 72.07 | [59.925926, 83.407407] | 68.50 | [52.500000, 83.500000] | 0.574047 |
| clean | 73.15 | [60.555556, 84.629630] | 70.00 | [53.333333, 86.666667] | 0.569450 |

Same16 exposed VidSTG sources/one query each, same source TA checkpoint and96cells.87 cells contain nonempty Sa2VA masks. Corruption critic denominator72cells/15sources;2592 all pairs and288 antithetic pairs, GT ties0, expert ties36/4. Expert ties receive.5; decisive-only corruption accuracies71.3333%/67.6667%. All-empty cells remain in16-source task scores with native/no-op, excluded from critic accuracy. Candidates were previously generated without experts; no new model/GPU/experts. Reward is detached mean reference-box IoU over nonempty masks only. All rewards and reward-only margin cuts sealed before old GT metrics opened.

Primary accuracy aggregates pairs inside each cell, then conditions inside source, then sources.10000 source bootstrap seed20260929; pairs are not independent sample count. GT ties excluded from binary accuracy and separately counted; literal three-way sign accuracy also in SUMMARY. Prior source-development exposure remains.

| Antithetic reward margin | Corruption accuracy % | Sources | Clean accuracy % | Sources |
|---|---:|---:|---:|---:|
| low | 55.60 | 14 | 62.50 | 12 |
| mid | 82.38 | 14 | 73.61 | 12 |
| high | 72.61 | 11 | 81.82 | 11 |

Margin grouping uses separately locked reward-only tertiles for each group and pair family. Corruption is not monotonic (mid > high); bins have different source composition. No reliability gate was added. All-pair bins, each of4 antithetic directions, literal sign/tie coverage and source-level successes/failures are retained in SUMMARY/PAIRS.

Corruption top1 gain 0.574047pp CI[0.259161, 0.924149], uniform expected gain 0.002650pp, oracle 0.811895pp; GT-best top1 agreement 58.67%. These are critic diagnostics, not online parameter gains.

783 independent rectangle reward checks,3 CPU contracts,32716 independent public scalar/order/CI checks. This qualifies critic information in the exposed local support, not universal ranking accuracy or future online transfer.
