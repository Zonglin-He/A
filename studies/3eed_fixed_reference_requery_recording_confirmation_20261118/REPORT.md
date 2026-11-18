# R170 — Recording-separated confirmation of the locked R169 reference requery

The locked D requery with the left/right term improved subject localization over the matched no-left/right control. Verified joint subject–reference identity stayed at zero. R158 with PCA remains the deployed readout. See the [full report in Chinese](REPORT_ZH.md).

We evaluated 48 distinct physical frames from 20 ordinary recordings outside the R169 D-selection scope. Each of three target platforms contributed 16 frames and was evaluated from both other source checkpoints. Earlier fixed-baseline evaluations had used these recordings. All 96 directional records were scored, including 23 fallbacks per requery variant. A single prediction-blind reader independently checked endpoint identities from the full image, expression and shuffled annotation inventory. Both endpoints were verifiable for 14 frames and 28 directional records.

| Readout | mIoU | Acc@25 | Acc@50 | Verified reference correct / 28 | Verified dual correct / 28 |
|---|---:|---:|---:|---:|---:|
| R158 with PCA | 11.654% | 20.833% | 10.417% | — | — |
| C without the left/right term | 11.654% | 20.833% | 10.417% | 2 | 0 |
| C joint | 12.783% | 21.875% | 12.500% | 3 | 0 |
| D without the left/right term | 11.654% | 20.833% | 10.417% | 7 | 0 |
| D joint | 14.124% | 25.000% | 14.583% | 8 | 0 |

D joint improved IoU on five directional records across five recordings with no losses relative to its matched no-left/right control. Relative to C joint, D improved five and worsened two. All five gains against the no-left/right control have unresolved reference truth. D kept seven correct mapped Top-1 references through one loss and one recovery. Its correct reference coverage went from 17 to 12 out of 28. All five coverage losses occurred where the fixed bank lacked any receiver attributable to the verified reference.

[Frozen protocol](experiment/PROTOCOL.json) · [Metrics](results/metrics.csv) · [Paired results](results/paired.csv) · [Mapping summary](results/mapping_summary.csv) · [Independent scalar audit](SCALAR_AUDIT.json)
