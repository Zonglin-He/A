# One ordered Spatial WHO prompt: matched two-by-two results

One prelocked Referent -> Role -> Visible Extent -> Abstain decision protocol, sixteen exposed development parents, eight per domain, original Full_E1/A controls reused exactly. No Temporal prompt, compression indices, candidates, clock, optimizer, acceptance or hyperparameter change. No further wording search.

| Actual combined Student | HC vIoU % | VidSTG vIoU % |
|---|---:|---:|
| Frozen | 27.721111 | 4.565213 |
| Temporal | 30.566418 | 5.426607 |
| Full_E1 | 30.623179 | 5.631445 |
| FULL_ORDER | 31.047806 | 5.517491 |
| A | 30.690368 | 5.794307 |
| A_ORDER | 30.807190 | 5.770998 |

| Domain | Fixed visual evidence | New minus old prompt (pp) | Positive / negative / unchanged |
|---|---|---:|---|
| hc2 | Full_E1 | +0.424626 | 3 / 2 / 3 |
| hc2 | A | +0.116822 | 2 / 0 / 6 |
| vidstg | Full_E1 | -0.113954 | 0 / 2 / 6 |
| vidstg | A | -0.023308 | 0 / 1 / 7 |

| Domain | Arm | Known identity windows | Same entity | Complete target | Role supported / contradicted / unknown (8 windows) |
|---|---|---:|---:|---:|---|
| hc2 | Full_E1 | 7 | 6 | 5 | 5 / 0 / 3 |
| hc2 | FULL_ORDER | 7 | 4 | 3 | 4 / 0 / 4 |
| hc2 | A | 7 | 5 | 2 | 5 / 1 / 2 |
| hc2 | A_ORDER | 7 | 5 | 2 | 5 / 1 / 2 |
| vidstg | Full_E1 | 4 | 2 | 2 | 0 / 2 / 6 |
| vidstg | FULL_ORDER | 4 | 3 | 3 | 1 / 1 / 6 |
| vidstg | A | 4 | 3 | 2 | 1 / 1 / 6 |
| vidstg | A_ORDER | 4 | 2 | 2 | 0 / 1 / 7 |

These semantic labels come from the SAME original fixed sixteen panels, one exposed annotator. Unknown/mixed labels remain unknown; same entity includes partial target. No relabeling after seeing prompt outcomes. Prompt-compatible selected labels do not reveal hidden model reasoning, and more complete targets in a few inspected windows do not prove that every adopted reference is geometrically sound.

The interaction (A_ORDER-A)-(FULL_ORDER-Full_E1) is descriptive and exported for all five metrics. It separates a prompt effect under each visual input from the compression main effect. Full retains original padding/context and1280 tokens; A retains exact original160 native tokens and valid-area/ID safeguards. This is not a pure padding or runtime-speed contrast.

All16 prediction streams and saved Student arithmetic/final-readout roots were sealed before each group GT scoring. The old Full_E1 and A baselines are reused only with matching Source, candidate, physical clock, original prompt and configuration. All160 unique arm-parent rows, metric nulls, every paired sign, sparse geometry, accepted/nonzero and zero/reference-absent cases are retained. No Bootstrap, fullset, LN carry or CURRENT change.

This bounded protocol ends here. A cross-domain method or unified final prompt is not selected from a per-dataset development best score. Conditional larger confirmation would require a separately locked version and cohort; this study does not launch it. Discordant semantics and Student gains are evidence for a remaining transfer bottleneck, not a solved mechanism. The public prompt template and actual configuration are included; private media/GT coordinates/query-caption values/raw responses/tensors/parameters remain excluded.
