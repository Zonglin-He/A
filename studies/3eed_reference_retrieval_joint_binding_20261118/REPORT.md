# R167 · Reference retrieval and matched joint target–reference grounding

On 18,620 R165 directional records, broadening reference retrieval increased dual-end candidate coverage from 31 to 411 of 750 label-eligible edges. The shared bank includes distinct native queries and the non-native R166 initialized regions. Each main readout consumes the same geometry, frozen unary scores, and pairwise factors. The source checkpoints and PCA are fixed.

| Readout | All mIoU | All Acc@25 / Acc@50 | Camera-relation Acc@25 / Acc@50 | Both identities |
|---|---:|---:|---:|---:|
| Independent, with the exact R158+PCA subject | 14.496% | 27.340% / 10.161% | 29.796% / 11.842% | 6/750 |
| Reference first, followed by relation-aware subject selection | 13.325% | 24.997% / 8.349% | 23.334% / 6.537% | 21/750 |
| Joint | 13.785% | 26.009% / 8.864% | 26.159% / 8.037% | 47/750 |
| Joint with the reference restricted to the R88 top three | 14.014% | 26.415% / 9.365% | 27.533% / 9.746% | 23/750 |

Joint assignment improved over reference first in every direction at both thresholds, but decreased accuracy from the independent readout. The 750 labelled edges contain 722 conservative annotation proxies and 28 prior output-blind visual labels. The full relation table includes unlabelled edges and unresolved outcomes.

In 2,526 fixed-evidence predicate interventions, joint assignment switched the reference box 1,376 times. The two predicates were both consistent with measured geometry in 1,574 cases. Requiring the same known reference instance and a different known subject reduced that count to 38, versus 88 with reference first. A blinded review found one valid pair among 20 original cases. Five additional constructed scene pairs yielded ten source–target instances, with zero correct subject-and-reference assignments on both sides for every arm. The supplemental test freezes the neutral text encoding and tests the explicit predicate path.

This round supports an incremental improvement over committing to one reference within the same relational system. The stronger independent baseline and the identity-correctness results do not support adopting the current joint recipe. The existing literature already models contextual objects and spatial relations in visually grounded inference. [CORE-3DVG](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9b91ee0da3bcd61905fcd89e770168fc-Abstract-Conference.html) · [CoT3DRef](https://proceedings.iclr.cc/paper_files/paper/2024/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html)

[Full report and paired evidence](REPORT_ZH.md) · [Frozen protocol](experiment/PROTOCOL.json) · [Decision](DECISION.json)
