# Strong-Anchor Joint Support Refinement

The Chinese report gives paired outcomes for the four source-trained heads and the archived R158+PCA baseline. The ordinary Validation population contains 18,620 directional records. All paired scalar outcomes are included in results/anonymous_pair_scores, with arbitrary order within direction.

Reproduction inside the research workspace uses the pinned R173 source feature and roster seals, the R165 ordinary predictions, and the R97 frozen score cache. R174 first prepares the source anchors, trains all four heads per source, runs the source reference intervention and freezes the target method. It then seals target predictions before independent scoring and the target identity intervention. Private prerequisites are listed in the code and protocol. This export contains code, aggregate diagnostics and paired scalar metrics.
