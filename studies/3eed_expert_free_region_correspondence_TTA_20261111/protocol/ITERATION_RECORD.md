# R74 iteration record

Hypothesis: known same-region identity across point sampling supplies discriminative supervision for the source matching projection without external target labels. Counter-hypotheses: region/query mismatch; auxiliary-language misalignment; false negatives; update drift; benefit only from added views; generic perturbation rather than correct matching.

Fixed source/A256; actual native cached forward prefixes; correct/shuffled/noRGB contrastive arms; source keep; one objective-accepted step; C not in fitting. Pilot64 disclosed before fixed remaining2240 expansion. No tuning. All full-cohort outputs sealed and independently audited before postseal saved-label join.

Observation: actual accepted updates but +0.014107pp over probability averaging, below stronger frozen two-view representation fusion; largest positive nearly explains net gain, C grounding/retrieval worsen, noRGB task identical. Correct-vs-shuffled overall weak positive reverses on TRAIN.

Decision: do not adopt this recipe; preserve all gains/losses and controls. No new grid, extra optimizer steps, layers, source training, external feedback or automatic follow-on. Absence of a source-trained regional embedding and source auxiliary/task preparation is an implementation-condition difference from DARTH/Point-TTA, not silently attributed to an impossible task.
