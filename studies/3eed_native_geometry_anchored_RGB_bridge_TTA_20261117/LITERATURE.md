# Mechanism sources and migration boundary

This round reuses R112 primary-source reading records: [xMUDA author paper](https://arxiv.org/abs/1911.12676), [MM-TTA author paper](https://arxiv.org/abs/2204.12667), and [xMUDA official implementation](https://github.com/valeoai/xmuda). Those records covered author abstracts and relevant official README material, not a newly asserted full-paper reproduction. The exact URLs in R112 LITERATURE.md remain authoritative for its prior retrieval.

The transferable question is how to prevent a cross-modal auxiliary objective from damaging an already useful native task path. Here the intervention is a hard computational separation: native point/text hidden states remain fixed for each observation, while existing geometry-to-RGB transfer and the late RGB readout may change. This is a local migration hypothesis, not xMUDA/MM-TTA reproduction; there are no two semantic-segmentation prediction heads or external teacher.

The tested signal is exactly the existing source-prepared physical recovery objective. No paper is cited as guaranteeing that RGB reconstruction or point-feature reconstruction identifies a referring instance. Same-information zero-update and noLate controls decide whether the proposed parameter update contributes to grounding.

Source URLs were independently refreshed against the author arXiv entries and official valeoai/xmuda repository during R113 development (2026-11-17 CST); full paper text was not newly read in this round.
