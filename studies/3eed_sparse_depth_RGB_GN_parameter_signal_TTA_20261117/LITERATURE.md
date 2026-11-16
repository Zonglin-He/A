# Verified primary-source mechanisms and limits

- Tent official repository: https://github.com/DequanWang/tent . It adapts through normalization affine parameters using entropy in its own classification setting. This study uses existing RGB GroupNorm and measured sparse-depth loss instead; it is not a reproduction, and does not establish that normalization is new or generally effective in 3EED.
- Test-Time Adaptation for Depth Completion (Park, Gupta, Wong): https://arxiv.org/html/2402.03312v1 . Its main source-prepared proxy embedding maps sparse-depth features to image/depth features and regularizes target updates, alongside sparse-depth reconstruction. We do not add that mapping or claim its method/numbers: the pre-existing R76 I-only depth head is used under a narrower supervised-measurement experiment. No new source preparation is performed.

These papers motivate a bounded parameter/signal check, not a guarantee of improved grounding. The principal evidence for the scope change is the measured zero bridge gradient in R113 and nonzero existing depth-to-RGB-GN path in R116. Auxiliary recovery and final grounding are separately audited.
