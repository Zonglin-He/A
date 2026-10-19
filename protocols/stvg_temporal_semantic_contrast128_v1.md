# Full-versus-static textual temporal contrast, HC128, 2026-10-20

User temporal/joint autonomy, spatial frozen, no extra expert/model training.
Previous actual-window endpoint readout, decoder foreground and source motion
ROI pooling probes did not establish >=2pp. This changes the semantic question
asked of the existing Source, rather than ROI geometry, prior strength or local
OPD parameters. Hypothesis: the full-description temporal posterior contains
object-visible responses that can be identified by a matched static-description
Source pass. It is an ablation hypothesis, not an established factorization.

Same RC128 exposed development, original65 actual Source frames, source Vid
checkpoint, original target subject prefix, preprocessing and two offsets. Static
text is exactly cached parses.old.phrase from the already frozen conservative
visual-noun/attribute parser; no new parser/model/prompt alternatives, GT category
or manually written captions. 127 have nonempty phrase, one missing retains p0.
Exact same caption input reuses original q, no repeat Source. Otherwise one
frozen Source pass with caption replaced by that single static phrase. All new
Source boxes ignored. No DINO/ReferDINO/space adaptation. Same original full
description posterior p0 remains authoritative baseline and support.

Reconstruct static qS via identical peroffset endpoint legal probabilities and
min-start/max-end physical envelope on original65 timestamps. Its positive
structural support must equal p0; no epsilon mass, clipping or changed decisions.
Contrast qC proportional p0^2/qS, log space, fixed unit contrast coefficient.
Retain zero p0 mass. This heuristic is not calibrated event evidence and text
removal also changes language distribution/identity detail; do not claim a pure
causal event component. Candidate intervals and original finalMBR are unchanged.

Outputs Native p0, PriorSquare proportional p0^2 (no extra semantic information),
and Contrast. PriorSquare explicitly controls simple posterior sharpening.
Missing static phrase: Contrast=p0, keep parent in all metrics. No pervideo
switch, threshold or exponent sweep. Full/static identical posterior must return
p0 exactly within numerical tolerance, uniform qS gives PriorSquare. All positive
support/normalization and independent exact MBR utility checks before GT.

Primary Contrast-Native fullv using fixed existing Live41/42/43 mean within
parent; E auxiliary. Only Contrast eligible for later parent-disjoint frozen
confirmation, mean>=2pp and lower95CI>0, and Contrast-PriorSquare lower95CI>0.
All three signed results/tails reported, no selecting a seed or dropping failures.
RC128 are repeatedly exposed development and never final independent proof.

<=128 Source queries/256offset (expected127 absent exact duplicates), shared
gpu.lock and GPU4GiB/host6GiB/disk30GiB guards, lease900s/process900s, first2 noGT
cost qualification, finite4h. Count repeated actual encoding even same RGB.
Freeze source/config/subject/cache hashes, no pretrained updates. All128 Source
posteriors/readouts sealed before GT; independent CDF reconstruction, zero
baseline and frozen spatial identity plus official/DenseTube score checks.
Readout/caption contrast capability is not temporal OPD; positive result still
requires confirmation and matched Direct/Fixed/Live. Publish exact code/config,
all signed/unknown scores, negative tails, costs and limitations, excluding raw
captions/identities/media/GT/weights/caches. No registry promotion.
