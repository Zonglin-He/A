# Cached32 correspondence failure decomposition

User asks 2026-10-18: explain real~shuffle and below interpolation; separate
resolution, matching computation and wrong anchors, no new training/models.
Original b7e84a17 qualification32 remains unchanged and gate failed. New scope
is retrospective CPU-only mechanism diagnosis of already exposed sealed32.
No model/visual encoder/DINO/Student calls, downloads, TTA, new monitor, production
change or new image decoding. Original masks/maps/samples/weights/geometry and
old model-assisted identity codes are immutable/hash-verified.

Freeze every original candidate, sigma/tau/Native time, query/parent denominator,
annotation availability and old semantic labels. GT only evaluates frozen
candidate quality/geometry; no GT feedback substitution, training, reference
admission, sample generation or online choice. Planned four slots excluded from
primary unobserved support. Keep far>=25% original physical span and unknown.

Prespecified controls computed from existing tensors only, not a parameter scan:
- original real / original one-step cyclic shuffle / geometric interpolation;
- half-cycle non-planned-frame feature/map shift, anchors fixed, no GT choice;
- uniform affinity (each anchor's fractional area constant over valid target
  patches), revealing box-area prior with all correspondence removed;
- static anchor rectangle masks (no transport, original image coordinates);
- anchor self-matching then copy self-map to target frames: isolates same-frame
  softmax blur and temporal variation;
- rasterize the SAME interpolated rectangle on original grid and use identical
  soft-IoU, isolating scoring rasterization from correspondence information;
- hard argmax affinity (zero-temperature limit, no fitted threshold) vs original
  soft affinity, isolating soft aggregation; and per-video valid-patch channel
  mean subtraction before same L2/.07 affinity, isolating common-component
  anisotropy. These are diagnostic withdrawals, not candidate selection;
- half-resolution adaptive-average-pool cached features (floor(H/2),floor(W/2)),
  same affinity/scoring. No artificial upsample is called new information.
  Exact nearest-neighbor replication of maps is an invariance audit only.

Resolution diagnostics: DINO/GT width-height in original cells, fractional mask
self-IoU, small-target bins, pure continuous-vs-raster interpolation rank.
No genuine finer-resolution feature cache exists: cannot claim high-resolution
features would fix the issue or quantify resolution's causal percentage.

Matching diagnostics: normalized attention entropy/effective patch count, top1
mass, same-frame diagonal mass/self-mask loss, real/map-shift similarity and
feature similarity, spatial concentration/foreground contrast, reward variance,
ESS and weight-range. Correctness separately uses rank Spearman/pairwise and
weighted true-IoU improvement. Weight-strength control gives each feedback the
SAME sorted interpolation weights by its reward ranks (ties get tied mean;
flat feedback uniform). No target-GT temperature tuning or top-k search.

Wrong-anchor diagnostics use existing actually-viewed model-assisted CI/CP/WI/
MX/UN labels (not independent human GT). Leave each admitted anchor out and
recompute real and geometric reward on identical frames; removing WI is strictly
an offline label-assisted diagnostic. Empty remaining support uniform; all
queries remain. Compare WI vs CI/CP within mixed queries and separately report
all-WI cases, extent/uncertainty. Do not call every non-WI anchor correct or every
geometrically bad anchor wrong person. No new semantic labels.

Cache-only controls/diagnostics are sealed for all32 before recomputing GT rank
summaries. Report original and common-strength ranks, unobserved/far, all signed
query/parent bootstrap CI, size/motion/query and old-reference labels. Correlations
and single-factor counterfactuals need not add to a causal percentage. No auto
continuation even if a control improves. CPU cap4 threads/1h, GPU0, host guard6GiB
and disk30GiB. Publish anonymous code/protocol/full results/limitations, verify
remote files/refs and archive check-snapshot-check.
