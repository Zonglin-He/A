# STVG local motion-region candidate readback

## Result and decision

The fixed local-motion proposal recipe constructed a separate pool of 37 new window-local motion regions plus 32 unchanged Source tracks; it was compared against the existing 152-candidate pool. The old non-Source candidates were not appended to the new pool. Across 32 windows, the sealed comparison contains 221 candidate rows: 152 from the old pool and 69 from the new pool, including 32 unchanged Source candidates. The new pool supplies 170 valid frame boxes across its 37 regions; the 32 Source tracks supply 154 valid frame boxes. These are proposal-support counts, not object coverage or identity accuracy.

On the 19-position primary support, the best old-pool candidate exceeded the Source center-box IoU by **4.099918 percentage points on average**. The new pool’s corresponding oracle advantage was **1.401724 pp**, a **−2.698195 pp** difference versus the old pool. On all 23 centers with known center annotations, the old-pool advantage was **5.711018 pp** and the new-pool advantage was **1.157946 pp**, a **−4.553072 pp** difference. The nine centers without a usable center annotation remain explicitly unknown.

| Support and domain | n | Old pool minus Source (pp) | New pool minus Source (pp) | New minus old pool (pp) | New better / worse / equal |
|---|---:|---:|---:|---:|---:|
| Primary 19, all | 19 | 4.099918 | 1.401724 | -2.698195 | 2 / 7 / 10 |
| Primary 19, HC | 10 | 0.626196 | 2.663276 | 2.037079 | 2 / 2 / 6 |
| Primary 19, VidSTG | 9 | 7.959610 | 0.000000 | -7.959610 | 0 / 5 / 4 |
| All known 23, all | 23 | 5.711018 | 1.157946 | -4.553072 | 2 / 9 / 12 |
| All known 23, HC | 14 | 4.265495 | 1.902340 | -2.363155 | 2 / 4 / 8 |
| All known 23, VidSTG | 9 | 7.959610 | 0.000000 | -7.959610 | 0 / 5 / 4 |

The two windows where the new pool’s best candidate scored higher were HC02 `window_slot=2` and HC04 `window_slot=1`; slots are zero-based. These are post-seal, GT-assisted diagnostics and do not select windows for a method. The aggregate comparison favors the existing pool, particularly on VidSTG; the fixed recipe therefore stops before any R1 call, OPD fit, or expansion. The decision does not establish that all local-motion proposals are ineffective; it applies to this recipe on these eight previously exposed sources.

The reported deltas are center-frame candidate-pool oracle diagnostics: for each available center, the maximum candidate IoU is compared with the unchanged Source candidate. GT was read only after the candidate outputs were sealed, for scoring and support availability. It was not used to generate or rank candidates, choose a threshold, or select a window. A pool oracle chooses its candidate using the annotation and is not an online method result. These center-box numbers do not measure full-tube vIoU or prove that a proposal depicts the referred object.

## Fixed recipe and execution record

The generator processed the already sealed eight-source grayscale/RGB cache on CPU. It used 32 fixed five-frame windows and no full-video decode. The generator barrier records `RGB_read=true`, `GT_read=false`, `GPU_used=false`, `model_loaded=false`, `teacher_called=false`, and `scoring_started=false`. The separate post-seal scorer records `GT_read=true`, `GT_used_for_generation=false`, zero GPU use, zero teacher calls, and zero OPD fits.

Each adjacent frame pair uses full-frame Shi–Tomasi corners (maximum 500, quality 0.01, minimum spacing 5 px, 3×3 block), pyramidal LK (21×21 window, level 3, 30 iterations, epsilon 0.01), forward/backward error at most 1.5 px, and a full-frame partial-affine RANSAC fit (2 px reprojection threshold, 2,000 iterations, confidence 0.99, 10 refinement iterations, at least four inliers and 0.5 inlier fraction). A failed camera fit has no fallback.

Dense Farnebäck flow uses `pyr_scale=0.5`, 3 levels, `winsize=15`, 3 iterations, `poly_n=5`, `poly_sigma=1.2`, and no flags. The residual threshold and forward/backward cycle-error limit are both 1.5 px. The fixed residual mask receives a 3×3 open then close; 8-connected components are retained from 64 px through 25% of frame area. Component tracks require a center component and at least one accepted match on each side of center. Matching is reciprocal nearest positive warped-mask IoU; missing transitions are not bridged. At most four regions are retained per window, ranked by supported-frame count, accepted matches, median cycle error, then row-major center location. IDs 2–5 are local to one window; they make no cross-window identity claim. The Source candidate is copied unchanged as ID 1.

The six-group figures above retain both the 19-position primary support and all 23 known centers, including the negative, neutral, and unknown outcomes. The anonymized public readback is designed to retain all 32 window rows and all 221 candidate rows without publishing box coordinates, masks, RGB, GT geometry, raw case records, or runtime contents.

The completed independent readback `ROOT_INDEPENDENT_READBACK002.json` passed 302 checks with zero failures, covering all 221 candidate rows, 32 windows, and six group aggregates. Its truth lookup uses the same official GT loader as the scorer, so it is an independent code/readback audit rather than an independent annotation-source replication. It reconstructs per-row center IoU from sealed candidate boxes and loaded truth, then recomputes pool maxima and group summaries from those IoUs and cross-checks them against the serialized geometry report. It does not independently rerun dense optical flow or regenerate warped masks. The first readback remains preserved as a failed historical receipt and is superseded by the separately hashed 002 pass; the two receipts are not conflated.

## Source and audit bindings

The locked generator configuration hash is `5b272f9aeb6e8b5a45aa3d50d1d500d5d26cb7596fc4ba5549e651ff2eef9551`. The runtime, generator barrier, and post-seal scalar readback are bound by SHA-256 as follows:

| File | SHA-256 |
|---|---|
| `scripts/run_stvg_local_motion_regions_v1.py` | `07cae3327d2d34b236d3c2309c970e2d69da4004342f544e95d61ce3d5c6d8d5` |
| `scripts/test_stvg_local_motion_regions_cpu_v1.py` | `8a456c1d0a76e347f8a48c305706a68ffabd55a56ee948fcc2bdc9b6d97b7517` |
| `scripts/score_stvg_local_motion_regions_v1.py` | `1deed46815c3f850f4cf4c67ce55fb28315307a272dd242f38478315ef585e16` |
| `scripts/render_stvg_local_motion_regions_v1.py` | `1b1ed5746104da890e883d2799f87562c42bd9a65862031a30c423f6373dde98` |
| `scripts/review_stvg_local_motion_regions_v1.py` | `075126d3265ba7d4d9b677d3e62ed90307d5bf5843f9bb3668fac22e040cf84b` |
| `scripts/run_stvg_short_event_tracklets_v1.py` | `27682e8bb7ca8014a3b8e49861af74aefc42c7a505215d4dfada01db0284c401` |
| `scripts/prepare_stvg_short_event_tracklets_v1.py` | `2b113d7997b8e64d00223baed05e732f6615a03c474932e69098e56a335e418a` |
| `scripts/stvg_adjacent_affine_tracklet_v1.py` | `00af1dc7921e539a27993fef1cd0374e9f8e9e4077c81df6007143fbb6910313` |
| `protocols/stvg_local_motion_region_candidate_v1.proposal.md` | `d5fe493b4f50ae3be5db9acc6e780df123506568490549b29a51c24c0bc5534c` |
| `protocols/stvg_short_event_tracklets_execution_v1.md` | `d8765aa43b0a91eb50945301ac1f7de8bfa230ca0f810d0bbd30ca46c4f09451` |
| `artifacts/stvg_local_motion_regions_v1/RUNTIME.json` | `945ee3b66ee353df93300475479527e4f00373d70a16c5044fd25849a2eca9ac` |
| `artifacts/stvg_local_motion_regions_v1/CPU_GENERATOR_BARRIER.json` | `b9699427b1848819da26a4d5f4aa2750e019fe53e88af2b0c4626d5750ca180f` |
| `artifacts/stvg_local_motion_regions_v1/ROOT_POSTSEAL_GEOMETRY.json` | `70f39e9b75e474cbcadea6b06cfc7dc342065500d9ccc0deb234aa54c3761d80` |
| `artifacts/stvg_local_motion_regions_v1/ROOT_INDEPENDENT_READBACK.json` (historical failed readback) | `a03642c8e259697cc928c857aff6c8b436d778a1e6c2918d4fa6cea23f6b0e52` |
| `artifacts/stvg_local_motion_regions_v1/ROOT_INDEPENDENT_READBACK002.json` (302-check pass) | `878ca9511255e343f6e7b8d2f624a9421c7e4de0f9698cc40a3f6477685f7d49` |

The included local-motion proposal note predates execution and labels the method “proposal only.” Its protocol description and source pin are preserved as historical provenance; the locked runtime, completed barrier, post-seal readback, and this report record the later authorized execution and final stop decision. The recovery ledger preserves three CPU-only engineering corrections: one rejected-component diagnostic assumed dense masks were retained, one post-seal reader initially joined on the wrong center key, and the first independent reader used an incorrect schema/support dispatch. The first independent receipt remains a failed 93-of-302 check result; the separately sealed 002 readback passes 302 checks with zero failures. The corrected reader reused the official GT loader and did not regenerate candidates. These corrections did not change the locked candidate recipe. The public projection records sanitized status, reason/scope, Boolean/count fields, receipt hashes, and source snapshot hashes; it excludes failure traces and runtime contents.

The root visual review reports eight private contact sheets and 160 frames reviewed, plus one public scalar chart. Private views are not part of the public export. The current method registry was not changed. This report records a finite development result, not a deployment promotion.
