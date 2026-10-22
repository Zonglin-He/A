# Cached448-only /224six /fusion decomposition

**Status: available-cache work completed; full64 branch comparison remains incomplete because63 full448 distributions were never saved. No new high-resolution model run was started.**

## Coverage audit

The64 complete224 six-layer posteriors and previous fusion decisions/metrics are present. Historical student caches and prediction archives contain the final Strong interval, not its distribution or448 hidden/logits. One engineering capture retains the full Strong posterior. Its Source/ERef/frame IDs/eight observations/224 hidden are bitwise equal to H001.

For that one case only, q448=2qStrong−q224six is nonnegative, sums to1 within floating precision and reconstructs the saved Strong distribution and interval exactly. The64 matched mean comparison cannot identify the missing448-only mean. The final MBR decision is non-injective: many distributions yield the same interval. It is invalid to invent the missing distribution from that decision. The human has now authorized the fixed63-query supplement after the active full evaluation completes.

## One complete case, H001

| Branch | tIoU % | Actual full vIoU % | Fixed same spatial track vIoU % | End absolute error s |
|---|---:|---:|---:|---:|
| Only224six | 34.7458 | 4.5312 | 4.5312 | 0.4391 |
| Only448 | 22.2222 | 3.2472 | 2.8980 | 6.0679 |
| Fusion | 24.3323 | 3.1732 | 3.1732 | 4.4711 |

This is a negative engineering case, not a random sample or an estimate of448 performance across64.224 improves the448-only outcome here, but the blend still harms224. All three remain below tIoU.5. The448 posterior is more concentrated yet produces a worse MBR decision: entropy alone cannot certify correctness. The posterior expected tIoU against GT is .2472 for448 versus .2011 for224, while their decoded actual tIoUs reverse (.2222 versus .3475); posterior summaries do not determine the correctness of a single decoded decision.

## All64 actual fusion rescue/harm

| Case | Global6 tIoU | Fusion tIoU | Start-error change s | End-error change s | Predicted length change s |
|---|---:|---:|---:|---:|---:|
| H007 | 0.4710 | 0.5254 | +0.0000 | -0.6000 | +0.6000 |
| H023 | 0.3133 | 0.5863 | -0.5990 | -3.1281 | +2.5291 |
| H043 | 0.4986 | 0.5648 | -1.2400 | +0.3200 | +0.9200 |
| H052 | 0.4955 | 0.6199 | +0.0000 | -2.1956 | +2.1956 |
| H008 | 0.5161 | 0.3977 | +1.8800 | +0.2800 | +1.6000 |
| H018 | 0.5236 | 0.4419 | -0.2995 | +1.8636 | +1.5641 |

All four rescues and both threshold harms expand the interval, so a simple extension rule cannot distinguish them. H007/H052 mainly correct premature ending; H023 improves both boundaries; H043 mainly improves the start while its end gets slightly worse. H008 expands toward a wrong start; H018 expands toward a wrong end. These are observations of boundary effects, not proof that448 recognized a particular action visually.

In equal posterior fusion, expected-IoU utility is linear. If224 selects a and fusion selects b, then u448(b)−u448(a) >= u224(a)−u224(b). The64 saved224 distributions permit this lower bound on the missing448 preference, but not its actual magnitude or standalone MBR choice. H023 requires at least .06467 of relative448 utility preference; that is a bound, not a reconstructed448 score.

## Implication for cheap call rules

Posthoc benefit>1pp versus harm>1pp cases total55; native normalized-entropy AUC is 0.3829, start-spread 0.6400, end-spread 0.4343. These exposed-data diagnostics do not establish a gate or confidence interval; no threshold was selected. In particular high entropy is not demonstrated to predict benefit.

## Concrete lightweight candidate: one high-resolution view

Keep the full224 six-layer global posterior. Replace the two-offset448 branch with one **fixed offset0**448 view. Map its normalized interval distribution onto the same physical grid, retain the existing .5/.5 fusion, originalMBR and intervalBoxFixed DP. Offset1 is a prespecified sensitivity control, never chosen by per-query GT. The interface needs only interval posteriors and physical endpoints; no actionness/hidden/attention score or extra teacher. The prototype operators are in scripts/branch_decomposition/lightweight_design.py.

The structural change is high-resolution views2→1, not a new mixing-weight search. Actual deployment would need one visual/multimodal/temporal view, not computing both and discarding one; it may roughly halve the incremental high branch work but no measured speed ratio or precision is claimed. It also removes the high branch's cross-offset hull: this could suppress some overextension harms but could lose rescue cases and offset complementarity. The low-resolution global context remains.

This candidate is a **prepared hypothesis, not evaluated or selected**. Full448-only versus fusion-only cases must first show whether useful evidence is already in one high view or depends on both. Missing raw448 per-offset logits prevent that test now; the one recovered aggregate448 posterior cannot uniquely recover the two original view distributions either.

## Minimal missing evidence, prepared only

The private fixed supplement plan lists63 original queries after the active full queue finishes:126 original448 offset forwards,0 new224 forwards,0 expert calls,0 spatial decoder,0 training. It preserves the exact inputs/checkpoint/precision/seed/two-offset/native flow and saves both per-offset logits and posterior. Reconstruct the historical equal fusion and require every old Strong interval, without weakening source hashes; seal before labels. The human explicitly authorized this extension, conditional on the current full evaluation finishing. The supplement is prepared and source/input/asset locked, but no model has started. The existing half-hour heartbeat was updated to continue this stage after full publication; no new automation was created.

After coverage is complete, the planned table is448-only /224six /full fusion /fixed-offset0 fusion /offset1 sensitivity, with common spatial trajectory and actual finalDP reported separately. No per-query GT branch choice or weight sweep. Only then decide whether the prepared lightweight candidate has precision and actual cost value.

## Verification and limits

One new cached query prediction panel sealed before locked-label scoring; 8 official/dense comparisons max error 5.55e-17. All other64 results are existing-score reaggregation. Source/ERef/positions/hidden binding and originalG/F predictions recover exactly. No RGB/model/head/decoder/expert/GPU/parameter update. All434 active-full sources and plan remain unchanged.

All64 have repeated development exposure. No new production method, no448-only full-panel mean and no cross-backbone precision claim. Detailed signed evidence and explicit coverage are in report.json/signed_queries.tsv. Private queries, raw boundaries/boxes/posteriors/features, video and checkpoints remain excluded.

## Authorized deferred execution

The approved63-query supplement is now locked to fingerprint `4015113aed7a56a88e7bf5c4bbec032c4e32ccfa849628f833d9a0c65a961dcb`. Ten semantic/barrier tests and actual early-run refusal pass; CUDA visibility survives module import without initialization. The hard gate requires full state `complete_published`, both domains scored, old owned processes exited and the sharedGPU lock. See prepared_supplement.json for exact commands and budgets. No further confirmation is required. Two preparation-only versions with syntax/import-environment issues produced no model calls or predictions; their plans/source snapshots are retained privately, not resumed.
