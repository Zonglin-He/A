# Corrected Spatial WHO time: finite matched control

Forty-eight fresh STVG-R1 calls on twelve previously exposed development sources are complete. All historical videos, prompts, candidate IDs/geometry, FPS, weights and generation settings are fixed. Only the isolated fractional-time coercion changes. Historical Temporal-Lite output intervals and physical centers stay fixed. This is not a full-method corrected-Temporal comparison.

Actual prefill positions change for 36/48 requests. Only one selected ID changes: one HC background region switches to Source. All Vid selections stay the same. The 12 requests whose positions did not change also preserve the same selected ID. These actual-input controls do not establish that correcting Temporal caption generation has no effect; that was not measured here.

The unchanged eleven sources reuse their exact previous M2 fit and complete Student predictions after geometry/source/configuration/seed-key checks. They are not eleven fresh GPU runs. The single changed-reference source receives one real 1792-parameter fit with five accepted updates and one complete Student re-decode. Independent saved rollout/loss/Adam/response-solver/rollback checks pass, and the full model's final boxes exactly match the saved decoder output. The whole twelve-source group is sealed before this control's raw-GT scoring.

| Corrected Spatial clock Full minus legacy Full | Mean vIoU change | Paired signs |
|---|---:|---|
| HC, 6 sources | −0.121771 pp | 0 positive / 1 negative / 5 unchanged |
| VidSTG, 6 sources | 0.000000 pp | 0 positive / 0 negative / 6 unchanged |

The one changed HC source loses 0.730626 pp relative to its legacy Full result. Removing a visibly poor background reference does not guarantee better complete-Tube propagation under the remaining reference. A reference can be semantically poor while its parameter effect accidentally helps some geometry. This result does not validate learning from bad references, nor does it justify retaining defective time arithmetic for a higher score.

The panel contains two positive, two negative and two unchanged spatial outcomes per domain, selected from sealed development128. Its means are diagnostic, not a representative performance estimate or a replacement for the earlier fixed64 confirmation. Every Frozen, Temporal-Lite, legacy Full, corrected-clock Full and legacy M0 result, including negative/tied/null values, is exported. Legacy M0 uses the historical reference, so it is not claimed as a new corrected-reference same-information M0 control. All arms use the same four original observation positions when comparing supervised/unsupervised metrics.

Actual teacher worker wall is 59.0405 s for the 48 calls, including worker initialization. Actual one-source Student worker wall is 20.8669 s; the independent saved-math root worker is 1.5822 s. These costs exclude inherited candidates/Temporal and do not support a cold deployment speed claim. No new model, tracking algorithm, prompt/LR search, Oracle fit, Bootstrap, CURRENT promotion or official full-set run occurred.

## Decision

The historical geometry patch is not qualified: none of the 22 adopted component masks exactly replays, and visual decomposition does not show correct-instance truncation dominating errors. R0/R1/R2 historical fitting is therefore not run. The isolated Spatial time correction does not solve Vid identity selection or its spatial losses. There is no evidence-qualified new spatial recipe to freeze and expand to confirmation in this finite branch; no extra confirmation sources are consumed.

The unresolved next mechanism is obtaining instance-specific spatial evidence rather than a background/part region endorsed by an ID decision. No new candidate generator or identity prompt is silently chosen here. The earlier whole-scene and pairwise-rejection negative findings remain intact; this control is not a renamed rerun of either. The correction is an interface repair, not a new paper contribution. Historical old-protocol results remain labeled and preserved, with no cross-version cache relabeling.

See the preceding [geometry decomposition](STVG_SPATIAL_REFERENCE_GEOMETRY_AUDIT001.md) for the twelve-source visual review, all provenance failures and the broader time-interface audit. Private media, annotation coordinates, weights/tensors, actual Query/caption text and raw teacher outputs are not included in this publication.
