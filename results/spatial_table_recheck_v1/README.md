# Independent full-list recheck of the simple-combination and fixed-time spatial tables

User suspects implementation error or data imbalance. This study independently reconstructs pure interpolation from original paid ReferDINO raw8box payloads, rescores both full target lists using original GT, cross-checks the official evaluator, and independently audits normalized-coordinate scores. Earlier results/scientific plans are immutable; no method/paper change and no new Source/expert/model/GPU/optimizer.

Findings:

1. Exact within-domain query pairing: all7arms HC1902/227parents, Vid10303/732parents, no missing/extra/duplicate/outcome-filtered query; same nativephysicalgrid/8observations/GTsupport. Fixed-time rows have tIoU identical per query. Equal-query main means and equal-parent secondary means retain allqueries; HC-positive/Vid-negative mean directions persist under both aggregations. Different benchmark sizes alone do not establish unfair pairing.
2. Historical numbers reproduce for historical center/log-size interpolation. Independent interpolation maxcoordinate roundoff<=8.881784197e-16; maxold perquery metricdifference3.663735981e-15. Main simple baseline25.140858HC/14.709589Vid; fixed-timepure25.177364/14.704394; unchangedFull27.349044/14.558151.
3. Prior wording pure interpolation failed to distinguish log-size from direct linear box-coordinate interpolation. The latter(equivalent to center+linearwidthheight) was prespecified as a diagnostic before new GT scoring and is stronger here: simple25.556767HC/14.810023Vid; fixedtime25.592737/14.804402. Full gains against fixedtime linearpure+1.756307HC/-0.246251Vid. Report the chosen baseline definition explicitly; never pick a rule by favorable target results or relabel older scores.
4. Intended hard-anchor contract is still unsatisfied. Raw expert anchors retained exactly by both pure rules; Full outputs different E boxes at10556/15216HC and50874/82424Vid observed positions, including3735/15555positions inside predicted Full events(unrestricted by GT support). Every other observed output is exactly rawexpert; no random coordinate corruption. This is the known unchanged soft-candidate implementation, not a repaired hard-anchor method. Numerical validity does not establish intended-method correctness. No final overwrite/DP constraint patch/adoption is performed; observed constraints must eventually enter DP itself for the intended method.

36615new official t/v/s checks, independent all7arm dense scoring and separate normalized-coordinate bracket-interpolation scalar audit(maxerror2.331468351713e-15),219690array-field checks=18fields/query*12205queries(not any resumed corruption suite),126means,120ROOTpaired/timegroupCIchecks. All6actualchildwait0/controller88812actualwait0, ROOT54902actualwait0; allownedincarnations exited. Four fixed actual E-ROIupdate/MAP/finalDP cached replays reproduce existing Full exactly; this is a four-codepath diagnostic, not an allquery new DP method rerun. Full old arrays themselves are hash-verified on everyquery and freshly rescored.

Original raw teacher bodies/receipts/context include current and31original archivedHC producer states, identified by exact dataset/caption/fullphysicalgrid/positions/RGBcontext and all32rawboxcoordinates matching immutable production. Exact producerfingerprints/payloadSHA/receiptSHA bound privately, no hash relaxed or oldcache rewritten. Three preparation-only provenance/path exceptions occurred before0predictions/model/Source/GTscore/planfreeze; original sources/logs and sanitized diagnosis retained. No skip/fallback/blindnewteacher retry.

- [Complete analysis](ANALYSIS.md)
- [Fresh HC scores](hc2/report.json)
- [Fresh Vid scores](vidstg/report.json)
- [Equal-parent secondary aggregation](equal_parent_secondary.json)
- [Coverage preflight](coverage_preflight.json)
- [Frozen protocol](protocol.json)
- [Four cached DP replays](four_case_current_DP_replay.json)

Allsigned/severe t/vuniononce>5pp/rescue/shortlong/recall/observed-unobserved metrics are retained.10000pairedparent bootstrap seed20261022,queryequal main,pointwise,exposeddata,no equivalence/familywise/unknownvalidation/targettuning claims. Rawboxes/features/captions/GTcoordinates/media/masks/weights remainprivate. Alloriginalmethod/oldplan/hash/production/paper/protecteddrafts and paused/cancelled tasks unchanged. Cached CPU scoring/IO wall is not deployment speed; inherited Source/expert acquisition nonzero, nested clocks not additive. Final complete_published requires independent ALLremote receipt.
