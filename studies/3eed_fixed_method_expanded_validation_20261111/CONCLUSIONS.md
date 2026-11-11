# R69 interpretation and branch decision

**Do not promote the fixed R66 recipe to a robust cross-platform TTA method. Stop gamma/eta/update-site searches on this recipe. Preserve its positive VAL result, R59, frozen ABC and fixed random sampling as explicit baselines; no new algorithm or follow-on experiment is queued.**

The experiment held Waymo-only epoch100, gamma1, eta1, candidate boxes, correspondence, partial-set/source-KL loss, original one-step/backtracking rule and three random-seed rules fixed. It changes evaluation coverage only. Scores below are platform-equal mIoU percentages.

| Fixed method | VAL1280 Drone | VAL1280 Quad | VAL1280 average | TRAIN1024 Drone | TRAIN1024 Quad | TRAIN1024 average |
|---|---:|---:|---:|---:|---:|---:|
| Source | 10.712182 | 25.584854 | 18.148518 | 16.504735 | 19.482304 | 17.993519 |
| Frozen ABC | 11.255613 | 25.965582 | 18.610598 | 16.673721 | 19.468151 | 18.070936 |
| R59 | 11.483505 | 26.517059 | 19.000282 | 16.830404 | 19.169256 | 17.999830 |
| R66 + R59 | 11.644900 | 26.527055 | 19.085978 | 16.268579 | 19.028274 | 17.648427 |
| Random + R59, mean of three policies | 11.212686 | 26.427003 | 18.819845 | 16.703122 | 19.640218 | 18.171670 |

The random platform means in this table are arithmetic policy means, not one ensemble prediction. All three policies/negative rows remain in the complete report. R66−R59 is +0.085696pp on VAL and −0.351404pp on TRAIN. R66−random policy mean is +0.266133pp on VAL and −0.523244pp on TRAIN. New-recording failure is visible in both platforms and in three of four TRAIN recording sequences (the fourth is unchanged), rather than only one harmed example. This is not an all-scenes-must-improve adoption rule: the overall TRAIN metric itself is negative.

## Revise the tentative source-restoration explanation

The old DEV observation that one source-restoration case dominated R66's increment does **not** describe the new positive changes. Relative to R59, all ten improved VAL expressions and all seven improved TRAIN expressions also exceed their own source IoU. There are zero exact-source-query-restoration improvements and zero equal-source-IoU restoration improvements in either cohort. Thus the new evidence cannot be summarized as merely undoing R59 failures.

On VAL, ten improvements contribute +0.190779pp gross; eleven deteriorations contribute −0.105083pp, giving +0.085696pp net. The largest positive contributes +0.041093pp and the remainder remains +0.044602pp, each using all1,280 expressions as denominator. This is weaker concentration than the earlier DEV result. Acc@25 stays34.6875%; Acc@50 rises17.5→17.65625%. The positive is small but real under the fixed task metric and should be retained.

On TRAIN, seven improvements contribute only +0.011018pp gross while fourteen deteriorations contribute −0.362422pp. Net is −0.351404pp; Acc@25 falls34.863281→34.179688%, Acc@50 falls16.699219→16.601563%. Accepting a nonzero loss-decreasing edit is not sufficient to select a better target.

## Feedback and transfer limitations

The same-evidence E-direct/set-direct results are18.639013/19.011778% on VAL and17.158543/17.365964% on TRAIN. The learned update exceeds these particular direct-use controls, including on TRAIN, but still fails to exceed R59 on new recordings. Parameter fitting can moderate a harmful direct-use rule without making the resulting system beneficial. This supports a limitation of the current feedback/readout/update combination; it does not independently identify every wrong pseudo-label or prove that another parameter interface could never work.

C adapted minus C source is +0.081649pp on VAL and −0.292524pp on TRAIN. Therefore neither a universal C-transfer failure nor a universal C-transfer success is supported. C is another sampling of the same physical cloud, mapped back to canonical A256 boxes. It is not a new camera view or independent scene.

R59 beats source by +0.851764pp on VAL but only +0.006311pp on TRAIN, and is −0.071106pp below frozen ABC on TRAIN. Random sampling is −0.180437pp below R59 on VAL (all three seeds below) and +0.171840pp above R59 on TRAIN (all three above). Therefore neither R59 nor random is established as a uniformly stronger replacement across these recording conditions. Keep both as necessary no-update competitors; do not select the best seed or merge the cohorts to hide opposite directions.

Frozen A256 contains an Acc@25-capable candidate for1,010/1,280 VAL and796/1,024 TRAIN expressions. There are566 and439 recoverable R59 failures respectively. This leaves selection headroom under the fixed boxes, but does not supply an unlabeled rule for finding those candidates. Candidate-oracle coverage is an offline diagnostic only.

## Data, audits and operational boundary

VAL1280 has1,033 unique media pairs in the same five official VAL recording sequences and excludes DEV336/FINAL384 media identities. Full VAL historically had source/other-method exposure; it is not a pristine unseen-domain test. TRAIN1024 has1,024 unique media pairs in four additional official recording sequences and distinct nominal place-name families. It is byte-disjoint from all VAL media; empty origin headers and sequence-local timestamps do not certify global geographical or absolute-time independence. Report both cohorts separately.

All347 eligible A/B fits were sealed before C;346 accepted nonzero updates and one rejection remain in the denominators. All2,304 predictions were sealed before task labels. The independent all-input evidence/parameter/correspondence/11-arm readout audit has zero maximum probability error; prelocked72-input GPU replay is bitexact. Independent polygon geometry checks all589,824 candidate IoUs with maximum error2.604584e−13 and matching25/50 threshold decisions. All25,344 task values and13,296 ordered negative comparisons are retained, with recording/frame-cluster summaries.

Engineering corrections were limited to a prospective streaming Decimal/float and integer-height-cast interface check, and a missing independent-auditor import. Synthetic preflight, previous drafts, failed controller/auditor record and corrected audit-plan chain are preserved. The fixes preceded task scoring and did not change source weights, predictions, feedback, fit parameters or any method hyperparameter. The scoring parser now explicitly matches the legacy float64-transform/float32-GT convention. No production-model rerun was needed for these corrections.

**The next decision is to stop this fixed R66 optimization branch, rather than claim that 3EED or all TTA is impossible.** The current benchmark still has recoverable ranking errors, but stable unlabeled supervision on additional recording conditions remains unestablished. Any future attempt needs a separately justified evidence or training-preparation hypothesis, not another small strength/threshold/editor search on these newly exposed labels. No such attempt is launched in this round. Source retraining and new FINAL384 inference/scoring remain off; FINAL identity/media hashes were used only for overlap exclusions.
