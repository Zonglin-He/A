# HC2 temporal calibration: bounded search and failed fixed validation

The selected temporal profile did not pass the locked 32-parent development validation. The maintained BoxFixed + Global6 + original physical MBR configuration is unchanged. **No new HC1902 full run was launched.** The already published full v1 remains intact.

## Protocol and verified scope

A label-free SHA-ranked official panel used 64 tuning parents and 32 disjoint validation parents, one query per parent. All 96 captures sealed before panel scoring. Six complete captures were reused; 90 native Source captures were generated, with zero new ReferDINO forwards. Source/ERef/native intervals matched v1 bitwise. Every trial prediction sealed before independent CPU scoring. All official full outcomes had already been exposed: this is target-domain development, not an independent test.

Only endpoint temperature, global temporal-layer weighting and physical posterior duration tilt varied. Source, original eight expert observations, ERef, spatial propagation/weights, FP32, physical input support, old two-offset hull and original MBR stayed fixed. There was no actioness rule, 448 branch, conditional replay, training or additional expert. The common frozen temporal head has auxiliary supervision at preceding layers; early layers are not assumed untrained.

The finite search used 14 profiles, including the baseline, three neighboring refinements and one combination. Selection was frozen before the single validation. No fallback is selected using validation and no protection rule is relaxed. [Locked protocol](protocol.json), [stage decisions](decisions.json), [pre-validation selection](selection.json), [all signed validation values](validation/tau150.json).

## All tuning profiles

Deltas below are percentage points against the same original Global6 baseline on the same 64 queries. Both favorable and unfavorable profiles are retained.

| Profile | Temperature | Layer slope | Length power | ΔtIoU | ΔvIoU | Δunobserved sIoU | Eligible |
|---|---:|---:|---:|---:|---:|---:|:---:|
| [base](tune/base.json) | 1 | 0 | 0 | +0.0000 | +0.0000 | +0.0000 | False |
| [combination](tune/combination.json) | 1.5 | -1 | 0 | +0.7269 | +0.2899 | +0.0432 | False |
| [early1](tune/early1.json) | 1 | -1 | 0 | +0.4313 | +0.2870 | +0.3517 | True |
| [early2](tune/early2.json) | 1 | -2 | 0 | +0.1598 | +0.0729 | +0.5703 | False |
| [early_half](tune/early_half.json) | 1 | -0.5 | 0 | -0.1300 | -0.0441 | +0.0398 | False |
| [late1](tune/late1.json) | 1 | 1 | 0 | -0.4592 | -0.1463 | +0.0382 | False |
| [late2](tune/late2.json) | 1 | 2 | 0 | -0.4387 | -0.1647 | -0.0210 | False |
| [length_neg05](tune/length_neg05.json) | 1 | 0 | -0.5 | +0.5379 | +0.2660 | +0.2301 | True |
| [length_pos05](tune/length_pos05.json) | 1 | 0 | 0.5 | -0.6973 | -0.1542 | +0.1453 | False |
| [length_pos1](tune/length_pos1.json) | 1 | 0 | 1 | -1.7306 | -0.5991 | +0.3031 | False |
| [tau075](tune/tau075.json) | 0.75 | 0 | 0 | -1.4101 | -0.3917 | +0.2778 | False |
| [tau125](tune/tau125.json) | 1.25 | 0 | 0 | +0.4913 | +0.0546 | +0.1659 | True |
| [tau150](tune/tau150.json) | 1.5 | 0 | 0 | +0.8584 | +0.2973 | +0.1200 | True |
| [tau175](tune/tau175.json) | 1.75 | 0 | 0 | +0.3996 | +0.1329 | +0.0874 | True |

## Locked validation result

| Method | vIoU (%) | tIoU (%) | Mean selected duration (s) | Boundary MAE (s) |
|---|---:|---:|---:|---:|
| Source | 23.3553 | 46.7538 | 9.9489 | 3.8952 |
| Baseline | 26.7771 | 47.3888 | 9.9521 | 3.7141 |
| Candidate | 25.6912 | 45.8283 | 10.9872 | 3.8289 |
| Candidate_fixed_space | 25.6856 | 45.8283 | 10.9872 | 3.8289 |

Temperature 1.5 was selected on tuning: ΔtIoU +0.8584pp and ΔvIoU +0.2973pp. Validation instead yielded ΔtIoU −1.5605pp (paired parent CI [−4.9105,+1.0596]) and ΔvIoU −1.0859pp (CI [−3.6867,+0.8626]). It failed both gain and vIoU protection gates, and its tIoU was 0.9255pp below Source. Severe vIoU declines >5pp versus Source were 6 for the baseline and 7 for the candidate. Against baseline, 17 queries lost vIoU and 10 improved; 2 lost more than 5pp.

Holding the complete baseline spatial trajectory fixed gives vIoU 25.6856%, close to the candidate’s 25.6912%. Thus the decline is attributable almost entirely to changing time intervals, rather than to a spatial path failure.

## Mechanism and parameter interpretation

Lower temperature sharpens the posterior and shortens intervals. It improved short events but hurt long-event coverage, producing a net tuning tIoU loss at 0.75. Higher temperature softens the posterior and improves some sharp wrong/truncated decisions; however it can spread mass across wrong boundaries. Temperature 1.75 weakened the tuning gain and damaged already-good cases.

The selected 1.5 increased validation mean duration from 9.9521s to 10.9872s, while mean GT duration was 6.9798s. Short-event tIoU changed by −5.8982pp (9 queries); long-event change was +0.1369pp (23). Already-good baseline time predictions changed by −4.6377pp (15); baseline-bad cases improved +1.1548pp (17). These GT strata are offline explanatory analysis only, never an inference condition. A global uncertainty change trades truncation against overcoverage and does not reliably distinguish these cases.

Moderate early-layer weighting improved the tuning ensemble without materially expanding expected duration; stronger early weighting lost the advantage and failed severe-decline protection. Later-layer weighting hurt this tuning panel. This is evidence about target-domain ensemble behavior, not proof that early layers should replace late layers in general. Duration penalties produced the same short-versus-long conflict in the opposite direction. Combining the best temperature and early weighting failed the prespecified severe-vIoU protection gate on tuning.

The experiment does not show that every possible time method or parameter is ineffective. It does reject promoting the best observed global calibration candidate. Repeating weight search on these 32 validation parents would not supply confirmation. A subsequent study must name a distinct failure mechanism, freeze its selection rules before a fresh assessment, and disclose prior exposure; none is silently added here.

## Execution and evidence status

32 CPU engineering checks passed, including exact default-profile compatibility and posterior validity. These checks establish implementation behavior, not model improvement. All 14 tuning reports retain per-query signed changes and confidence intervals. Private raw captures, sealed predictions, source snapshots and receipts remain local. Public reports include no media, text queries, GT boxes, weights, masks or feature tensors. Capture reuse costs exclude the original generation cost and are not an end-to-end speedup claim.

The prepared full controller is unexecuted and refuses preparation unless the locked validation is eligible. No new full plan or full prediction exists. The selected candidate is rejected; default temperature remains 1, equal six-layer mean, duration power 0.
