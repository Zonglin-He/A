# HC2 temporal readout: broader parameter search and exploratory full comparison

The selected start-temperature profile improves the previous Global6 on both development panels, but **does not pass the original validation Source-superiority gate**. That failure is preserved. The maintained main method is unchanged. A separate exploratory HC1902 full comparison was started but the user rejected this launch because the Source gate had failed. It is now paused at338 verified queries; no scoring or automatic resume is authorized. This launch was an execution error, not an acceptable substitute for passing the gate. It is not labeled validation-passed or an independent confirmation.

## Scope and implementation

This study fixes64 tuning and32 validation parents selected from the official HC1902 input list by a label-free SHA ranking, excluding all96 parents used in the previous v2 study. All official full outcomes had already been exposed; these remain target development panels. All96 captures sealed before scoring. Ten complete captures were reused and86 native Source captures were saved, with **zero new ReferDINO forwards**. Source/ERef/native intervals recovered old full outputs exactly. The original Source forward, precision, two offsets, sampled RGB, eight observations, ERef and spatial BoxFixed are fixed.

Five output-side parameters are inspected: separate start/end logit temperatures; original hull versus native-offset mixture; last1/3/6 temporal layers; and exact expected-IoU utility power. Default parameters reproduce old Global6 posterior/time bitwise on all96 old saved captures and final time/boxes on the64 new tuning inputs. All37 CPU engineering tests passed. No actioness rule,448,replay,training or new expert is added. [Protocol](protocol.json), [pre-label coarse schedule](coarse_schedule.json), [sequential decisions](decisions.json), [implementation/parameter map](../../docs/temporal_hc_v3.md).

## Every tested profile

Deltas are percentage points versus original Global6 on the same64 tuning queries. Eligibility is the unchanged predeclared development rule, not evidence of independent generalization.

| Profile | Start τ | End τ | Hull weight | Last layers | IoU power | ΔtIoU | ΔvIoU | Δunobserved sIoU | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| [base](tune/base.json) | 1 | 1 | 1 | 6 | 1 | +0.0000 | +0.0000 | +0.0000 | False |
| [end075](tune/end075.json) | 1 | 0.75 | 1 | 6 | 1 | -0.7092 | -0.1976 | +0.1044 | False |
| [end125](tune/end125.json) | 1 | 1.25 | 1 | 6 | 1 | +0.1177 | -0.1029 | -0.0536 | False |
| [hull000](tune/hull000.json) | 1 | 1 | 0 | 6 | 1 | -2.5101 | -1.0851 | +0.4278 | False |
| [hull050](tune/hull050.json) | 1 | 1 | 0.5 | 6 | 1 | -0.0530 | +0.1107 | +0.1514 | False |
| [hull075](tune/hull075.json) | 1 | 1 | 0.75 | 6 | 1 | -0.3000 | -0.1658 | +0.1008 | False |
| [last1](tune/last1.json) | 1 | 1 | 1 | 1 | 1 | -0.2573 | -0.3605 | -0.1535 | False |
| [last3](tune/last3.json) | 1 | 1 | 1 | 3 | 1 | -0.2787 | -0.3818 | -0.1480 | False |
| [risk050](tune/risk050.json) | 1 | 1 | 1 | 6 | 0.5 | -0.4351 | -0.3359 | +0.0490 | False |
| [risk150](tune/risk150.json) | 1 | 1 | 1 | 6 | 1.5 | +0.0908 | +0.0748 | -0.0074 | False |
| [risk200](tune/risk200.json) | 1 | 1 | 1 | 6 | 2 | +0.4267 | +0.2884 | -0.0511 | True |
| [risk300](tune/risk300.json) | 1 | 1 | 1 | 6 | 3 | +0.3323 | +0.2356 | +0.0063 | True |
| [start075](tune/start075.json) | 0.75 | 1 | 1 | 6 | 1 | -1.1907 | -0.7508 | +0.2886 | False |
| [start125](tune/start125.json) | 1.25 | 1 | 1 | 6 | 1 | +0.3232 | -0.0485 | -0.0918 | True |
| [start150](tune/start150.json) | 1.5 | 1 | 1 | 6 | 1 | +0.5326 | -0.0271 | -0.0287 | True |
| [start150_risk200](tune/start150_risk200.json) | 1.5 | 1 | 1 | 6 | 2 | +0.4375 | +0.0047 | -0.0930 | True |

## Why these parameters behave differently

The hull is not just averaging two estimates: it combines minimum starts with maximum ends. On the old tuning64, without opening labels, expected duration was7.9594s under the hull versus5.0662s under an equal native mixture. This structural expansion is not proof of damage. The new controlled trial found that removing the hull improved short events+2.8730pp but harmed long events−4.6165pp and net tIoU−2.5101pp. Partial hulls were also not eligible. They demonstrate why correcting mean duration or signed boundary bias is insufficient. [Process-only probe](structural_probe.json).

Reducing the six-layer ensemble to last1/last3 hurt short-event tIoU by more than2.3pp and failed vIoU/unobserved protection. Last1 recovers Source time exactly, but its BoxFixed spatial path is not the unadapted Source spatial output. The six-layer baseline therefore remains for subsequent controls.

Sharpening start or end reduced mean signed bias but hurt total tIoU. Softening only start to1.5 improved long and baseline-bad cases, at a short-event cost. End-only softening was weaker. This is empirical uncertainty/readout calibration, not a claim that every early prediction should move earlier or later.

Changing MBR utility power to2 improved tuning tIoU+.4267/vIoU+.2884pp, with short+1.3165 and long+.0786pp, while mean selected duration barely changed(8.4863→8.5429s). Power1.5 and3 were weaker; .5 hurt. It changes the decision preference among posterior modes without fixing the underlying posterior. Its cached CPU trial cost was.0525s/query versus.0106s baseline, including posterior/DP/file writes; this excludes original Source,expert and RGB costs and is not a deployment speedup claim.

The sole allowed combination of start1.5 and utility2 had tIoU+.4375pp, below start1.5 alone+.5326pp. Improvements were nonadditive; the combination was not chosen. Selection remained maximum eligible tIoU then vIoU, fixed before validation. [Frozen selection](selection.json).

## Single locked32-parent validation

| Method | vIoU (%) | tIoU (%) | Mean interval (s) | Boundary MAE (s) |
|---|---:|---:|---:|---:|
| Source | 26.8425 | 50.6267 | 8.9105 | 3.1430 |
| Baseline | 25.9784 | 49.7809 | 8.9989 | 3.2252 |
| Candidate | 26.3889 | 50.1492 | 9.8587 | 3.0941 |
| Candidate_fixed_space | 26.3527 | 50.1492 | 9.8587 | 3.0941 |

Selected start1.5/end1 changes tIoU by+.3683pp and vIoU by+.4105pp versus old Global6 on the32 validation parents. Unobserved sIoU changes+.0341pp. Holding spatial boxes fixed still gives vIoU+.3743pp, so most gain is temporal. However, tIoU is.4775pp below Source. **Original validation eligible remains false**; the Source gate is not removed, rewritten, or reported as passed. The validation tIoU difference has paired95% CI[−1.3296,+2.0819]pp and vIoU CI[−.6982,+1.6022]pp; both cross zero, so these positive point estimates do not establish a statistically stable improvement. All confidence intervals and per-query signed values remain in the [validation report](validation/start150.json). No fallback was selected or scored on these validation parents.

## Explicit exploratory extension and full status

The original user requested tuning followed by full reevaluation. To assess the one already frozen profile on the full official list, a separate extension was recorded before any new full outcomes. The original v3 protocol, failed Source eligibility and selected profile are unchanged. This is additional descriptive target-domain assessment, not a way to relabel the failed gate. It does not automatically promote the candidate or allow parameters to be chosen from the full results. [Extension card](exploratory_full_extension.json), [outcome](outcome.json), [new full protocol](../hc_temporal_full_v3_exploratory/protocol.json).

The new HC1902 queue has fingerprint `f33e75a4652dfc369277b92e90cb63e3014b0c046f13a9ed02e9975828cb208d`. It stores independent predictions/receipts and compact native logits/geometry, reuses original content-bound expert evidence, and must recover every old Source/ERef/Global6 interval and BoxFixed trajectory bitwise. The completed old HC/Vid full remains immutable; VidSTG is not rerun. The main CLI and default configuration remain BoxFixed + original Global6 + original physical MBR.

After all1902 predictions seal, a separate CPU scorer uses fixed official labels, verifies old Source/Previous full metrics, and reports Source/Previous/Candidate plus fixed-space time control, all signed effects, serious declines and parent-cluster intervals. No partial-full result selects parameters. The user paused this full at338/1902 and no full score exists. All338 prediction/receipt/capture hashes were verified, no orphan files remained, and controller/worker exited. No full improvement is claimed. Further full launch requires actual Source-primary validation, with no exploratory exception.
