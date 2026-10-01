# Where correct predictions are lost: sealed Paper48 CPU diagnosis

The clearest observed damage point in the completed VidSTG panel is current-query temporal expert reranking. It contributes a small positive mean but substantially more gross harm than the inherited spatial state. This is a post-hoc diagnosis of the original frozen J01 recipe, not an evaluation of the ongoing tuned v3 configuration, and not a prescription to remove the temporal branch.

All 8,040 VidSTG P1 and 768 HC-STVG-v2 P5 saved arrivals were processed using CPU geometry and diagnostic GT. No backbone, expert, decoder, gradient, or GPU execution was performed. All existing receipt and raw-prediction hashes were verified before label access. The 35,232 Frozen/Ours endpoint scalar comparisons agree with the original audited scores to maximum error 1.67e-15. Current tuning and its selection rules were not changed.

## Setting and exact decomposition

P1: 670 VidSTG-test sources, one fixed query/source, two independent orders, clean and five 5% transient corruption streams. P5: 128 HC2 validation sources, one query/source and one order, same six conditions; original project 224px/nominal64 sampling, not the official420px benchmark. Both use their official same-domain TA-STVG checkpoint and the frozen default recipe: 25% temporal/spatial expert availability, 1,792 spatial parameters, SGD .005, K1, radius .05. Original sampling is retained. All sources are historically exposed; this additional GT inspection is development/diagnostic exposure.

For each arrival we evaluated: Frozen boxes/interval -> inherited-state boxes with Frozen interval -> actual inherited-state boxes/interval -> current temporal-reranked output. The mixed intermediate is an output substitution for accounting, not a neural-module intervention. The inherited intervals were unchanged in every saved arrival here. Consequently inherited-state gains/losses arise from boxes in this observed panel. The current spatial update occurs after the current output is sealed and can affect subsequent arrivals; the diagnostic cannot attribute future loss to one particular prior update.

All following gains and intervals use source-macro averaging with 10,000 source bootstrap replicates, conditional on the existing order(s). Transition counts/rates instead count repeated arrival cells; their denominators are explicit and they are not independent-source probabilities.

| Corrupt all-arrival effect, vIoU pp | VidSTG P1, 6,700 cells | HC2 P5, 640 cells |
|---|---:|---:|
| Inherited spatial state | +0.3640 [+0.3013,+0.4346] | +0.2838 [-0.2082,+1.0678] |
| Current temporal reranking | +0.1072 [-0.1707,+0.4070] | +0.2100 [-0.1917,+0.6543] |
| Final total | +0.4712 [+0.1905,+0.7756] | +0.4939 [-0.1964,+1.3960] |

The signed contributions sum exactly. Gross loss means the magnitude of negative changes only, divided by all cells in that group; it is not net effect. On the common 6,700-cell VidSTG denominator, inherited-state gross loss is 0.0951 pp and temporal reranking gross loss is 0.5197 pp (about 5.47 times larger). Their gross gains are 0.4591 and 0.6268 pp. Thus temporal reranking has both useful rescues and substantial harmful replacements. HC2 gross losses are 0.2980/0.3547 pp, respectively; its failure profile differs.

## Originally correct predictions becoming incorrect

Correctness here is the existing strict vIoU>.3 or vIoU>.5 recall criterion, not semantic perfection. Stage denominators count predictions correct immediately before that stage.

| VidSTG corrupt expert-scheduled cells (1,680 total) | Before-stage correct at .3 | Correct -> wrong | Conditional damage rate | Wrong -> correct |
|---|---:|---:|---:|---:|
| Inherited spatial state | 496 | 1 | 0.20% | 21 |
| Temporal reranking | 516 | 70 | 13.57% | 71 |

At threshold .5, temporal reranking destroys 30/257 previously correct expert arrivals (11.67%) and rescues 53. The 70 threshold-.3 damaging cells come from 20 unique video sources, not 70 independent videos. Across all 6,700 corruption arrivals, inherited-state damage is 10/2,046 at .3, whereas temporal damage is 70/2,107; these denominators include arrivals on which the temporal branch is not called and should not be mistaken for its invocation-conditioned rate.

HC2 differs: temporal reranking destroys 0/68 at .3 and rescues 5; at .5 it destroys 3/15 and rescues 8. This is a small, one-order cohort; the VidSTG rate is not universal.

## Correct candidates are available but not selected

Temporal candidate GT evaluation keeps the same inherited boxes for every candidate interval. This separates existing-support availability from critic selection. vIoU and tIoU oracles are reported separately; neither becomes an online selector.

| Dataset / correctness | Scheduled cells | At least one correct candidate | Correct candidate present, selected output incorrect | Conditional miss rate |
|---|---:|---:|---:|---:|
| VidSTG vIoU>.3 | 1,680 | 691 | 174 | 25.18% |
| VidSTG vIoU>.5 | 1,680 | 363 | 83 | 22.87% |
| VidSTG tIoU>.5 | 1,680 | 971 | 241 | 24.82% |
| HC2 vIoU>.3 | 160 | 76 | 3 | 3.95% |
| HC2 vIoU>.5 | 160 | 37 | 17 | 45.95% |

For VidSTG, 989/1,680 scheduled cells have no vIoU>.3 candidate even with oracle selection. That joint limitation includes inherited box quality as well as temporal support and must not be labeled a pure temporal-candidate failure. Conditional on the actual saved support, source-macro oracle-minus-selected vIoU headroom is +4.9444 pp among scheduled sources, or +1.2732 pp when propagated as a zero-on-nonexpert diagnostic over the full original population. These are diagnostic attainable candidate choices using GT, not achieved method gains.

## Why the temporal ranking fails in the observed cases

The deployed critic assigns each student interval the maximum, over expert proposals, of proposal confidence times temporal overlap. It uses argmax, with native first as tie fallback. It is a ranking bridge, not a learned accept/reject gate.

For the 70 VidSTG .3-correct predictions destroyed by reranking:

- All 70 had at least one expert proposal with GT tIoU>.5.
- In 63/70, the expert proposal furnishing the selected candidate's winning score had GT tIoU<=.5, despite a good expert proposal being available.
- None of the 70 was a tie between the chosen candidate and the vIoU-oracle candidate.
- 41 lost temporal overlap with the GT event; 50 increased predicted interval length. These descriptions overlap and are not an exclusive partition.

Among the 174 missed-correct-candidate cases, 142 have the same good-proposal-available/bad-winning-proposal pattern; six have no proposal above GT tIoU .5. None is a top-score tie with the vIoU oracle. The evidence localizes a useful next investigation to confidence-weighted proposal-to-candidate ranking. It does not prove confidence weighting alone is causal: candidate geometry, proposal quality, and joint space/time scoring interact, and no altered ranking rule was run.

A representative preserved case (order2 arrival300, occlusion5; anonymous source index is included in the public rows) changes vIoU 0.5443 -> 0.0297 and tIoU 0.8677 -> 0.0537. The expert contains a proposal at GT tIoU0.8735, yet the proposal supporting the chosen candidate is at0.0261. Critic scores prefer the harmful selected interval0.3090 over native0.2882. This is a concrete selection failure without any newly generated prediction. Representative worst cases are explicitly selected by damage for illustration, not representative prevalence estimates.

## Spatial rejection and unavailable evidence

Spatial update has no confidence-based correctness gate in this recipe. With zero valid expert frames, rewards are None and update is skipped; otherwise SGD is attempted. Among corrupt scheduled arrivals, VidSTG skips211/1,680 (12.56%) for no valid expert frames and HC2 skips20/160 (12.5%). No zero-gradient skip appears in these panels. Flat rewards occur in114 VidSTG and5 HC2 arrivals; a flat target need not produce zero gradient. Decreasing pseudo loss is not evidence of improved GT accuracy.

These counts do not establish that a correct spatial candidate was rejected. The compact full-run payloads omit spatial candidate box tubes and post-update output tubes; the current v3 compact writer also omits them. Their GT ranking and own-query update effects cannot be reconstructed by CPU geometry from these files. No model was rerun to fill this gap, and no live pinned writer was modified.

## Continuation and reproducibility

The user's final tuned full-evaluation diagnosis remains deferred until tuning finishes and that evaluation is concretely locked and sealed. This CPU pass does not start a full GPU job, select a new hyperparameter, or change the frozen method. Before such a future run, the root should arrange isolated diagnostic logging of spatial candidate/post-update outputs without changing inference behavior, if full spatial attribution is required. Existing v3 search/confirmation remains isolated from this GT case inspection.

Run `scripts/diagnose_tastvg_pipeline_cpu_v1.py` then `scripts/diagnose_tastvg_critic_cpu_v1.py` in the authorized private workspace to reproduce geometry. Public anonymous rows and aggregate checks are reproducible using `python scripts/audit_tastvg_pipeline_cpu_public_v1.py results/tastvg_pipeline_diagnosis/2026-10-01/P1` (or P5), needing only NumPy. Public results exclude captions, identities, media, GT coordinates, raw tubes, expert proposals/features, model states and weights.
