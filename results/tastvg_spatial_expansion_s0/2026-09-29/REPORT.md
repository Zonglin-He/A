# S0: Sa2VA-guided spatial hypothesis expansion

**Result and decision:** on corruption, native six-layer oracle gain is +0.6965pp and RVOS trajectory oracle gain is +0.8993pp (source-bootstrap95% CI[+0.5453,+1.3098]). Expansion minus six-layer is only +0.2028pp, CI[−0.5857,+0.9146]. Clean expansion is +0.8949pp. This establishes a small candidate increment in this configuration, not the requested large 5–10pp opening of spatial support. Do not launch S1 selector/OPD from this result. Temporal remains frozen at its existing critic-reranking design.

Completed / independently audited. This is an offline candidate-support diagnostic, not a learned or deployed selector. Temporal research remains native-candidate + UniversalVTG reranking; no new temporal run or OPD rescue.

| Condition | Native sIoU % | Native six-layer oracle % | RVOS trajectory oracle % | Expansion gain pp | 95% source CI pp | Expansion minus six-layer pp |
|---|---:|---:|---:|---:|---|---:|
| corruption | 45.3509 | 46.0474 | 46.2502 | 0.8993 | [0.5453, 1.3098] | 0.2028 |
| clean | 45.8489 | 46.6194 | 46.7438 | 0.8949 | [0.5310, 1.3202] | 0.1244 |
| frame_drop_5 | 44.9906 | 45.7052 | 45.9716 | 0.9810 | [0.6101, 1.3974] | 0.2664 |
| frame_freeze_5 | 45.7540 | 46.4941 | 46.6048 | 0.8508 | [0.4867, 1.2837] | 0.1107 |
| motion_blur_5 | 45.5072 | 46.2024 | 46.4109 | 0.9036 | [0.5371, 1.3195] | 0.2085 |
| occlusion_5 | 45.2275 | 45.9171 | 46.1008 | 0.8734 | [0.5326, 1.2583] | 0.1837 |
| exposure_5 | 45.2751 | 45.9181 | 46.1629 | 0.8878 | [0.5346, 1.3009] | 0.2448 |

## Configuration and denominator

Original C3 first16 historically exposed VidSTG target-test sources, one query/source. Clean + five existing source-hash seed0 5% transient corruptions:96 cells. Corruption primary averages five conditions within source, then16 sources; paired10000-source bootstrap. Clean is separate. This is development evidence. The historical +0.705pp six-layer result used different conditions; only the matched comparator above is used.

Official Sa2VA-4B revision3fee777d49ee9276eac51ea3e5f9b69e81d09be6, BF16/eager, greedy256-token cap. Five uniformly sampled observed frames per video, original caption prompt, first SEG mask only. Sparse nonempty mask-derived boxes guide checkpoint-weighted L1/GIoU on observed positions. Exactly3 normalized H_app-only steps of .004 original combined visual-H norm, total<=.012. Text/motion/weights remain frozen; official detach and dynamic native routing retained. Native B0 plus B1/B2/B3 retained, no backtracking/GT-guided optimization or intermediate stopping.

All96 trajectories sealed before16 previous GT keys loaded. The oracle selects one entire tube among four, then evaluates sIoU over all GT-valid sampled frames, not frame-wise GT mixture or temporal intersection. Direct sparse expert quality has a different observed-frame denominator. Cached suffix reuses exact native H; no backbone per gradient step.

## Evidence and trajectory diagnostics

| Group | Nonempty cells | Valid evidence frames / sampled | Unique expert calls | Best step counts 0/1/2/3 | Step3 gain pp | Step3 harms >5pp |
|---|---:|---:|---:|---|---:|---:|
| corruption | 72/80 | 325/400 | 15 | 14/10/5/51 | 0.4344 | 0 |
| clean | 15/16 | 66/80 | 16 | 3/2/1/10 | 0.4181 | 0 |

corruption: oracle gain outside the five observed positions 0.3654pp; on observed valid positions 11.9481pp. Direct RVOS evidence sIoU 71.6326% versus native 45.5328% on the same nonempty-evidence/GT-valid positions. Oracle union with six-layer candidates adds 0.6920pp over six-layer. s-oracle-selected native tube vIoU change 0.2840pp with its dynamic temporal output; this is diagnostic, not final temporal reranking.

clean: oracle gain outside the five observed positions 0.3439pp; on observed valid positions 12.1516pp. Direct RVOS evidence sIoU 71.6167% versus native 45.5335% on the same nonempty-evidence/GT-valid positions. Oracle union with six-layer candidates adds 0.6575pp over six-layer. s-oracle-selected native tube vIoU change 0.2857pp with its dynamic temporal output; this is diagnostic, not final temporal reranking.

Alignment loss decreases from B0 to B3 in 86/96 cells. Zero oracle harm is structural because B0 is retained; it does not prove online selection safety. Individual-step scores and negatives remain in ROWS.json. No clean-specific tuning, severity adjustment, additional seeds, selector, spatial OPD or online state was run.

## Verification and resources

GPU-process time 276.76s, including load/setup failures; stage totals {'capture': 116.19700804899912, 'expert': 83.50343506698846, 'reinsertion_audit': 9.641692067991244, 'ta': 67.42108196401387}. New expert calls 31; exact sparse-observation cache reuses 65.288 native suffix gradient steps. Model weight download is separate CPU/network setup. AUDIT.json records exact native replay, whole-pipeline reinsertion, independent loss/mask/dual task metrics; PUBLIC_AUDIT.json independently reconstructs scalar oracles and bootstrap summaries. Three CPU tests cover mask geometry, empty evidence and independent loss/gradient.

No production method changes. S1 is a possible next experiment only; S0 oracle support cannot establish critic ranking, OPD transfer, online-TTA gains or a publishable end-to-end method.

[Official expert model](https://huggingface.co/ByteDance/Sa2VA-4B), [official implementation](https://github.com/bytedance/Sa2VA).

## Matched component readout and retained cases

Post-hoc analysis uses the already sealed predictions, with no new inference or tuning. On the same12 sources with both observed and unobserved GT-valid frames, corruption s-oracle gain is +11.9481pp on the five observed positions versus +0.3145pp outside them. The paired gap is +11.6336pp, CI[+5.0596,+21.1175]. The earlier all16-source unobserved result is +0.3654pp; it has a different denominator. Thus the measured change is concentrated at observed positions. This suggests limited spread from sparse evidence to the whole tube in this objective/interface/budget, without isolating which one is responsible.

On the same11 sources with nonempty expert masks and GT-valid observations, the expert scores71.6326% versus native45.5328%; paired difference+26.0998pp, CI[+4.6360,+48.7977], across130 repeated frame-condition observations. This is not an all16-source/full-tube expert score. The expert is informative on average in this subset, yet can be wrong on individual sources.

Corruption oracle improves14/16 source averages. Q09 and Q12 gain+3.0132/+2.2152pp; they are retained positive cases. Q03 and Q06 have oracle gains+1.0676/+0.5002pp but step3 changes−2.4869/−2.4656pp; their early/native candidates protect the offline oracle, not a deployed selector. Step3 source averages improve11, worsen3 and are neutral2 using0.1pp tolerance. The combined six-layer+trajectory union oracle gains+1.3886pp over native, still far below a large support expansion. COMPONENTS.json retains all source-level values.

The result does not identify sparse sampling, alignment objective, radius/step budget or representation interface as a unique cause. No follow-up grid, dense-expert rerun or selector was started. The current configuration is insufficient grounds for S1; this is not a universal RVOS or spatial-OPD rejection.

The four full reinsertion checks include two original empty-evidence cases and two genuinely edited cases; all are exact. The latter two required6 additional replay gradient calls beyond the288 primary trajectory calls. Total GPU-process wall time is276.7632s (4.61min), including model loading, initial expert smoke and verification. The one-time official15.16GB weight preparation took77.84min of resumed download wall time and reused earlier partial bytes; this is separate from GPU experiment time.
