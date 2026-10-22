# Native temporal motion-versus-appearance evidence: HC64

Completed negative; not adopted. Same repeatedly exposed64-parent development set,
all new predictions sealed before new scoring. Space/expert8 fixed, full still paused.

Text interventions failed event ranking and disturbed target geometry. This branch
therefore keeps FULL text fixed and changes original native pre-fusion visual tensors.
MotionNull replaces the projected3D-encoder tensor by its temporal mean at each
spatial/channel position. AppearanceNull applies the same operation to the2D tensor.
Mean is computedFP32 then cast to originalFP16. Original crossmodal fusion and exact
native two-pass FP32 replay execute at both offsets; original sixlayer hull/MBR/DP.

Primary q0*qAppearanceNull/qMotionNull tests dependence on motion variation beyond
appearance variation. Both individual ratios, direct null outputs, sharpening and
reversed primary placement are retained. No weights/thresholds/query updates.
Native3D features are NOT assumed pure action; a temporal-mean tensor is NOT claimed
a realistic static video or calibrated identity-only negative. D-TSG and CAEM learn
debiased matching during training; this is only a frozen intervention feasibility test.

Percent metrics, time improved/harmed/tied vsNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|SelectiveMotion|27.2510|50.7391|29/34/1|30|
|MotionContrast|27.8709|52.0569|26/34/4|27|
|AppearanceContrast|24.1863|44.0186|12/48/4|40|
|MotionNull|27.6298|52.0964|22/39/3|24|
|AppearanceNull|27.4646|51.9684|31/31/2|26|
|Sharpen|28.7258|53.7518|16/38/10|20|
|ReverseSelective|24.4551|45.5324|21/39/4|34|
|Source|26.8938|54.8230|0/0/64|0|

Primary−G6 v−2.4442ppCI[-5.5319,.6159],t−5.2222pp[-11.3818,.8987];29timewins,
34harms/1tie,30time losses>5pp. Shortt−7.4591pp,long−4.6520pp. No Source or Global6
gain. Both directly nulled models also lose~3.9pp tIoU: native prediction uses both
pathways, and dependency alone does not validate a correction signal.

Same-length/near-native-prior721,976-pair macro feedback accuracy52.8146%
CI[48.9957,56.5576], native52.1362%; no demonstrated reliable incremental ranking.
Ratio feedback also amplifies native-low-support hypotheses: mean posterior mass on
the lowest native-log-probability quartile rises from.000912 to.042626, overallTV.5786.
This measured tail amplification and null-domain ambiguity argue against just tuning
ratio power or treating an ablated model as an event-absence likelihood.

All64 full pre-fusion replays restore original hidden/posterior bitwise.1152 independent
official/DenseTube checks max9.99e−16. All48+434 frozen source hashes intact. Actual384
fusion-offset calls/192 native replay queries,0RGB/visual/text/expert forwards and0
optimizer steps. Worker46.85s, including model startup; no end-to-end deployment claim.

Decision: reject this fixed motion/appearance ratio feedback, preserve all signed
positive/negative tails. Next explore valid overlapping temporal context views rather
than stronger ablation or another ratio scale; source-trained event outputs should be
tested on real feature sequences, with crop-boundary censoring handled explicitly.

[Signed outcomes](signed.tsv),[severe tails](severe_declines.tsv),
[information audit](information_audit.json),[protocol](protocol.json).
References:[D-TSG](https://arxiv.org/abs/2207.13457),[CAEM](https://github.com/CFM-MSG/CAEM_Code).
