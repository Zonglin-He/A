# Sequential parameter tuning and expanded confirmation

Actual scope: 64 parent sources, 32 per direction. The formerly scored32 are now explicitly development; the other32 were excluded from all coordinate selection. Both cohorts have historical project exposure. This is not Table1 or an untouched full-test estimate.

## Failure diagnosis

28 usable temporal inputs reproduced exactly from frozen features. The original final regularized preference objective exceeded its initial value in19 cases (HC11, Vid8). On the exposed32, Projection-minus-Native mean vIoU was +0.585524/+0.041443pp, while temporal adaptation-minus-Projection was −2.608173/−0.277270pp. HC06 preferred a lower-quality contracted candidate and then decoded an even worse interval; HC14 had both a worse Projection prior and further adaptation harm. These support separate feedback-ranking and update-amplitude explanations, not a proof that one parameter explains every failure.

Four sources improved training pseudo-reference agreement but lost full-tube performance: HC02, HC14, V06, V12. Teacher agreement is not a correctness guarantee.

## Sequential search

Every stage inherits the previous selected values, per dataset; all other factors stay fixed. Each stage seals all32 predictions before scoring. Selection uses development GT only. All search points, negative and neutral results are exported. Finite grid winners are not global optima. Spatial fitting always starts from the source checkpoint; exact config/source/reference/seed reuse is separately counted.

|Stage|Dataset|Parameter|Selected value|Development vIoU (%)|
|---|---|---|---:|---:|
|01_lr|hc2|lr|0.005|30.799196|
|01_lr|vidstg|lr|0.0075|9.950818|
|02_sigma|hc2|sigma|0.0125|30.803494|
|02_sigma|vidstg|sigma|0.2|10.061378|
|03_tau|hc2|tau|0.2|30.806679|
|03_tau|vidstg|tau|1|10.067195|
|04_steps|hc2|steps|5|30.824940|
|04_steps|vidstg|steps|5|10.067195|
|05_lr_refine|hc2|lr|0.0035355339|30.843203|
|05_lr_refine|vidstg|lr|0.0075|10.067195|
|T01_lr|hc2|lr|0|33.485816|
|T01_lr|vidstg|lr|0.0003|10.706815|
|T02_tau|hc2|tau_T|1|33.485816|
|T02_tau|vidstg|tau_T|0.5|10.811254|
|T03_steps|hc2|steps|5|33.485816|
|T03_steps|vidstg|steps|5|10.811254|
|T04_l2|hc2|l2|0.001|33.485816|
|T04_l2|vidstg|l2|0.001|10.811254|

Spatial stage scores use fixed original Temporal plus varied Spatial; temporal stage scores use the frozen final spatial winner. Both coordinate chains therefore select the same joint T+S endpoint; Temporal-only is an attribution control.

## Frozen final configuration

```json
{
  "spatial": {
    "hc2": {
      "lr": 0.003535533905932738,
      "sigma": 0.0125,
      "tau": 0.2,
      "steps": 5,
      "writeback": 0.0625,
      "samples": 32
    },
    "vidstg": {
      "lr": 0.0075,
      "sigma": 0.2,
      "tau": 1.0,
      "steps": 5,
      "writeback": 0.125,
      "samples": 32
    }
  },
  "temporal": {
    "hc2": {
      "lr": 0.0,
      "steps": 5,
      "tau_T": 1.0,
      "l2": 0.001
    },
    "vidstg": {
      "lr": 0.0003,
      "steps": 5,
      "tau_T": 0.5,
      "l2": 0.001
    }
  }
}
```

The original five-step QA output still fixes spatial observation positions in both cohorts. The selected temporal configuration governs the final interval. This deliberate fixed-evidence protocol does not claim references were regenerated under the tuned interval. A zero-lr winner makes the final-output head Projection-only, not learned temporal adaptation. The original QA head still supplies the fixed spatial observation clock. Once coordinate selection chooses zero lr, later temporal temperature/step/regularization sweeps are algebraically inactive for that domain; flat results do not prove these parameters are intrinsically irrelevant or that no positive-lr joint configuration could work. This is a finite single coordinate pass, not global optimization. Episodic writeback is inactive.

## Confirmation (16 parent sources per direction)

|Arm|HC vIoU (%)|VidSTG vIoU (%)|
|---|---:|---:|
|Frozen|18.611228|7.486117|
|Previous Temporal|23.040519|6.325392|
|Tuned Temporal|28.170651|7.450866|
|Previous Spatial|19.072535|8.349824|
|Previous T+S|23.785216|7.194655|
|Tuned Spatial|18.939728|8.377117|
|Previous T + tuned S|23.485611|7.228663|
|Tuned T+S|28.517853|8.310106|

|Dataset|Contrast|Mean delta (pp)|Worst source delta (pp)|Negative sources|
|---|---|---:|---|---:|
|hc2|T_selected − T|0.445092|-3.026984|1|
|hc2|T_original − T|0.744697|-3.036077|1|
|hc2|T_selected − T_original|-0.299604|-2.060620|7|
|hc2|selected − F|0.328500|-0.695439|2|
|hc2|T_tuned − T|5.130132|-8.548003|5|
|hc2|T_tuned_selected − T_original|4.732638|-8.548003|6|
|hc2|T_tuned_selected − F|9.906626|-2.297386|2|
|hc2|T_tuned_selected − T_tuned|0.347202|-2.901209|1|
|vidstg|T_selected − T|0.903271|-4.874393|2|
|vidstg|T_original − T|0.869263|-4.632241|2|
|vidstg|T_selected − T_original|0.034008|-0.344431|2|
|vidstg|selected − F|0.890999|-5.010406|2|
|vidstg|T_tuned − T|1.125474|-5.665512|3|
|vidstg|T_tuned_selected − T_original|1.115451|-5.777616|3|
|vidstg|T_tuned_selected − F|0.823988|-3.799787|4|
|vidstg|T_tuned_selected − T_tuned|0.859240|-5.314936|2|

## Actual confirmation workload

Temporal teacher calls: 240; spatial teacher calls: 128; local proposal windows: 128; sources with selected references: 22/32, anchors: 50. All counts and measured component costs are exported in CONFIRMATION_COST_COUNTS.json. These are component measurements, not an end-to-end speedup against full-video YOLO/SAM2.

## Execution and limits

Actual new spatial fits/full Student redecodes: development 414; confirmation 44. Development exact config/source/reference/seed reuse records: 115; no-reference no-update records: 207. Shared temporal-head trials reuse exact frozen backbone features and saved teacher feedback, and are not counted as new full-backbone or MLLM inference. Confirmation reuses source captures only after source/pixel/state checks; fresh teacher calls and input generation are separately recorded. No new detector/segmenter, GT Oracle fit, CURRENT promotion, old queue resume or Table1 launch.

The first launcher failed its own cgroup-name assertion before any GPU fit; a separate002 supervisor fixed initialization ordering. Original capture/source/runtime were retained. Synthetic contracts are not scientific-effect evidence.

Selection bias remains in development scores. Confirmation is not used to restart tuning in this round. Bootstrap is omitted by the latest user instruction; mean, all paired source differences and negative tails are reported. Small parent samples and historical exposure limit generalization claims. Negative outcomes remain in the public rows.

## Final confirmation interpretation and engineering correction

On the32 confirmation sources, tuned joint vIoU is28.517853% HC and8.310106% VidSTG, versus Frozen18.611228% and7.486117%. The full gains are+9.906626/+0.823988pp. Spatial added to the tuned temporal output gives+0.347202/+0.859240pp, with8/1/7 and3/2/11 improved/degraded/unchanged parents. VidSTG benefit is concentrated: its best Spatial parent contributes+21.504637pp, while the worst loses5.314936pp; the positive mean is not evidence of uniform gains.

Tuned Spatial is not uniformly better than the prior spatial configuration on confirmation: with identical old Temporal, HC changes by−0.299604pp and Vid by+0.034008pp. The large HC improvement of the complete method principally follows the final-output Projection-only temporal choice. Vid tuned temporal-only remains0.035252pp below Frozen, and tuned joint is0.067011pp below tuned Spatial-only. These are observed limitations, not new selection gates. The confirmation result does not change the frozen per-domain recipes.

Candidate generation originally stopped after21 complete sources because V22 had one Source box with y2 slightly beyond the image. A separate Source-only tracker ROI intersection revision retained the raw source prediction and scoring coordinates. All128 anchor ROIs were checked; only one required intersection. The original21 complete proposals and280 existing input files matched exactly after regeneration, including arrays for compressed masks; all32 canonical/video/render input readbacks passed. The first005 resumption failed before candidate execution due to an immutable receipt-name collision with archived originals;006 used a separate output directory and all subsequent phases exited successfully. Both failures and all partial inputs remain preserved. No temporal re-inference was required.

Actual workload totals are458 new spatial fits/full Student redecodes (414 development+44 confirmation),115 exact reuse records and588 development shared-head trials (including253 zero-learning-rate trials with0 optimizer steps). Confirmation used240 temporal and128 spatial frozen-MLLM calls,128 windows,1006 appearance mask track candidates,50 selected reference anchors across22 sources;10 sources had no accepted non-Source reference. Component timers exclude failed-prefix regeneration and are not whole-run latency.

All arithmetic/root passes establish saved evidence and calculation consistency. They do not establish correct target identity, statistically stable improvement or full-test generalization. Bootstrap is not executed.
