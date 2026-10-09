# Figure 1 presentation revision: correct event does not guarantee correct instance

The human requested a motivation figure with E3M Figure 1's continuous filmstrip
style and a right panel tied to the current method's design. This presentation
revision reuses the complete, sealed v6 cross-domain native cohort and all four
original error categories. It changes no checkpoint, roster, input, output,
metric or threshold. It performs no GPU inference, optimizer update, expert
call or active OPD P1 payload access.

The original claim that T-correct/S-wrong dominates T-wrong/S-correct is not
supported: original category percentages are respectively 14.844/34.375,
13.281/45.313, 16.406/22.656, and 12.500/24.219. The new panel cannot be used
to assert that temporal grounding is generally reliable or spatial errors
dominate all errors. Full original quadrants and the registered contrast stay
in the result export, alongside this expressly post-hoc conditional view.

The supported, narrower observation is that correctly localized events still
contain spatial failures. Among predictions with tIoU > .5, spatial failure
(mean frame IoU <= .5) rates are:

| Source → target | Model | Spatial failures / temporal successes | Percent |
|---|---|---:|---:|
| VidSTG → HC2 validation | TA-STVG | 19 / 50 | 38.0 |
| VidSTG → HC2 validation | TubeDETR | 17 / 39 | 43.5897 |
| HC2 → VidSTG test | TA-STVG | 21 / 38 | 55.2632 |
| HC2 → VidSTG test | TubeDETR | 16 / 46 | 34.7826 |

The denominator is displayed for every bar. Each full cohort has 128 historical
parent sources, one query per parent, paired across the two source-trained
backbones. All 512 native outputs remain represented in the underlying data;
the conditional graph explicitly focuses on temporal successes. Spatial IoU
uses all legal GT frames and zero for uncovered support, independent of the
predicted interval. Strict .5 thresholds are unchanged. 10000 paired parent
bootstrap draws quantify this descriptive conditional rate. Different native
preprocessing/training prevents a pure architecture interpretation. This is
not a fresh independent test or a full-benchmark prevalence estimate.

Panel A retains the already audited common HC2-source → VidSTG illustrative
case from v6: a child behind another child. Five evenly spaced actual sampled
frames within both native intervals and the annotated event are shown as three
filmstrips: GT, TA-STVG, TubeDETR. Box coordinates and pixels are unmodified;
the black film rails and perforations are decoration. Physical-time bars retain
the original intervals, including endpoints outside the sampled-frame hull.
Case selection is qualitative and post-statistics, not a frequency estimate.

This observation motivates a spatial correction channel at test time, with
additional visual feedback rather than confidence in event timing alone. The
registered OPD preserves Native WHEN and updates spatial output via a single
frozen visual expert's detached feedback and an on-policy Gaussian likelihood.
This figure establishes the need for spatial correction, not the efficacy or
causal necessity of that particular feedback/optimizer. Direct, shuffled,
fixed-rollout and full mechanism comparisons belong to the separately locked
P2 study. P0's adverse examples and failed HC2 stability gate remain preserved.

Suggested caption: **Cross-domain event localization does not ensure instance
localization.** (a) The same query and five actual frames illustrate instance
ambiguity in two source-trained STVG backbones; rows show GT and native outputs.
(b) Even among temporally correct predictions, spatial failures persist across
both domain directions. Bars show conditional rates, exact denominators and
95% paired parent-bootstrap intervals on the historical 128-parent panels.

The stylistic reference is the authors' Figure 1 in [E3M, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11011.pdf).
No E3M pixels, boxes, query or measured result is copied into this figure.
Private dataset RGB/query/GT overlays stay local. Public export consists of
code, protocol, anonymous native scalars, original full quadrants, conditional
statistics and the statistics-only panel. Whole P1 and the paper are not complete.
