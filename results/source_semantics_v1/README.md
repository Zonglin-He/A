# Source naming correction: official native readout versus added MBR

The official TA-STVG implementation does not use MBR. `models/post_processor.py` forms legal start/end log-probabilities and takes a flattened argmax; `engine/evaluate.py` evaluates the two native temporal offsets and takes the physical hull of their MAP intervals. Our `stvg/readout.py` preserves that native decision for observation selection. Our `stvg/predictor.py` then constructs a physical-hull posterior and calls our `final_mbr`; previous stored reports often named this added-readout baseline simply `Source`.

**That internal `Source` alias means Source+MBR, not an untouched official source.** Changing the temporal decision rule changes the inference method even though network weights remain frozen. Earlier wording such as “Original Source already uses MBR” or “original MBR” conflated the historical local baseline with the official method and is corrected here. This is our pipeline/reporting ambiguity, not evidence that the official authors named MBR as Source.

Official references:
- https://github.com/HengLan/TA-STVG/blob/main/models/post_processor.py (legal endpoint argmax)
- https://github.com/HengLan/TA-STVG/blob/main/engine/evaluate.py (offset0/1 plus min-start/max-end hull)

The local copies and our decoding files are SHA-bound in `comparison.json`. Official native readout here means the official **decoding rule on the same saved local source outputs**. Our resolution/precision/cross-domain evaluation protocol must still be stated; these numbers are not claims of reproducing the paper's published in-domain benchmark accuracy.

## Three rows that must stay separate

| Display name | Definition | What comparison measures |
|---|---|---|
| Source (original native readout) | Frozen source boxes, per-offset native MAP and physical hull | Original temporal decision under this local source protocol |
| Source + MBR | SAME boxes/logits, our physical-hull posterior + our MBR | Added temporal decision-rule effect |
| Full | Our spatial/temporal feedback plus MBR | Overall system gain versus Source; incremental feedback gain versus Source+MBR |

Full−Source includes MBR and feedback effects; it is not all evidence of new expert/visual/TTA benefit. Full−Source+MBR is the matched-readout feedback comparison; it is not Full versus untouched Source. Both comparisons should be reported. No trained network changed does not justify omitting a changed decoder from the method description.

## Existing complete TA results, relabeled without recomputation

These are already sealed/scored full1,902HC/10,303Vid results from the fourteen-arm conditional control. `SourceBox_P0Native` is native per-offset MAP-hull on saved source boxes; `SourceBox_P0MBR` is added MBR on the same boxes; `FullBox_Soft` matches the existing actual full result. No video, model, new annotation scoring or new tuning occurred.

| TA evaluation | Readout/system | tIoU (%) | vIoU (%) |
|---|---|---:|---:|
| HC1,902 | Source (native MAP-hull) | 40.141475 | 20.704678 |
| HC1,902 | Source + MBR | 48.647345 | 24.994752 |
| HC1,902 | Full | 49.576842 | 27.218099 |
| Vid10,303 | Source (native MAP-hull) | 31.728454 | 11.138484 |
| Vid10,303 | Source + MBR | 33.804566 | 11.777655 |
| Vid10,303 | Full | 34.604378 | 14.383314 |

Added MBR alone accounts for HC+8.505870ppt/+4.290074ppv and Vid+2.076112ppt/+.639171ppv. Full−Source+MBR remains HC+.929497ppt/+2.223347ppv and Vid+.799812ppt/+2.605659ppv. Overall Full−nativeSource is HC+9.435367ppt/+6.513421ppv and Vid+2.875924ppt/+3.244830ppv, explicitly including the readout effect. Raw precision, input report hashes and exact differences are in `comparison.json` and `comparison.tsv`; `studies/source_semantics_v1/export.py` reproduces this view from existing scored scalar reports only.

## Frozen artifacts and current CG full run

No historical prediction, score, acceptance value, raw column key or frozen scientific plan is overwritten. Historical `Source` keys remain intact with an explicit semantic mapping. Historical Source+.9 pass/fail records refer to the stored MBR control; they do not become claims about native source or get retrospectively relabeled passed.

The active CG full plan already saves both `SourceNative` (official CG MAP) and `Source` (our Source+MBR) on identical source boxes. Its frozen internal keys cannot be silently renamed while running. Public display must instead map SourceNative→Source(original), Source→Source+MBR, Full→Full. The extra report-only adapter `studies/source_semantics_v1/cg_full_reporting.py --dataset hc2|vidstg` gates on full domain seal+successful score, reads already scored metric rows, and produces all three paired comparisons, signed outcomes, severe harm against ORIGINAL Source, parent CIs and duration strata under `named_source_comparison/`. No new GT scoring, model inference or query/parameter selection. That adapter has been syntax checked and rejected an unfinished domain as intended; current full data are not yet scored, so its final report is not claimed complete.

For each domain result notification, present BOTH Full−originalSource and Full−Source+MBR with clear baseline labels. The new Source+MBR label is a transparent stronger/control baseline, not a replacement for original Source. The CG inference worker, profiles and all paid acquisitions continue unchanged.
