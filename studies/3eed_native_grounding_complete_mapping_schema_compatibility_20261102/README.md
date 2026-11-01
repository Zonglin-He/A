# R37/R38 — Native grounding, complete mapping, schema compatibility

Read REPORT.md and original R37/R38 decisions plus COMBINED_DECISION. R37 is
the preregistered strict direct-G pipeline. R38 is a separately sealed engineering
parser compatibility correction of the same416cached answers, not a new method,
TTA, new model call or groundtruth-selected rule. Original R37 retained exactly.

Strict point22.705713 and compatible22.740349 mIoU vs source19.663563 and
legacyG24.727373. Parsing recovers62 unchanged unique locations but only+.034635pp
net; recovered subset6improved9worsened47unchanged. Coverage is not accuracy.
All145 previously accepted point values and208SoM predictions preserved;1badJSON
remainsUNKNOWN. Compatible rule applies to all416, permits only dict/singleton
object list and optional inert stringlabel; multiple locations/objects/duplicate
keys/nonfinite/bools/executable payloads rejected, no first-of-multiple choice.

Masks/geometry/source256boxes/scores/representatives fixed. Full mapping extends
only lookup dictionary: every old same-mask group equals full group. Point uses
actualgrid*14 absolute resized pixels, inverse then floor membership in exactly
one independent mask. Unknowns fall back to source, no nearest rescue, no newbox.
Point sees1image,SoM2images, so interface comparisons are not equal-cost/pure
output-format causality. Source ROI/point evidence is not identity ground truth.

Independent R37 audit rederived208rawscene geometry matrices exactly,7arms1456
IoUs and617negative comparisons. Independent R38 audit reuses thatgeometryhash,
416rawhashes unchanged,11arms2288IoUs and1419negatives reproduce. Duplicated arms
are not independent observations. Full task scalars/status/negative tails retained.
Same presealed24identity annotations joined after R38predictionseal, no relabeling
or GT feedback filter. See identity summary for bounded provisionalreview limits.

Old loader0call failure, unit2control1/2failed, new8diagnostic1/8capability, root
pre-dataset gate reassessment, documentary-reference snapshot repair and audit
technical corrections retained. Software pass is separate from synthetic hitrate.
416dataset+10synthetic=426ledger includes24reused preflight, no hidden retries.
Original model files hashed, runtime frozen receipts reviewed; CPUaudit did not
inspect GPU parameter bytes. Exact native-unit interpretation remains qualified.

AllA208 is exposed development with2Drone/3Quad scenes, not new confirmation.
No U/T/FINAL work, no automaticTTA. No actualpoints/nativeanswers/labels/captions,
mask/3Dcoordinates/opaqueIDs/rawscenes/paths/media/features/weights or upstream
notebook snapshots exported. Explicit field allowlists, not generic PRIVATE-key
removal, protect recovery/identity/readout records. Synthetic images omitted;
own code, aggregate controls and official source links retained. Private licensed
inputs/prior source dependencies needed to reproduce sealed experiments.
