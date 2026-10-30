# Root qualitative review of fixed post-seal cases

The coordinator actually opened six local images using view_image after prediction seals and scoring. They were selected by fixed signed-tail rules and were not used to change methods. Images, raw targets, coordinates and per-frame activity are private and excluded from publication.

AVIS developer duet case: both guitar and violin are visible, GT and source retain both; competitive audio gate deletes guitar, independent reference restores it. The plotted class evidence alternates under class competition even while the independent evidence remains high. This directly illustrates the reference-rule failure, without claiming the still images prove acoustic timing.

AVIS confirmation worst-tail case: a three-person broadcast composition, right-side GT person and source mask agree in the displayed frame; both activity variants remove that correct source. The plot shows low person-vs-background evidence at the selected time. Audio-only class evidence is neither a calibrated absence cue nor same-class instance correspondence. No identity or speech-content inference is made from faces.

GREAT fixed positive Kettle/pour: GT spout response is largely present already. Full vsQ75 adds oneTP and removes5FP at.5, but a spurious handle response persists, and source has betterMAE thanfull. Full/HM-only maps are nearly identical in this case. This supports local coverage/probability change rather than discovery of a new function region.

GREAT fixed negative Kettle/pour: Q75 recovers more spout support than source; full removes2TP with unchangedFP40 and lowersAUC. Handle false responses remain. Both view plots use the sameXYZ and camera; interaction photo and3Dshape differ, so pixel-to-point claims are inappropriate.

3EED confirmation primary worst case: sourceIoU.86907, PCA.790, directlatent.816, constrained3.77890. Geometry accepts the case with uncalibrated modeposterior1, but rotates/changes an already-good box in the wrong direction. Optimizer constraints reduce possible damage without making that posterior correct.

3EED maximum PCA rescue: raw4966points, sampled16384withreplacement and4765unique. Source/direct/primaryIoU.26007, PCA.74947. The top view shows the missing source orientation and usefulPCA alignment, while the structuralteacher rejectsgeometry. This is evidence that current qualification is overly restrictive in some helpful cases, not proof that all sparse geometry is uninformative.

These cases accompany all signed scalar outcomes; they do not substitute for the fixed full-cohort metrics or turn posthoc observations into independent confirmation.
