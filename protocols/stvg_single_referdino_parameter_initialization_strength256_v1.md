# Supplement: initialization correction strength, same expert/evidence

Following the user's request that the root autonomously analyzes parameter effects and chooses promising tuning priorities, retain the original optimizer coarse/fine experiment unchanged and add a finite initialization-strength axis. This adds no expert, evidence, network or training head.

The inherited global consensus correction is scaled by gamma in (cx,cy,logw,logh): E_gamma = decode(encode(Source)+gamma*delta). Gamma0 is Source exactly; gamma1 is the frozen E_Ref exactly. The same8 ReferDINO boxes and consensus assignment are retained. This differs from output alpha, which scales the post-adaptation residual around the chosen initialization.

Screen gamma0/0.25/0.5/0.75/1/1.25/1.5 on both original128-parent panels. All256×7 predictions are sealed and independently reconstructed before offline GT scoring. The first inline audit used float32 logs against a float64 consensus and was corrected to explicit float64 before GT; no prediction was changed. All controls and signed parent outcomes remain. Choose one gamma per dataset by whole-panel mean complete vIoU, ties closest1. This is development calibration, not an independent confirmation.

A favorable initialization-only readout does not substitute for an adapted method result. Any nonunit winning gamma must be tested through actual current-state decoder OPD, paired to the existing tuned gamma1 configuration. A subsequent finite execution plan is locked before those new adaptation runs and their scoring, using the same overall7200s GPU wall limit. Existing optimizer coarse/fine predictions and all original results remain untouched; no online GT, per-video gamma/alpha choice, expert replacement or new observation position.
