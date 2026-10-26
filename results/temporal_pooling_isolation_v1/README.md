# Temporal appearance pooling region: full-domain isolation

The latest direct user request is completed using the exact existing full-domain strict control, plus a new independent all-query numerical-chain verification. Existing AuxSourceROI and selected FullMAP are the requested Source-box and corrected-box appearance arms; BOTH retain the same MAIN spatial candidate geometry. No Source/expert/model/optimizer/GT scoring was repeated. Four actual raw grids additionally verify both ROI pools with the same function and context. Full replay uses the audited sufficient features.

Only the temporal pooling region differs. Same eight expert observations, native Source feature grid, 1.5 context, centering/normalization, one all-video prototype pair, kappa0/beta1 and one posterior update. Each arm selects its own event interval and runs the unchanged final DP on that interval. Full-span candidate boxes, raw unary affinities, box-IoU links and objective coefficients are fixed; interval-dependent eligibility and the original observed/unobserved/edge normalization are recomputed by the same code for each own interval. Same candidate construction does not imply identical final paths after the event interval changes.

|Full target set|Temporal appearance ROI|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---|---:|---:|---:|---:|
|hc2 (1902 queries / 227 parents)|SourceBoxAppearance|49.689566|27.315220|41.114616|16.824395|
|hc2 (1902 queries / 227 parents)|CorrectedBoxAppearance|49.764548|27.349044|41.219769|16.876972|
|vidstg (10303 queries / 732 parents)|SourceBoxAppearance|34.908528|14.552160|19.392410|8.385907|
|vidstg (10303 queries / 732 parents)|CorrectedBoxAppearance|34.918192|14.558151|19.411822|8.463554|

Corrected ROI minus Source ROI, percentage points; query-equal differences with parent-video cluster10000bootstrap pointwise95CI:

|Target set|Delta tIoU pp [95% CI]|Delta vIoU pp [95% CI]|Severe t/v/union once|Rescue union|
|---|---:|---:|---|---:|
|hc2|+0.074982 [-0.094384, +0.236031]|+0.033824 [-0.046807, +0.114818]|88/27/88|86|
|vidstg|+0.009665 [-0.116164, +0.132882]|+0.005991 [-0.038913, +0.051228]|748/188/751|754|

Both full target sets have zero-crossing tIoU and vIoU confidence intervals. The evidence does not establish a stable average temporal-localization benefit specifically from shared correction of the pooling boxes. It is not an equivalence test or proof of zero effect. The supported mechanism claim is limited to object-region appearance supporting temporal correction; a separate causal benefit of corrected ROI is not established. No manuscript/default/parameter edits are made, and no new configurations are selected from these exposed full target sets.

All signed query-level results, duration groups, severe losses/rescues, interval/path change counts, internal fixed-support sIoU/boundary checks and actual cached replay costs remain in the two domain reports. sIoU is intentionally outside the compact table. Inherited Source/teacher costs are nonzero; new cached verification wall is neither process CPU time nor deployment latency.

The first new preflight falsely detected geometry mutation by hashing a Python dictionary as NumPy object-pointer bytes. The preserved science/numerical code is unchanged; independently frozen runtime_v2 canonicalizes only metadata hashing. Four raw-grid cases and both full-domain verifications use this version. Failed-attempt elapsed time was not recorded, rather than fabricated as zero.

Current CG and gamma owners were not signaled or changed. EdgeTAM remains the external mature propagation comparison, and every corruption task remains paused by direct user instruction.
