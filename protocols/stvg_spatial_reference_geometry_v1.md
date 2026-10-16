# Spatial reference geometry: one finite decomposition

Authorized by the 2026-10-16 attachment following commit 093ca021. This is a research candidate, not a replacement for CURRENT.

## Question and competing explanations

Under identical Temporal intervals, spatial adaptation may harm localization because (a) a correct instance is truncated by GrabCut or largest-component bounding, (b) the selected region belongs to another instance, (c) the pool lacks the referent, or (d) a useful sparse reference propagates poorly through Student. Teacher agreement alone does not establish correctness. These explanations must be separated before a new fit.

## Fixed inputs and scope

TA-STVG source checkpoints, frozen STVG-R1-7B, original RGB/Query, four observation centers, proposal algorithm, Temporal-Lite shared-candidate/no-L2/full-grid-trust 512-parameter interface, and episodic 1792-parameter Spatial Lite remain fixed. HC learning rate .0045; VidSTG .027. No new model, tracking model, prompt search, LR sweep, Bootstrap, official full-set run, or CURRENT promotion.

Initial decomposition uses twelve already exposed development sources: per target domain, two positive, two negative and two unchanged Full-minus-Temporal cases from the sealed development128. Within each sign stratum prefer sources with actual spatial references, then sort by parent SHA and query-key SHA. This is a result-stratified diagnostic panel, not an unbiased effectiveness estimate. The confirmation64 are excluded from this selection. Preserve the other development cases and all negative/unknown outcomes.

## Stage 1: time interface qualification

Audit saved final-Lite Temporal response metadata and both Spatial observation clocks across development128 and confirmation64. Compare the installed Qwen temporal-axis formula to the isolated fractional-seconds correction at the actual model's 2 tokens/second. Use saved frame counts, FPS and original 2-frame temporal patches; exercise the real installed get_rope_index in synthetic CPU contracts. No model load or task scoring is needed for the audit.

Record unchanged positions, changed positions, collapsed positions and missing/unavailable responses separately. A changed request cannot reuse a legacy teacher response in a corrected-clock namespace. Historical protocol results stay intact; this audit does not measure the correction's task-performance effect. Subsequent geometry-only comparisons keep the same identity decision and physical clock within each comparison. A corrected-clock baseline, if needed, is a separately labeled interface control rather than a frame-selection improvement.

## Stage 2: intermediate geometry replay

Read exact historical candidate records/masks, verify source RGB and canonical-video hashes, and replay the pinned Selective Search/GrabCut function with only intermediate-return instrumentation. Strip the instrumentation and require exact original AST recovery. Preserve raw rectangles, complete foreground, largest component, boxes and proposal identity privately. Compare replayed candidate order, score, selected component mask and box to history. Nonexact replays are unknown historical intermediates; proposal_rank by itself is insufficient. A reconstructed rectangle is not a historically cached rectangle.

Seal every intermediate and its SHA before reading annotation geometry. Review positive, negative and unchanged cases visually. Identity judgments must use the original Query and scene evidence; ambiguous identity remains unknown. GT is restricted to postseal geometric diagnosis and final prediction scoring. Never use GT to generate candidates, choose a candidate, construct a reference, choose an update step, or fit an Oracle.

## Conditional modification and stop rule

Only if visual and geometric decomposition supports correct-instance truncation as the main failure, compare R0 current component bbox, R1 the same selected candidate's complete foreground bbox, and R2 the same candidate's Selective Search rectangle when exact full-pool replay qualifies it. Preserve candidate ID, physical observation times, training parameters and all acceptance rules. R1's old semantic approval does not validate the replacement boundaries.

If identity errors or missing pool support dominate, stop this geometry patch and record the evidence before committing to one targeted identity/evidence-source revision. Do not substitute extra gates, learning-rate searches or response-strength adjustments for this diagnosis. Missing historical intermediates may block a matched geometry comparison without showing that geometry cannot help.

After one evidence-qualified variant, perform a real complete Student T+S decode under fixed parameters, then lock the rule before confirmation. Do not pick a variant per query or per dataset using final GT. Existing confirmation exposure must be disclosed and does not become untouched by renaming the cohort.

## Completion and publication

Publish each meaningful completed stage promptly: implementation, actual configuration/provenance, anonymous full positive/negative/neutral/unknown results, measured costs, recovery attempts and limitations. Exclude media, annotation coordinates, model weights, tensors, actual Query/caption text and raw teacher responses. Independently verify seals, source/math, actual rendered chart pixels and remote contents; update RESEARCH_HISTORY with check, snapshot, check. Finite stage completion is distinct from final-method qualification.
