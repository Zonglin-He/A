# One ReferDINO: initialization + OPD, both original128-parent panels

The user explicitly paused parameter tuning and selected a one-expert design. Freeze native ReferDINO ryt_mevis_swinb, use its same eight selected-track boxes for spatial initialization E_Ref and state-only Live OPD. Remove the independent GroundingDINO detector, four image calls, admission rule and cache dependency. Strong remains TA-STVG-only and unchanged. ReferDINO's internal architecture is preserved; skip the redundant independent GroundingDINO-B checkpoint pre-load and load its complete ReferDINO state instead. All learned state must be covered. Verify new native outputs against the saved native outputs, with independent detector checkpoint access denied.

Use the original HC128 and Vid128 parent/query rosters, seed41. Source checkpoints remain VidSTG→HC2 and HC2→VidSTG. No new expert/model weights or network training. HC40 steps, Vid30 steps, Adam0.01, tau0.2, scales(0.05,0.25), M32, alpha1. Vid30 is the highest mean of the completed old-E nine-grid; the later lr/tau search was paused before any new parameter trial. It is not a proven optimizer optimum, and was selected with a different initialization. No retuning in this round.

E_Ref uses the existing robust global four-dimensional correction estimator, replacing its inputs with eight singleton ReferDINO boxes at the same physical positions. Inherit the numerical Source-domain scale/truncation from the existing frozen configuration; do not claim a new ReferDINO calibration. Singleton relative candidate-score cost is zero, so there is no absolute-confidence weighting. Robust geometric consistency may reject initialization frames; OPD retains the same eight-reference supervision as before. Record initialization support and the exact shared evidence hash.

Eight positions are reconstructed from the original Source Native index extent using the original physical-bin midpoint seeds and farthest-uncovered expansion. This geometric rule does not run a detector and must match the original locked eight positions on the panel. No unresolved-wh skip rule. The experiment may reuse existing Source decoder features, Source/Strong posteriors and native ReferDINO outputs after provenance/equality checks; sanitize the cache to contain no old E, GroundingDINO observations or admission state. Missing Vid Source features may be regenerated with the same frozen Source224. No new Source448/expert pass is needed for the cached comparison. Two prelocked first-query complete-chain profiles do execute Source224/448 and a single ReferDINO8 forward to verify the independent runnable path and charge real costs; their predictions may serve as primary first-query predictions after exact parity.

Main five rows, same frames/MBR/metric denominators:
Source = Source boxes + Native time;
Spatial_only = E_Ref+Live + Native time;
Temporal_only = Source boxes + Strong;
Full = E_Ref+Live + Strong;
Without_spatial_adaptation = E_Ref + Strong.

Also compare saved old two-expert E+Strong and Full as historical controls, never as inputs to the new predictor. New scores must not be relabeled old two-expert results. All256 primary predictions sealed before new GT scoring. All no-update, zero-overlap, harmful and successful parents retained. Parent paired bootstrap10000; all signed query results, fixed actual/unobserved sIoU, tIoU, full vIoU, tails and costs. These are exposed development panels, not independent confirmation.

Shared gpu.lock, disk>=30GiB, host available>=6GiB; bounded GPU worker wall per direction<=3600s. Only temporary8×256 states are optimized and reset independently for each query, network weights unchanged. No automatic further tuning. New formal implementation/config declares one ReferDINO; preserve historical predictions and CURRENT deployment. Research registration may point to the explicit new evaluated single-expert design with limitations, without inventing benefits or promoting CURRENT/MAIN/PAPER.
