# 3EED bounded exploration closeout — 2026-11-17 09:00 CST

Exploration ended at 2026-11-17 09:00 Asia/Shanghai. Execution is closed; no new model, training or scoring was performed for this publication. FINAL untouched, production R95 untouched, no expert or new source training. This report records results, not a claim of stable TTA success.

## Decision
Stable six-direction expert-free TTA was NOT established. Reject R123 registered-point InfoNCE, R125 native semantic consensus fitting, and R126 full-readout acceptance as tested recipes. Preserve existing frozen/direct baselines and all negative observations. Do not select a favorable one-step/readout/scene posthoc.

## Useful update interface, but not a successful target signal
R124 source-only controlled diagnosis, SourceVAL384 (128/source), real existing point Adapter Up weight+bias 4896 parameters, original A256, same three 1%-original-Adapter-norm steps: zero33.800991%; GT1+1.440473pp, GT3+4.222581pp, random3+.095858pp. GT3 70 improvements/13 harms. These ephemeral source-GT edits were restored per sample, not deployed, not source retraining, not TTA and not evidence of target unlabeled supervision.

## Main actual target interventions
All six directions, exposed DEV768 + EXP1536, 154 protected aliases retain fallback/no feature loading. AB-active786/2304. Predictions sealed before task scoring; original A256 and coordinates unchanged. C excluded from construction/fitting, then used for final observational readout; same physical scans, not new scenes.

| Method | DEV768 mIoU % | EXP1536 mIoU % |
|---|---:|---:|
| Same-information zero-update mean-context |17.349002|16.499566|
| R123 real point correspondence Adapter TTA |17.233621|16.410475|
| R125 native semantic Adapter TTA, fixed3 steps |17.183657|16.435119|
| R126 full-readout acceptance of same updates |17.183657|16.435119|

R125 DEV6good10bad, EXP10good12bad. R126 rejects16 active updates but all2304 final F Top1 choices remain identical, retaining all beneficial and harmful task changes. Acceptance proxy therefore does not discriminate these task outcomes. All wrong/random/free-feature/same-information direct controls and all negative rows are public. Independent audits replayed native heads/states/Armijo arithmetic; decoder backprop was not independently rerun for all cases, engineering finite-difference scope disclosed.

## Six-direction expansion rows
Every direction256, shared configuration. Values %; delta pp relative to same-info zero update.
| Direction | Original source | Same-info zero | R125/R126 | Delta |
|---|---:|---:|---:|---:|
| waymo→drone | 14.2044 | 15.1165 | 15.0212 | -0.0953 |
| waymo→quad | 29.4795 | 31.1548 | 30.9407 | -0.2140 |
| drone→waymo | 14.6446 | 14.3247 | 14.2324 | -0.0922 |
| drone→quad | 17.4349 | 17.5718 | 17.6011 | +0.0293 |
| quad→waymo | 12.2574 | 11.3873 | 11.2560 | -0.1313 |
| quad→drone | 8.8470 | 9.4423 | 9.5592 | +0.1169 |

## Refined failure attribution (posthoc, no method selection)
1. Useful update directions exist under source-GT; do not conclude this point interface is intrinsically too weak.
2. Registered B has real coverage: valid seed mean~69–70%, source-probability-weighted query-anchor support~77–79%. AB predictions disagree often. More data and disagreement alone do not identify the target.
3. Fit-to-observation transfer gap: on DEV matched AB-active/C-available178 inputs, complete readout A/B +.441795/+1.307464pp but C -.181846pp. A/B task improvement can coexist with C degradation. Same point sampling/region is not automatically a transferable identity constraint.
4. Fusion compatibility: EXP matched350 inputs, F native update +.582120pp; after frozen RGB late -.002759pp; late+sourceA competition -.123987pp; late+F competition -.342586pp. Fixed RGB and competition context can interact adversely with updated point/text features. DEV native F also decreases: deleting RGB is not an established remedy.
5. Fit uses individual A/B inputs; deployment can use F=mean registered ABC features or C. Nonlinear model plus per-input adaptation need not commute with feature aggregation.
6. Full-readout acceptance still measures agreement with source-derived teacher, not target identity correctness. Wrong source consensus can be fitted more consistently.
7. Fixed A256 constrains strict localization. Previous full-source audit oracle Acc50 only18.6874%; ranking adaptation cannot repair boxes absent from the candidate set.

## Good/failure cases
A private gallery contains 18 cached-RGB/BEV pages (8 improvements and 10 failures). Original media, captions, identities and coordinates are excluded from this public export.
Visually opened CASE001,002,009,010 at closeout. CASE009 restores source's .351 IoU from zero-update .030; not a novel correction beyond source. CASE010 moves from .508 to0 and its selected box is spatially far from GT in BEV; free-feature control retains .508. CASE001 moves from0 to.247; CASE002 from.441 to.017 with overlapping nearby box hypotheses. These support heterogeneous failures rather than one cause. Low-resolution cached RGB and projected seed markers are not instance masks; definite physical-object identity remains UNKNOWN. No rule or parameter was changed from these cases.

## Next experiment proposal only, NOT EXECUTED
If reopening research, first isolate fit/serve compatibility with this proven source-capable point interface: same input aggregation for fitting and serving versus current A/B-to-F/C transfer; source-fixed versus recomputed multimodal context from each adapted representation. Keep architecture, data budget, all boxes, same-information zero control and wrong correspondence control fixed. This requires an explicit bounded lock; do not treat posthoc improvements here as a selected method. Do not repeat step/layer/regularizer grids or add a new proxy before this mismatch is isolated. No guarantee of benefit.

## Publication
Combined immutable commit f18b87bed2dbe0e137ef2fa9da89711e728db850; both R125/R126 studies.169 files11,806,757bytes fully verified before CAS main;150 fresh per-file proofs (146 unique blobs),19 renewed immutable proofs;187 old studies+20 other roots preserved. Reports include failures, controls, scene summaries, audits and posthoc disclosure. No raw media/captions/IDs/coordinates/features/weights/gradients/secrets published.
