# Frozen spatial decoder reference geometry qualification32

Same exposed NH16/NV16, one query per parent, original opposite-source EMA TA-STVG weights, E/final-MBR and four DINO cached references. No new model/head training, Source capture, expert, DINO, QR/LN weight update or OPD. Old Fine and coarse bidirectional negative results remain unchanged.

Only SECOND-pass PosDecoder initial pred_boxes is replaced with the complete Source trajectory S or E, correctly split by original two offsets. Original native content, first pass/routing, full memory and fixed temporal output remain. Each subsequent layer predicts an absolute sigmoid box and detaches it for its next reference; this is not an iterative residual bbox head, nor spatial ROI resampling. No clamp/blend/layer selection.

Source-reference and E-reference outputs are raw absolute decoder outputs. The old E-retaining transform is deliberately absent: applying E+[D(E)-S] could double the E correction for an identity-like decoder. Native initial-reference control reproduces Source exactly; same S-reference clone gives exact output. D(S)=S is not required: S was the final prediction, not the original initial reference. No-E-support queries retain E in BOTH reseeded reported arms; raw outputs are saved privately for engineering diagnostics. Same support rule, all16 remain in means.

| Output | HC complete vIoU % | Vid complete vIoU % | HC unobserved actual sIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| Source_MBR | 32.385061 | 11.848986 | 54.378143 | 44.792411 |
| E_MBR | 35.305477 | 15.755931 | 56.660359 | 45.448473 |
| Source_reference | 32.360380 | 11.850038 | 54.348849 | 44.799761 |
| E_reference | 32.366023 | 11.849748 | 54.357803 | 44.790565 |

## hc2: paired parent differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Source_MBR_minus_E_MBR | -2.920415 [-6.351563,+1.402473] | -2.282216 [-8.218397,+5.507698] | 3/12/1 | 7 |
| Source_reference_minus_E_MBR | -2.945097 [-6.345387,+1.319597] | -2.311510 [-8.218039,+5.458451] | 3/12/1 | 7 |
| E_reference_minus_E_MBR | -2.939454 [-6.344346,+1.352012] | -2.302556 [-8.217315,+5.469663] | 3/12/1 | 7 |
| E_reference_minus_Source_reference | +0.005643 [-0.008988,+0.026476] | +0.008954 [-0.011022,+0.032771] | 7/8/1 | 0 |
Active support 15/16. All no-reference and missing-GT cases retained; absent actual GT stays unknown.

## vidstg: paired parent differences, pp

| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | v positive/negative/neutral | v below -5pp |
|---|---|---|---|---|
| Source_MBR_minus_E_MBR | -3.906945 [-13.362801,+2.764511] | -0.656063 [-15.833502,+15.255392] | 2/6/8 | 3 |
| Source_reference_minus_E_MBR | -3.905894 [-13.362328,+2.764960] | -0.648713 [-15.826327,+15.262047] | 2/6/8 | 3 |
| E_reference_minus_E_MBR | -3.906183 [-13.362181,+2.764913] | -0.657909 [-15.827976,+15.244251] | 2/6/8 | 3 |
| E_reference_minus_Source_reference | -0.000290 [-0.001047,+0.000260] | -0.009196 [-0.023972,-0.000407] | 4/3/9 | 0 |
Active support 11/16. All no-reference and missing-GT cases retained; absent actual GT stays unknown.

## Fixed eligibility
{'status': 'evaluated', 'passed': False, 'rule': 'Both-domain E_reference-minus-E and E_reference-minus-Source_reference complete-v and unobserved actual-s means must be positive; otherwise stop exact reseeding interface without parameter or blend search', 'OPD_not_run': True}

Only both-domain positive E-reference minus E and minus Source-reference complete-v AND unobserved-actual-s means qualify further work. Confidence intervals retain uncertainty; passing only a damaged reseeding control is insufficient. Failure stops this exact initial-reference reseeding without fusion weights, per-layer clamp or scans. This run tests existing decoder capability and is not an OPD effectiveness result.

All32 outputs/traces sealed before new GT scoring. This is historically exposed target development data, not independent confirmation. GT does not generate references, choose layers or deployment outputs. Temporal intervals unchanged across all four arms; tIoU identical. Official interpolation does not count as additional visual supervision.

Independent CPU audit rechecks actual Source/E reference tensors and offset order, identical native content, all six saved outputs and final raw reconstruction with no E transform, native-reference identity, no-reference no-op, independent official interpolation metrics and parent paired bootstrap. Frozen weights checked in actual model execution and registries pinned. No extra GPU/model rerun for audit.

D(E) versus D(S) complete-v differences are HC+.005643pp and Vid-.000290pp, intervals cross zero. D(S) versus original Source is only -.024681/+.001052pp, while E versus Source was +2.920415/+3.906945pp. Thus the large loss versus E mainly reflects not retaining the established E output correction, not a demonstrated large additional harm caused by choosing E rather than S as reference. Per-layer forward response is in REFERENCE_RESPONSE_PUBLIC.json; no new identity claims.
Initial independent CPU metric verifier formed xyxy in float32, unlike official float64 evaluation; verifier-only float64 input cast corrected, original verifier preserved, strict2e-10 tolerance retained. Predictions/scoring/model runs unchanged.
Costs (old Source/E/DINO/prefix generation excluded): {'total_Source_captures_completed': 0, 'final_worker_DINO_encodings_completed': 0, 'engineering_DINO_encodings_completed': 0, 'new_training_runs': 0, 'new_heads': 0, 'new_OPD_updates': 0, 'GPU_allocation_seconds': 45.08874348318204, 'actual_worker_GPU_seconds': 42.08874461520463, 'peak_allocated_bytes': 1278273536, 'readout_and_equal_control_seconds': 20.062990412581712, 'old_Source_E_DINO_prefix_cost_excluded': True, 'not_first_online_latency': True}

This interface is specific to the inspected TA-STVG PosDecoder. No production promotion or backbone-general claim. Single GPT-6.1 Sol reviewer, root decision authority and original09:30 absolute deadline retained.

## Every signed full-tube change vs E, pp

| Alias | Target | Source-reference | E-reference |
|---|---|---:|---:|
| NH001 | hc2 | -1.562469 | -1.550649 |
| NH002 | hc2 | -13.228225 | -13.229167 |
| NH003 | hc2 | -6.454114 | -6.505039 |
| NH004 | hc2 | +2.163773 | +2.159860 |
| NH005 | hc2 | -0.551430 | -0.550827 |
| NH006 | hc2 | +22.203764 | +22.342520 |
| NH007 | hc2 | -6.530432 | -6.530714 |
| NH008 | hc2 | -4.265553 | -4.264990 |
| NH009 | hc2 | +0.593582 | +0.603787 |
| NH010 | hc2 | -6.587943 | -6.587895 |
| NH011 | hc2 | -0.331191 | -0.332520 |
| NH012 | hc2 | +0.000000 | +0.000000 |
| NH013 | hc2 | -8.481730 | -8.497988 |
| NH014 | hc2 | -4.785522 | -4.785864 |
| NH015 | hc2 | -14.011504 | -14.012045 |
| NH016 | hc2 | -5.292556 | -5.289727 |
| NV001 | vidstg | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 |
| NV005 | vidstg | -64.071553 | -64.071553 |
| NV006 | vidstg | +0.000000 | +0.000000 |
| NV007 | vidstg | -2.394982 | -2.395991 |
| NV008 | vidstg | -4.864785 | -4.863860 |
| NV009 | vidstg | +0.000000 | +0.000000 |
| NV010 | vidstg | -0.844060 | -0.844008 |
| NV011 | vidstg | +14.494510 | +14.489437 |
| NV012 | vidstg | +16.342864 | +16.343365 |
| NV013 | vidstg | +0.000000 | +0.000000 |
| NV014 | vidstg | -5.282542 | -5.283953 |
| NV015 | vidstg | +0.000000 | +0.000000 |
| NV016 | vidstg | -15.873749 | -15.872367 |
