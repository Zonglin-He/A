# R144 — Source-reference positional conditioning prerequisite

## Decision

**retain_input_state_candidate_not_parameter_TTA**. This is a fixed, source-calibrated input-state prerequisite, not a claim of a new parameter-TTA gain. R142 remains an unchanged development reference. All1798 previously exposed records, sixdirections and allnegative/noB cases remain.

## Mechanism and fixed scope

The checkpoint-compatible archived runtime learns absolute positions in point visual self-attention, vision-to-language queries, and decoder query self-attention. Language-to-vision keys do not directly receive pos_feats. Source-specific absolute-origin priors are a hypothesis; no camera-coordinate/preprocessing error is asserted. The current public official bdetr uses a different backbone backend, so the actual checkpoint-compatible runtime is the computational authority. [Official positional modules](https://github.com/worldbench/3EED/blob/main/models/encoder_decoder_layers.py) support the entry-point observation, not the efficacy of our estimator.

One fixed source-only formula: Hsource is the median rawsample lower10percentile height over302deduplicatedsource-owninputs; delta=Hsource-q10(exactoriginalA16384sample). Labels do not enter source statistics, target quantiles or the sign/scale. Lower10percentile is NOT certifiedground/sensorheight; terrain, scenecomposition andFOV are competing explanations. No quantile/offset/strength/axis search.

On the original764B-available records only, recompute pointpos_embed atz+delta and translatecenterZ in allsix cachedqueryposition arrays. Keep physicalxyz, PointNet features, registeredFpoint1024, A/Bcamera graphs, originalroute/support andrelativegeometricpriors, sourceparameters, queryidentities andoriginalA256outputboxes fixed. Recompute complete native matching andlivecompetition. This isolates positional conditioning, not a fully translated pointcloud/backbonepass. Frozenrouting is an intentionally held prior for allarms. All1034noB remain exactcompleteFrozenA, including the newcontrols. No sourceweighttraining, expert, architecture, loss or parameter update.

Controls are same-magnitude opposite-reference adjustment and fixedone-half originaljointzero/aligned probabilitymean, plus allR142 zero/direct/wrong/TTA/pureB/canonicalmeans byte-reused. A positive inputstate method would be an additional frozen baseline; parameteradaptation must later exceed that baseline before an adaptation contribution. No matching to an unexecuted globallytranslated PointNet is claimed.

## Whole-cohort results

| aggregation | arm | mIoU | Acc25 | Acc50 |
|---|---|---|---|---|
| raw | Frozen_A | 13.40304063926085 | 25.899514674225376 | 5.056806006058658 |
| raw | Frozen_F | 13.605853773885677 | 25.898010098227072 | 5.154946976306667 |
| raw | R135_A | 13.506393386808027 | 26.220572058927896 | 5.102343346677966 |
| raw | R135_F | 13.62524419518555 | 26.175826142677636 | 5.343192915516926 |
| raw | C_B_A | 13.472876819351491 | 25.937807437927976 | 5.0604282945170125 |
| raw | direct_C_B_A | 13.483204067665383 | 26.12684480150729 | 5.103491346861646 |
| raw | F_RGBB_only | 13.670413083165842 | 26.08740403762101 | 5.154946976306667 |
| raw | F_RGBB_only_wrong | 13.601881542824756 | 25.89915809841075 | 5.249643946003637 |
| raw | F_Bimage_only_teacher | 13.666392050183278 | 25.959362880770787 | 5.484575226016884 |
| raw | direct_image_B | 13.560908557733756 | 25.806935416988456 | 5.341866627425934 |
| raw | direct_wrong_image_B | 12.884815787950117 | 25.160585222663332 | 5.157421264581342 |
| raw | image_B_TTA | 13.701826022244445 | 26.22419434738625 | 5.344340915700607 |
| raw | wrong_image_B_TTA | 13.6245791986593 | 26.178478718859623 | 5.342044915333247 |
| raw | FA_FB_probability_mean | 13.535399176653073 | 25.724609888664855 | 5.343192915516926 |
| raw | FAB_FB_probability_mean | 13.54793075841048 | 25.816832570087147 | 5.343192915516926 |
| raw | aligned_reference | 13.716845698214941 | 26.56221517419685 | 5.249643946003637 |
| raw | opposite_reference | 13.577198921362617 | 25.7592803639091 | 5.247347945636277 |
| raw | half_zero_aligned | 13.630067729578494 | 26.18457529559265 | 5.154946976306667 |
| design_weighted | Frozen_A | 13.394463465154152 | 26.26330584715755 | 4.599296733787808 |
| design_weighted | Frozen_F | 13.580861241247362 | 26.099350265391482 | 4.688055597703281 |
| design_weighted | R135_A | 13.498989254903094 | 26.600013496665493 | 4.6448340744071155 |
| design_weighted | R135_F | 13.606613571201727 | 26.393334446143836 | 4.859257252330195 |
| design_weighted | C_B_A | 13.459843379670076 | 26.27080991908078 | 4.601034064844766 |
| design_weighted | direct_C_B_A | 13.477421661915146 | 26.44400022063034 | 4.639678907165158 |
| design_weighted | F_RGBB_only | 13.670699786296026 | 26.377786302278206 | 4.688055597703281 |
| design_weighted | F_RGBB_only_wrong | 13.575009890183594 | 26.100757201323347 | 4.753881508597696 |
| design_weighted | F_Bimage_only_teacher | 13.681997091241767 | 26.281907798110392 | 5.002192919558421 |
| design_weighted | direct_image_B | 13.560985917308079 | 26.101943522949373 | 4.849542504149657 |
| design_weighted | direct_wrong_image_B | 12.877941295183637 | 25.395895406548494 | 4.696965748443405 |
| design_weighted | image_B_TTA | 13.695323877823503 | 26.49839657604141 | 4.859808261761123 |
| design_weighted | wrong_image_B_TTA | 13.620422957809708 | 26.454584516708696 | 4.883327911208479 |
| design_weighted | FA_FB_probability_mean | 13.537643294584846 | 26.0255123491001 | 4.859257252330195 |
| design_weighted | FAB_FB_probability_mean | 13.550180994034699 | 26.11799396627058 | 4.859257252330195 |
| design_weighted | aligned_reference | 13.739713257337598 | 27.044149189023084 | 4.793982350866708 |
| design_weighted | opposite_reference | 13.566661145279816 | 26.053982412669857 | 4.75277948973584 |
| design_weighted | half_zero_aligned | 13.64252453806843 | 26.525001795123256 | 4.688055597703281 |

{
  "retain_input_state_development_candidate": true,
  "parameter_TTA_contribution_claimed": false,
  "comparisons": {
    "raw": {
      "F_RGBB_only": 0.04643261504909901,
      "opposite_reference": 0.13964677685232374,
      "half_zero_aligned": 0.08677796863644716,
      "image_B_TTA": 0.015019675970496138,
      "Frozen_A": 0.3138050589540917
    },
    "design_weighted": {
      "F_RGBB_only": 0.06901347104157196,
      "opposite_reference": 0.17305211205778193,
      "half_zero_aligned": 0.09718871926916783,
      "image_B_TTA": 0.044389379514095495,
      "Frozen_A": 0.34524979218344676
    }
  },
  "FINAL_access": false,
  "no_grid_rescue_if_negative": true,
  "source_q10_not_certified_ground": true,
  "disposition": "retain_input_state_candidate_not_parameter_TTA"
}

The predeclared retention rule requires aligned > ownzero/opposite/fixedhalfmean inbothraw anddesignweightedmeans. R142 comparisons are also shown. This is exposedtarget development, not an independent test, and no per-direction bestoutput is selected. A negative recipe ends without an offset/quantilegrid; it only weakens this tested estimator/intervention, not every possible coordinate adaptation.

## Input reference distribution

| source | target | B | delta_mean | delta_q10 | delta_median | delta_q90 |
|---|---|---|---|---|---|---|
| drone | quad | 178 | -1.0888072859705162 | -1.2442113876342773 | -1.0999269247055055 | -0.8535018914937973 |
| drone | waymo | 50 | -1.6696019193679092 | -1.9714708685874938 | -1.8148001074790954 | -0.8588436102867127 |
| quad | drone | 154 | 0.5692478652898368 | -0.04350905776023871 | 0.4677511274814605 | 1.2113932192325596 |
| quad | waymo | 50 | -0.6258130869418382 | -0.9276820361614228 | -0.7710112750530242 | 0.1849452221393584 |
| waymo | drone | 154 | 1.3017882998887595 | 0.689031376838684 | 1.2002915620803833 | 1.9439336538314824 |
| waymo | quad | 178 | 0.6875219810544774 | 0.5321178793907165 | 0.6764023423194885 | 0.9228273755311965 |

Signed reference changes differ because each source has its own frozen statistic. These are aggregated diagnostics, not object coordinates or inferred identity labels.

## All gains and harms

| control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|
| Frozen_A | 110 | 96 | 0.34524979218344765 | 0.053304207197498364 | -0.04909164122994028 | 120 | 174 | 17 |
| F_RGBB_only | 49 | 50 | 0.06901347104157265 | 0.053304207197498364 | -0.036117173255342604 | 78 | 91 | 16 |
| opposite_reference | 80 | 74 | 0.1730521120577801 | 0.053304207197498364 | -0.036117173255342604 | 98 | 135 | 16 |
| half_zero_aligned | 33 | 25 | 0.09718871926916427 | 0.053304207197498364 | -0.026944995983505412 | 51 | 56 | 11 |
| image_B_TTA | 70 | 72 | 0.04438937951409554 | 0.040096090214912596 | -0.036117173255342604 | 92 | 122 | 16 |

IoU changes need not imply a different physicalobject. Frames/content groups can be shared by multiple descriptions/source directions. Full sixdirection/scenegroup and changedrecord tables remain, without deleting large harms or claiming a singleDrone scene yields a populationconfidenceinterval.

## Audit and resources

{
  "decision": {
    "retain_input_state_development_candidate": true,
    "parameter_TTA_contribution_claimed": false,
    "comparisons": {
      "raw": {
        "F_RGBB_only": 0.04643261504909901,
        "opposite_reference": 0.13964677685232374,
        "half_zero_aligned": 0.08677796863644716,
        "image_B_TTA": 0.015019675970496138,
        "Frozen_A": 0.3138050589540917
      },
      "design_weighted": {
        "F_RGBB_only": 0.06901347104157196,
        "opposite_reference": 0.17305211205778193,
        "half_zero_aligned": 0.09718871926916783,
        "image_B_TTA": 0.044389379514095495,
        "Frozen_A": 0.34524979218344676
      }
    },
    "FINAL_access": false,
    "no_grid_rescue_if_negative": true,
    "source_q10_not_certified_ground": true,
    "disposition": "retain_input_state_candidate_not_parameter_TTA"
  },
  "preparation": {
    "n": 1798,
    "eligible_B": 764,
    "source_own_median_q10": {
      "waymo": -0.07376708984375,
      "drone": -1.8500963568687439,
      "quad": -0.8063075244426727
    },
    "deduplicated_source_contents": 302,
    "target_input_sampledraws": 382,
    "max_seed_input_coordinate_error": 3.91504840286248e-06,
    "GT_or_model_read": false,
    "FINAL_access": false
  },
  "engineering": {
    "status": "PASS",
    "n": 12,
    "B": 6,
    "max_CPU_positional_MLP_error": 4.291534423828125e-06,
    "max_CPU_full_reader_error": 4.4466442716828425e-07,
    "all_references_nobox_noB_exact": true,
    "parameter_updates": 0,
    "fullTransformer_not_independently_recomputed_CPU": true,
    "FINAL_access": false
  },
  "native_audit": {
    "status": "PASS",
    "n": 1798,
    "B": 764,
    "max_CPU_positional_MLP_error": 1.33514404296875e-05,
    "max_CPU_full_reader_error": 1.051372955251928e-06,
    "all_references_nobox_noB_exact": true,
    "parameter_updates": 0,
    "fullTransformer_not_independently_recomputed_CPU": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "task_values": 32364,
    "max_error": 0.0,
    "all_negative_noB_ineffective_rows_kept": true
  },
  "runtime": {
    "n": 1798,
    "full_reader_forwards": 1528,
    "parameters_updated": 0,
    "max_engineering_zero_probability_error": 0.0,
    "max_engineering_pos0_recompute_error": 0.0,
    "GT_read": false,
    "FINAL_access": false
  },
  "delta_summary": [
    {
      "source": "drone",
      "target": "quad",
      "B": 178,
      "delta_mean": -1.0888072859705162,
      "delta_q10": -1.2442113876342773,
      "delta_median": -1.0999269247055055,
      "delta_q90": -0.8535018914937973
    },
    {
      "source": "drone",
      "target": "waymo",
      "B": 50,
      "delta_mean": -1.6696019193679092,
      "delta_q10": -1.9714708685874938,
      "delta_median": -1.8148001074790954,
      "delta_q90": -0.8588436102867127
    },
    {
      "source": "quad",
      "target": "drone",
      "B": 154,
      "delta_mean": 0.5692478652898368,
      "delta_q10": -0.04350905776023871,
      "delta_median": 0.4677511274814605,
      "delta_q90": 1.2113932192325596
    },
    {
      "source": "quad",
      "target": "waymo",
      "B": 50,
      "delta_mean": -0.6258130869418382,
      "delta_q10": -0.9276820361614228,
      "delta_median": -0.7710112750530242,
      "delta_q90": 0.1849452221393584
    },
    {
      "source": "waymo",
      "target": "drone",
      "B": 154,
      "delta_mean": 1.3017882998887595,
      "delta_q10": 0.689031376838684,
      "delta_median": 1.2002915620803833,
      "delta_q90": 1.9439336538314824
    },
    {
      "source": "waymo",
      "target": "quad",
      "B": 178,
      "delta_mean": 0.6875219810544774,
      "delta_q10": 0.5321178793907165,
      "delta_median": 0.6764023423194885,
      "delta_q90": 0.9228273755311965
    }
  ],
  "parameter_TTA_claim": false,
  "FINAL_access": false
}

Predictions sealed before taskscore. Engineering zero correction reproduces complete source readout, sourcefingerprints remain unchanged. CPU independently recomputes the actual positional MLP including evalBN, allsixcenterZ changes withunchangedXY/sizes, and fullfinalmatching/competition probabilities from saved nativefeatures. It checks originalA256/ref/noB/halfmean exactly. Source rawsample coordinates were verified against originalseed coordinates during preparation; this preparatory check is not an independent fullbackbone replay. WholeTransformer isnot independently regenerated onCPU. Every savedIoU/aggregate isrecalculated. Allruns use uniqueGPU/deadlineguard; no newexperiment after09:00BeijingNov18. Rawmedia/descriptions/coordinates/IDs/features/modelweights/secrets areprivate; noFINAL.


## Descriptive limits

Relative to ownjointzero, alignedreference changes IoU positively49times andnegatively50times. Fourdirections improve, two decline; Drone-to-Quad loses0.680956pp, while Drone-to-Waymo gains0.426165pp andQuad-to-Waymo gains0.706035pp. Thus this aggregate gain doesnot establish a universalheightmisalignment bug. The largestpositive contribution0.053304pp isbelow totalgain0.069013pp, but stillsubstantial. Acc50 is4.793982%, belowR1424.859808%; noallmetricdominance. These are postseal descriptions, not outputgates or grounds todelete cases.
