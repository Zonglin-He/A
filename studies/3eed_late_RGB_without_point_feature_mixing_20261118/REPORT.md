# R147 — Late RGB without new point-feature averaging

## Decision

**frozen_entry_prerequisite_FAIL_stop_entry_keep_R145**. This was the predeclared frozen-entry prerequisite, not parameterTTA. It failed, so no source-reader recalibration or newUpupdate wasrun; no loss/step/threshold grid rescues thisentry. RetaincoherentR145developmentcandidate, not perrecord/directionbestoutput. Parent>=2ppgoal remainsseparateandnotcountedfromGTdiagnostic/weakerreset baseline.

## Measured failure and controlled change

R146fixednew166path had rawA/Bpointfeaturemean-.140899pp, addedRGB+.012761, positionreference+.060488 andUpupdate-.045693pp. This is a specificpath decomposition, notproofallBpoints/registrationarebad. Original764B pointmixhadpositiveFrozenFvsA, so itisnotgloballyremoved. EarlierR105/R141/R142RGBexperiments areacknowledged; RGBB itself isnotnew.

R147 changesonlythe166previouslymissing-B inputs: useoriginalA1024pointfeatures insteadA/Bmean. OriginalBcamera/image/physicalprojectiongraphs, sparsevisibility/supportqualification, sameHsource-q10positionreference, sourcepreparedCNN/bidirectionalgrounder/fullmatch/livecompetition, originalA256/boxcoordinates/sixqueryidentities unchanged. ActualPointAdapterprehook recordsinputhasheseachnewforward andindependentCPU checksagainstexactAfeatures. No newBPointNet isneeded orclaimed; rawB isstillusedforcalibration/visibilitygeometry, notpointfeatures.

All1798exposed ordinaryinputs/sixdirections retainoriginalweightsandframe/content/sceneclusters. All49R146endpointrefs copiedfromsealedparentfiles andauditedbitexact. Newpa_* globallyfrozenarms useR145correspondingfrozenreadouts onunchanged1632; inc_pa_* useold1632R145norm_TTA andonlynew166specifiedfrozenreadout. These incrementalcontrols are NOT globallyno-update systems. Remaining868noBstaycompleteFrozenA. No targetGTchoiceperinput/platform/scene.

## Frozen-entry results

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
| raw | norm_joint_zero | 13.716845698214941 | 26.56221517419685 | 5.249643946003637 |
| raw | norm_Aimage | 13.661544723534073 | 26.46619191640889 | 5.249643946003637 |
| raw | norm_Bimage | 13.711942296165214 | 26.471140492958238 | 5.48324893792589 |
| raw | norm_joint_wrong | 13.59665405392136 | 26.41818028751491 | 5.154946976306667 |
| raw | norm_B_wrong | 13.657651326585928 | 26.421980863880574 | 5.436563597122903 |
| raw | norm_FA_FB_mean | 13.598653219807787 | 26.23753550103599 | 5.343192915516926 |
| raw | norm_FAB_FB_mean | 13.624958676076995 | 26.283072841655294 | 5.389878256319915 |
| raw | norm_direct | 13.61923556074858 | 26.46848791677625 | 5.48324893792589 |
| raw | norm_wrong_direct | 12.890186442322497 | 24.74217829208186 | 5.346815203975281 |
| raw | norm_TTA | 13.801914656549181 | 26.834112929821703 | 5.528786278545199 |
| raw | norm_wrong_TTA | 13.671675192005827 | 26.552496309005473 | 5.294033286439266 |
| raw | bf_raw_F_A | 13.492763350997327 | 25.573330425066203 | 5.154946976306667 |
| raw | bf_raw_joint | 13.570342475057373 | 25.76272436446014 | 5.154946976306667 |
| raw | bf_zero | 13.66882180935596 | 26.375295523077586 | 5.3407186272422535 |
| raw | bf_Aimage | 13.619923058313105 | 26.279272265289624 | 5.295181286622945 |
| raw | bf_Bimage | 13.655327298034146 | 26.143986531522696 | 5.437711597306584 |
| raw | bf_wrong_joint | 13.51456190378063 | 26.136563666698674 | 5.200484316925976 |
| raw | bf_wrong_B | 13.57892493619822 | 26.003752221206415 | 5.3910262565035945 |
| raw | bf_FA_FB_mean | 13.518359021306075 | 25.90923353941676 | 5.297655574897619 |
| raw | bf_FAB_FB_mean | 13.544664477575287 | 25.95477088003607 | 5.344340915700607 |
| raw | bf_direct | 13.50547236465058 | 25.999951644840746 | 5.437711597306584 |
| raw | bf_wrong_direct | 12.787873033917387 | 24.225986967067005 | 5.392352544594588 |
| raw | bf_TTA | 13.693713868254106 | 26.411113998505513 | 5.528786278545199 |
| raw | bf_wrong_TTA | 13.630630482334636 | 26.36557665788621 | 5.434267596755543 |
| raw | inc_zero | 13.7538907676902 | 26.647193278702442 | 5.619860959783814 |
| raw | inc_direct | 13.688151460451179 | 26.365576657886205 | 5.48324893792589 |
| raw | inc_Bimage | 13.745299658418114 | 26.50695896838616 | 5.48324893792589 |
| raw | inc_FA_FB_mean | 13.721620458047468 | 26.505810968202482 | 5.48324893792589 |
| raw | inc_FAB_FB_mean | 13.721620458047468 | 26.505810968202482 | 5.48324893792589 |
| raw | inc_wrong_TTA | 13.760869946877989 | 26.647193278702442 | 5.669020588861476 |
| raw | inc_raw_joint | 13.70184404844071 | 26.509433256660838 | 5.528786278545199 |
| raw | pa_raw_joint | 13.670413083165842 | 26.08740403762101 | 5.154946976306667 |
| raw | pa_zero | 13.668503533738786 | 26.421980863880577 | 5.20410660538433 |
| raw | pa_Aimage | 13.613202559057918 | 26.32595760609261 | 5.20410660538433 |
| raw | pa_Bimage | 13.62123437022538 | 26.19067187232568 | 5.437711597306584 |
| raw | pa_wrong_joint | 13.570515171003867 | 26.3726429468956 | 5.10940963568736 |
| raw | pa_wrong_B | 13.600457783705513 | 26.281746553564293 | 5.3910262565035945 |
| raw | pa_FA_FB_mean | 13.539804496875655 | 26.094826902445032 | 5.297655574897619 |
| raw | pa_FAB_FB_mean | 13.566109953144867 | 26.14036424306434 | 5.344340915700607 |
| raw | inc_pa_raw_joint | 13.801914656549181 | 26.834112929821703 | 5.528786278545199 |
| raw | inc_pa_zero | 13.753572492073024 | 26.693878619505426 | 5.48324893792589 |
| raw | inc_pa_Aimage | 13.753572492073024 | 26.693878619505426 | 5.48324893792589 |
| raw | inc_pa_Bimage | 13.711206730609346 | 26.553644309189153 | 5.48324893792589 |
| raw | inc_pa_wrong_joint | 13.775775773631684 | 26.7885755892024 | 5.48324893792589 |
| raw | inc_pa_wrong_B | 13.744721113668767 | 26.693878619505426 | 5.48324893792589 |
| raw | inc_pa_FA_FB_mean | 13.743065933617048 | 26.691404331230757 | 5.48324893792589 |
| raw | inc_pa_FAB_FB_mean | 13.743065933617048 | 26.691404331230757 | 5.48324893792589 |
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
| design_weighted | norm_joint_zero | 13.739713257337598 | 27.044149189023084 | 4.793982350866708 |
| design_weighted | norm_Aimage | 13.659170072455359 | 26.859601705238706 | 4.793982350866708 |
| design_weighted | norm_Bimage | 13.733419725521841 | 26.93794211447542 | 5.029399635265411 |
| design_weighted | norm_joint_wrong | 13.588304121042142 | 26.80061922862645 | 4.688055597703281 |
| design_weighted | norm_B_wrong | 13.684126964029248 | 26.889053320846184 | 4.982455358714241 |
| design_weighted | norm_FA_FB_mean | 13.616389042580542 | 26.70494601300616 | 4.89027865861377 |
| design_weighted | norm_FAB_FB_mean | 13.642479898952432 | 26.75048335362547 | 4.936367008664006 |
| design_weighted | norm_direct | 13.643483266302125 | 26.90261572620692 | 5.029399635265412 |
| design_weighted | norm_wrong_direct | 12.904351468797019 | 24.995010514649366 | 4.944884792156855 |
| design_weighted | norm_TTA | 13.81960316688109 | 27.305335032677846 | 5.063686987738404 |
| design_weighted | norm_wrong_TTA | 13.686396172517528 | 27.007052866504143 | 4.8446748587279735 |
| design_weighted | bf_raw_F_A | 13.439961806380603 | 25.68120801339887 | 4.688055597703281 |
| design_weighted | bf_raw_joint | 13.542561004840914 | 25.95964405028559 | 4.688055597703281 |
| design_weighted | bf_zero | 13.672062309362886 | 26.76402589482026 | 4.8850570321053235 |
| design_weighted | bf_Aimage | 13.597923722154503 | 26.579478411035875 | 4.839519691486016 |
| design_weighted | bf_Bimage | 13.670772933630664 | 26.556948937239724 | 4.983862294646104 |
| design_weighted | bf_wrong_joint | 13.500478203462741 | 26.466019318510998 | 4.733592938322589 |
| design_weighted | bf_wrong_B | 13.599380165133697 | 26.416985462371873 | 4.936918018094935 |
| design_weighted | bf_FA_FB_mean | 13.530363967563419 | 26.323401826339538 | 4.844741317994462 |
| design_weighted | bf_FAB_FB_mean | 13.556454823935313 | 26.36893916695885 | 4.890829668044699 |
| design_weighted | bf_direct | 13.558663737232417 | 26.515887247745155 | 4.983862294646103 |
| design_weighted | bf_wrong_direct | 12.806952845741987 | 24.46387871500492 | 4.990422132776162 |
| design_weighted | bf_TTA | 13.70625938173397 | 26.83097883371566 | 5.063686987738404 |
| design_weighted | bf_wrong_TTA | 13.626774964666192 | 26.72607364580038 | 4.991094821407688 |
| design_weighted | inc_zero | 13.75195221890638 | 27.02521173847502 | 5.154761668977019 |
| design_weighted | inc_direct | 13.734783637811384 | 26.918606554216087 | 5.018149647119096 |
| design_weighted | inc_Bimage | 13.756956374989912 | 26.92434185544216 | 5.018149647119096 |
| design_weighted | inc_FA_FB_mean | 13.733578091863968 | 26.92379084601123 | 5.018149647119096 |
| design_weighted | inc_FAB_FB_mean | 13.733578091863968 | 26.92379084601123 | 5.018149647119096 |
| design_weighted | inc_wrong_TTA | 13.759981959029753 | 27.02435581197409 | 5.210106950418118 |
| design_weighted | inc_raw_joint | 13.691464385425979 | 26.887192780685236 | 5.063686987738404 |
| design_weighted | pa_raw_joint | 13.670699786296026 | 26.377786302278206 | 4.688055597703281 |
| design_weighted | pa_zero | 13.676880564878127 | 26.810970171371427 | 4.7484450102474005 |
| design_weighted | pa_Aimage | 13.596337379995893 | 26.62642268758705 | 4.7484450102474005 |
| design_weighted | pa_Bimage | 13.626479203600242 | 26.566413743908868 | 4.983862294646104 |
| design_weighted | pa_wrong_joint | 13.567770591396922 | 26.75508188800714 | 4.642518257083974 |
| design_weighted | pa_wrong_B | 13.630777423115333 | 26.7507039679313 | 4.936918018094935 |
| design_weighted | pa_FA_FB_mean | 13.541307226612654 | 26.4714366002294 | 4.844741317994462 |
| design_weighted | pa_FAB_FB_mean | 13.567398082984546 | 26.516973940848704 | 4.890829668044699 |
| design_weighted | inc_pa_raw_joint | 13.81960316688109 | 27.305335032677846 | 5.063686987738404 |
| design_weighted | inc_pa_zero | 13.756770474421623 | 27.072156015026184 | 5.018149647119096 |
| design_weighted | inc_pa_Aimage | 13.756770474421623 | 27.072156015026184 | 5.018149647119096 |
| design_weighted | inc_pa_Bimage | 13.712662644959492 | 26.933806662111305 | 5.018149647119096 |
| design_weighted | inc_pa_wrong_joint | 13.799069637235869 | 27.259797692058537 | 5.018149647119096 |
| design_weighted | inc_pa_wrong_B | 13.766253625967174 | 27.166985679762963 | 5.018149647119096 |
| design_weighted | inc_pa_FA_FB_mean | 13.744521350913201 | 27.07182561990109 | 5.018149647119096 |
| design_weighted | inc_pa_FAB_FB_mean | 13.744521350913201 | 27.07182561990109 | 5.018149647119096 |

{
  "primary": "inc_pa_zero",
  "phase": "frozen_entry_prerequisite_not_parameter_TTA",
  "FINAL_access": false,
  "exposed_target_method_development": true,
  "new_source_weight_training": false,
  "endpoints": {
    "inc_pa_zero": {
      "comparisons": {
        "raw": {
          "norm_TTA": -0.048342164476157024,
          "inc_zero": -0.00031827561717534536,
          "inc_pa_Aimage": 0.0,
          "inc_pa_wrong_joint": -0.022203281558660137,
          "inc_pa_Bimage": 0.0423657614636781,
          "inc_pa_FA_FB_mean": 0.010506558455976034,
          "inc_pa_FAB_FB_mean": 0.010506558455976034,
          "inc_pa_raw_joint": -0.048342164476157024,
          "Frozen_A": 0.3505318528121748
        },
        "design_weighted": {
          "norm_TTA": -0.06283269245946599,
          "inc_zero": 0.004818255515242953,
          "inc_pa_Aimage": 0.0,
          "inc_pa_wrong_joint": -0.042299162814245506,
          "inc_pa_Bimage": 0.04410782946213132,
          "inc_pa_FA_FB_mean": 0.012249123508421889,
          "inc_pa_FAB_FB_mean": 0.012249123508421889,
          "inc_pa_raw_joint": -0.06283269245946599,
          "Frozen_A": 0.3623070092674716
        }
      },
      "retained_development_candidate": false
    }
  },
  "disposition": "frozen_entry_prerequisite_FAIL_stop_entry_keep_R145",
  "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
}

Registeredinc_pa_zeroprereq mustexceed coherentparentR145, FpointlateBzeroR146, samepositionAimage andwrongBbinding inbothraw/designweightedmeans. PureB/twofixedhalfmeans alsofullyreported, notselectedpercase. Passingthisstagewouldonlyauthorize sourceApointreader-matchedcalibration andsameUpTTA; failedstageisnotdirectevidencethatallpossibleTTAcannotwork.

## Gains and harms

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| inc_pa_zero | norm_TTA | 6 | 14 | -0.06283269245946904 | 0.014213462177684698 | -0.040551159847957464 | 18 | 20 | 8 |
| inc_pa_zero | inc_zero | 21 | 17 | 0.004818255515241364 | 0.016332841028220527 | -0.033029320825999416 | 25 | 35 | 4 |
| inc_pa_zero | inc_pa_Aimage | 0 | 0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 |
| inc_pa_zero | inc_pa_wrong_joint | 0 | 2 | -0.042299162814247476 | 0.0 | -0.040551159847957464 | 2 | 2 | 2 |
| inc_pa_zero | inc_pa_Bimage | 20 | 18 | 0.04410782946212929 | 0.024006698805007665 | -0.014165923248266201 | 24 | 32 | 4 |
| inc_pa_zero | inc_pa_FA_FB_mean | 11 | 11 | 0.012249123508418324 | 0.024006698805007665 | -0.014165923248266201 | 21 | 22 | 4 |
| inc_pa_zero | inc_pa_FAB_FB_mean | 11 | 11 | 0.012249123508418324 | 0.024006698805007665 | -0.014165923248266201 | 21 | 22 | 4 |
| inc_pa_zero | Frozen_A | 123 | 116 | 0.36230700926747017 | 0.053304207197498364 | -0.04909164122994028 | 132 | 201 | 20 |

ThesearecandidateboxIoU changes, notindependentlyverifiedphysicalobjectidentity. Allchangedpositiveandnegativecases kept; sharedframes/descriptions/source directions arenotindependentrecordings. No singleDrone scene populationCI or removinglargestharm.

## Measurement and cost

{
  "phase": "frozen_evidence_entry_only",
  "decision": {
    "primary": "inc_pa_zero",
    "phase": "frozen_entry_prerequisite_not_parameter_TTA",
    "FINAL_access": false,
    "exposed_target_method_development": true,
    "new_source_weight_training": false,
    "endpoints": {
      "inc_pa_zero": {
        "comparisons": {
          "raw": {
            "norm_TTA": -0.048342164476157024,
            "inc_zero": -0.00031827561717534536,
            "inc_pa_Aimage": 0.0,
            "inc_pa_wrong_joint": -0.022203281558660137,
            "inc_pa_Bimage": 0.0423657614636781,
            "inc_pa_FA_FB_mean": 0.010506558455976034,
            "inc_pa_FAB_FB_mean": 0.010506558455976034,
            "inc_pa_raw_joint": -0.048342164476157024,
            "Frozen_A": 0.3505318528121748
          },
          "design_weighted": {
            "norm_TTA": -0.06283269245946599,
            "inc_zero": 0.004818255515242953,
            "inc_pa_Aimage": 0.0,
            "inc_pa_wrong_joint": -0.042299162814245506,
            "inc_pa_Bimage": 0.04410782946213132,
            "inc_pa_FA_FB_mean": 0.012249123508421889,
            "inc_pa_FAB_FB_mean": 0.012249123508421889,
            "inc_pa_raw_joint": -0.06283269245946599,
            "Frozen_A": 0.3623070092674716
          }
        },
        "retained_development_candidate": false
      }
    },
    "disposition": "frozen_entry_prerequisite_FAIL_stop_entry_keep_R145",
    "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
  },
  "TTA_run": false,
  "source_calibration_run": false,
  "native_audit": {
    "status": "PASS",
    "n": 1798,
    "new_backfilled": 166,
    "max_CPU_full_reader_error": 9.738885988763712e-07,
    "max_graph_weight_error": 0.0,
    "max_CPU_position_MLP_error": 7.62939453125e-06,
    "max_engineering_CPU_B_interpolation_error": 0.0,
    "max_B_native_to_exactsample_distance": 0.0,
    "all_parent_predictions_and_remaining_noB_exact": true,
    "source_parameters_updated": 0,
    "newlateB_point_features_A_only": true,
    "no_new_B_PointNet_required": true,
    "all_actual_pointAdapter_input_hashes_exact_originalA": true,
    "all_positions_exact_R146": true,
    "fullTransformer_PointNet_not_CPU_recomputed": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "task_values": 116870,
    "max_error": 0.0,
    "all_negative_unknown_noB_retained": true
  },
  "runtime": {
    "n": 1798,
    "backfilled": 166,
    "new_full_forwards": 996,
    "fresh_backbone_calls": 0,
    "newlateB_point_features": "original_A_only_never_average_B",
    "max_completeA_bridge": 0.0,
    "parameters_updated": 0,
    "GT_read": false,
    "FINAL_access": false
  },
  "source_weight_training": false,
  "FINAL_access": false
}

Engineering18inputs(6new/6oldB/6noB), sourceweightsfingerprints/completeAbridge; nativeCPU independentlyrebuildsBprojection/permutation/visibilitygraphs, positionMLP/sixpositions, finalnativehead/livecompetition fromsavedfeatures; actualpointAdapterinputhashesoriginalA andpositionsR146exact. FullTransformer/PointNet areNOTindependentlyCPUrecomputed. NoUpchanges/newsourceweighttraining/expert/architecture/FINAL. Predictionsseal beforetaskGTscore; allscalarsindependentlyrecomputed. Incrementalresearchcosts reuseoldbackbones/registration/prefixes/parentoutputs andarenotcoldstartlatency. UniqueGPU/hard09:00BeijingNov18guard, scoped2GiBpayload/12GiBfreefloor/1GiBsafety, noolddata/results/weightsdeleted. SourceFpointκwasNOTsilentlyreusedforanApointTTAreader: no newC/teacher/updatetargetwasconstructed afterfailedfrozenprerequisite.


## Postseal query equality and interaction

{
  "status": "POSTSEAL_DESCRIPTIVE_ONLY",
  "GT_scores_or_models_not_rerun": true,
  "does_not_choose_outputs_or_parameters": true,
  "all1798_denominators_preserved": true,
  "rows": [
    {
      "comparison": "raw_Apoints_jointB_vs_originalA",
      "method": "pa_raw_joint",
      "control": "Frozen_A",
      "n": 166,
      "query_changed": 1,
      "IoU_better": 0,
      "IoU_worse": 0,
      "whole_cohort_weighted_contribution_pp": 0.0
    },
    {
      "comparison": "normalized_jointB_vs_normalized_Aimage",
      "method": "pa_zero",
      "control": "pa_Aimage",
      "n": 166,
      "query_changed": 0,
      "IoU_better": 0,
      "IoU_worse": 0,
      "whole_cohort_weighted_contribution_pp": 0.0
    },
    {
      "comparison": "normalized_jointB_vs_wrong_binding",
      "method": "pa_zero",
      "control": "pa_wrong_joint",
      "n": 166,
      "query_changed": 3,
      "IoU_better": 0,
      "IoU_worse": 2,
      "whole_cohort_weighted_contribution_pp": -0.042299162814247476
    },
    {
      "comparison": "position_effect_Apoints",
      "method": "pa_zero",
      "control": "pa_raw_joint",
      "n": 166,
      "query_changed": 30,
      "IoU_better": 6,
      "IoU_worse": 14,
      "whole_cohort_weighted_contribution_pp": -0.06283269245946903
    },
    {
      "comparison": "position_effect_mixed_points",
      "method": "bf_zero",
      "control": "bf_raw_joint",
      "n": 166,
      "query_changed": 30,
      "IoU_better": 8,
      "IoU_worse": 7,
      "whole_cohort_weighted_contribution_pp": 0.06048783348040022
    },
    {
      "comparison": "Apoints_vs_mixed_after_position",
      "method": "pa_zero",
      "control": "bf_zero",
      "n": 166,
      "query_changed": 61,
      "IoU_better": 21,
      "IoU_worse": 17,
      "whole_cohort_weighted_contribution_pp": 0.004818255515241358
    }
  ],
  "unused_B_interpolation_zero_fields_in_inherited_auditor_are_placeholders_not_new_checks": true,
  "actual_new_input_check": "allPointAdapterforward_input_hashesmatchoriginalA, no newBbackbone interpolation feedsreader"
}

The R146 sequentialpath doesnot imply removing pointmixturemust improve thefinalreader: the positioneffect changes withthatentry. This fixedApoint-versusmixedpoint comparison checks thisinteraction. EqualIoUmeans do notbythemselves implyequalqueries; actual querychangesarelistedabove. No newruntimegate orGT-selectedoutput isintroduced. R147 doesnotre-runBPointNet/interpolation; inheritedunusedzero auditfields arenotnew checks. The applicableinputproof is the actualPointAdapterA-inputhashcheck.
