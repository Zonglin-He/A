# R146 — Late observation backfill with fixed R145 reader and update

## Decision

**do_not_adopt_backfilled_TTA_keep_R145**. All1798 exposed ordinary inputs and sixdirections retain originalweights/physicalclusters/negative comparisons. This is development, not independent confirmation. The >=2pp goal is evaluated against intact complete Frozen-A13.394463%, never the weakerreset baseline or a GT diagnostic.

## Actual difference and previous negative

Only166originalmissing-B inputs(83physicalframes) use the existingR132 firstgeometry-qualified rank2/3 selection; originalavailableB isneverreplaced. Samegeometryregistration thresholds/raw1024Bfeatures/validity checks. Original1632outputs andevidence exactlyinheritR145. Remaining868missingBinputs retaincompleteFrozenA. No B-selection search, newregistration/model/expert/sourceweighttraining/architecture/A256 change.

R132B3 previously hadweightedupdate-.0020pp versusownzero underreset/fixedlate/absoluteprobabilitysets; thatnegative remains. R146doesnotclaim thisselectionpolicy isnew. It testsit withthe retainedintact/livejointimage-reader/loggap/R145positionalconditioning andsameUp4896 adaptation. RawBPointNet caches areverified againstexactR132 seals andfreshsamecheckpoint engineeringbackbones; oldreset teacher/update predictionsare neverused.

## Fixed source calibration and update

ExactR145source-own kappas andR144Hsource remainfixed. For166newlyBavailable inputs computeq10 oftheexactoriginalA16384sample; samefixedHsource-q10 reference changesonlypos_embed/sixquerycenterZ. Physicalxyz/graphs/rawPointNet/boxescopy remainoriginal. Thisisnot a certifiedgroundplane. Late-B selection changescalibrationconditions; thereisno targetcoverageguarantee andno targetlabel recalibration.

Samepartial(S,C)+withinC frozenBimageKL+source-adaptedFfeaturekeep1, Up4896only,3steps/rho0/9Armijo, sourceinitializationeachinput. C usesfrozenAimage/Bimage groupMEANloggap union; ranktarget usesBimage groupMAXconditionalC. Wrongconditional weights permuteonlywithinC/equalquerymultiplicity. Frozenwrongbindings retain images/visibility andpermutephysicalassociations/inversereversegraph. Neitherteacher norwrongcontrol certifies entityidentity.

## Strong incremental no-new-update controls

AllR145controls retained. bf_* frozenarms use newBwhereavailable. inc_* controls keep the OLD1632parentR145TTAoutputs, andusezero/direct/Bimage/twofixedhalfmeans/wrongTTA ononlynew166. Thus inc_zero andinc_direct are NOT globallyno-update systems: theyisolate theadditionalupdate benefit withoutdiscarding earlieradaptation. bf_TTA main inheritsoldcorrectTTA andaddsnewcorrectTTA. bf_wrong_TTA inheritsoldwrongTTA andaddsnewwrong; inc_wrong_TTA keepsoldcorrectTTA andchangesonlytheaddedfits. No perinput/directionbestoutput selection.

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

{
  "primary": "bf_TTA",
  "same_R142_loss_steps_and_Up": true,
  "R144_sourceonly_positionformula_fixed": true,
  "only166_original_missingB_backfilled": true,
  "source_kappas_exact_R145": true,
  "incremental_controls_preserve_old1632_R145TTA": true,
  "FINAL_access": false,
  "no_target_GT_numeric_parameter_or_per_direction_choice": true,
  "exposed_development_not_independent_confirmation": true,
  "endpoints": {
    "bf_TTA": {
      "comparisons": {
        "raw": {
          "norm_TTA": -0.10820078829507551,
          "inc_zero": -0.060176899436093834,
          "inc_direct": 0.005562407802926828,
          "inc_Bimage": -0.051585790164008216,
          "inc_FA_FB_mean": -0.027906589793362357,
          "inc_FAB_FB_mean": -0.027906589793362357,
          "inc_wrong_TTA": -0.06715607862388318,
          "bf_zero": 0.024892058898146274,
          "bf_direct": 0.18824150360352654,
          "bf_wrong_TTA": 0.06308338591946949,
          "bf_Bimage": 0.03838657021995928,
          "Frozen_A": 0.2906732289932563,
          "Frozen_F": 0.08786009436842868
        },
        "design_weighted": {
          "norm_TTA": -0.11334378514711929,
          "inc_zero": -0.04569283717241035,
          "inc_direct": -0.02852425607741438,
          "inc_Bimage": -0.05069699325594179,
          "inc_FA_FB_mean": -0.027318710129998536,
          "inc_FAB_FB_mean": -0.027318710129998536,
          "inc_wrong_TTA": -0.053722577295783225,
          "bf_zero": 0.03419707237108405,
          "bf_direct": 0.14759564450155338,
          "bf_wrong_TTA": 0.07948441706777842,
          "bf_Bimage": 0.035486448103306145,
          "Frozen_A": 0.3117959165798183,
          "Frozen_F": 0.1253981404866078
        }
      },
      "retained_development_candidate": false
    }
  },
  "disposition": "do_not_adopt_backfilled_TTA_keep_R145",
  "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
}

Mainmustbeat coherentparentR145 andincrementalzero/direct/Bteacher/bothmeans/matchedwrong inbothraw anddesignweightedmeans. Apositivefrozenablation isreported separatelyanddoesnotretroactivelypass a failedTTAprimary. AllremainingnoB andallnegativeinputs stayin the1798denominator.

## Positive and negative contribution

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| bf_TTA | Frozen_A | 135 | 133 | 0.311795916579818 | 0.053304207197498364 | -0.04909164122994028 | 140 | 225 | 21 |
| bf_TTA | norm_TTA | 18 | 31 | -0.1133437851471212 | 0.014775377046733635 | -0.040551159847957464 | 29 | 44 | 8 |
| bf_TTA | inc_zero | 14 | 11 | -0.04569283717241082 | 0.014775377046733635 | -0.033029320825999416 | 19 | 23 | 4 |
| bf_TTA | inc_direct | 14 | 24 | -0.02852425607741505 | 0.024006698805007665 | -0.040551159847957464 | 29 | 34 | 6 |
| bf_TTA | inc_Bimage | 10 | 28 | -0.05069699325594243 | 0.015709610914756932 | -0.02352439559704246 | 27 | 35 | 5 |
| bf_TTA | inc_FA_FB_mean | 12 | 21 | -0.02731871013000031 | 0.015709610914756932 | -0.02352439559704246 | 24 | 31 | 3 |
| bf_TTA | inc_FAB_FB_mean | 12 | 21 | -0.02731871013000031 | 0.015709610914756932 | -0.02352439559704246 | 24 | 31 | 3 |
| bf_TTA | inc_wrong_TTA | 15 | 11 | -0.0537225772957858 | 0.014775377046733635 | -0.023562070965277095 | 20 | 24 | 6 |
| bf_TTA | bf_zero | 66 | 58 | 0.034197072371080744 | 0.030260612572096602 | -0.033029320825999416 | 87 | 116 | 9 |
| bf_TTA | bf_direct | 100 | 110 | 0.14759564450155072 | 0.043932121024601284 | -0.040551159847957464 | 130 | 184 | 19 |
| bf_TTA | bf_wrong_TTA | 60 | 57 | 0.0794844170677747 | 0.043932121024601284 | -0.027670598778255808 | 86 | 109 | 13 |
| bf_zero | norm_joint_zero | 16 | 26 | -0.06765094797471038 | 0.033029320825999416 | -0.040551159847957464 | 28 | 39 | 8 |
| bf_zero | bf_Aimage | 7 | 4 | 0.07413858720838414 | 0.04124910190536532 | -0.007764959404338266 | 10 | 11 | 4 |
| bf_zero | bf_wrong_joint | 16 | 8 | 0.17158410590014728 | 0.053304207197498364 | -0.0018507138090753316 | 21 | 24 | 6 |
| bf_zero | bf_raw_joint | 57 | 57 | 0.1295013045219729 | 0.053304207197498364 | -0.036117173255342604 | 92 | 106 | 18 |
| bf_Bimage | bf_wrong_B | 22 | 13 | 0.07139276849696459 | 0.031481163809365754 | -0.012802773570742389 | 28 | 34 | 5 |

Counts describe candidateboxIoU changes, not independentlyverifiedphysicalidentity. New166subgroup rowsaredescriptive; mainrankings usewholecohort sixdirectionmeans. Sharedframes/recordings andsamephysicalobservations acrosssources arenotindependentreplicates; no singleDrone scene populationCI.

## Postseal ranking and coverage diagnostics

| source | target | n | new_B | eligible | recoverable_zero_errors | recoverable_with_good25_in_S | recoverable_with_good25_in_C | changed_by_new_TTA | zero_best_already | best_outside_S | best_outside_C | C_internal_order |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | 357 | 33 | 211 | 202 | 80 | 70 | 4 | 0 | 2 | 0 | 2 |
| drone | waymo | 176 | 12 | 62 | 71 | 16 | 9 | 0 | 0 | 0 | 0 | 0 |
| quad | drone | 366 | 38 | 192 | 278 | 146 | 68 | 3 | 0 | 0 | 3 | 0 |
| quad | waymo | 176 | 12 | 62 | 107 | 30 | 16 | 2 | 0 | 2 | 0 | 0 |
| waymo | drone | 366 | 38 | 192 | 178 | 80 | 41 | 12 | 2 | 0 | 4 | 6 |
| waymo | quad | 357 | 33 | 211 | 154 | 73 | 63 | 4 | 0 | 2 | 0 | 2 |

GTisusedonlyafterpredictionseal/nativeaudit forscalarsanddescription; itdoesnotchooseB,C,targets,updates,parametersoroutputs. Good25meansboxquality, not entityidentity.

## Measurement, resources and failures

{
  "decision": {
    "primary": "bf_TTA",
    "same_R142_loss_steps_and_Up": true,
    "R144_sourceonly_positionformula_fixed": true,
    "only166_original_missingB_backfilled": true,
    "source_kappas_exact_R145": true,
    "incremental_controls_preserve_old1632_R145TTA": true,
    "FINAL_access": false,
    "no_target_GT_numeric_parameter_or_per_direction_choice": true,
    "exposed_development_not_independent_confirmation": true,
    "endpoints": {
      "bf_TTA": {
        "comparisons": {
          "raw": {
            "norm_TTA": -0.10820078829507551,
            "inc_zero": -0.060176899436093834,
            "inc_direct": 0.005562407802926828,
            "inc_Bimage": -0.051585790164008216,
            "inc_FA_FB_mean": -0.027906589793362357,
            "inc_FAB_FB_mean": -0.027906589793362357,
            "inc_wrong_TTA": -0.06715607862388318,
            "bf_zero": 0.024892058898146274,
            "bf_direct": 0.18824150360352654,
            "bf_wrong_TTA": 0.06308338591946949,
            "bf_Bimage": 0.03838657021995928,
            "Frozen_A": 0.2906732289932563,
            "Frozen_F": 0.08786009436842868
          },
          "design_weighted": {
            "norm_TTA": -0.11334378514711929,
            "inc_zero": -0.04569283717241035,
            "inc_direct": -0.02852425607741438,
            "inc_Bimage": -0.05069699325594179,
            "inc_FA_FB_mean": -0.027318710129998536,
            "inc_FAB_FB_mean": -0.027318710129998536,
            "inc_wrong_TTA": -0.053722577295783225,
            "bf_zero": 0.03419707237108405,
            "bf_direct": 0.14759564450155338,
            "bf_wrong_TTA": 0.07948441706777842,
            "bf_Bimage": 0.035486448103306145,
            "Frozen_A": 0.3117959165798183,
            "Frozen_F": 0.1253981404866078
          }
        },
        "retained_development_candidate": false
      }
    },
    "disposition": "do_not_adopt_backfilled_TTA_keep_R145",
    "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
  },
  "preparation": {
    "n": 1798,
    "unchanged": 1632,
    "effective_B": 930,
    "remaining_noB": 868,
    "backfilled": 166,
    "new_scope_qualifies": 166,
    "new_physical_frames": 83,
    "new_sampledraws": 83,
    "max_seed_bridge_error": 3.902223783196999e-06,
    "old_B_not_replaced": true,
    "GT_read": false,
    "source_kappas_fixed_R145": true,
    "new_source_training": false,
    "FINAL_access": false
  },
  "TTA_preparation": {
    "exact_parent_evidence_reused": 1632,
    "n": 1798,
    "total_eligible": 930,
    "fresh_evidence": 166,
    "new_eligible": 166,
    "new_effective_wrong": 161,
    "new_partial_active": 164,
    "new_C_singleton": 5,
    "GT_read": false,
    "new_source_calibration": false,
    "kappas_exact_R145": true,
    "source_kappas_sha256": "1c9066c93e4ea7bd70b2f57a585ca659f6574d45f57a044914b0c59e1a4b6ba6"
  },
  "source_calibration_reused_exact_R145": {
    "status": "LOCKED_SOURCE_ONLY_IMAGE_READER",
    "quantile": 0.9,
    "source_eligible_counts": {
      "waymo": 22,
      "drone": 18,
      "quad": 37
    },
    "kappa": {
      "waymo": 2.29706174619496,
      "drone": 1.4053959552580049,
      "quad": 0.6205128759145743
    },
    "source_labels_used": true,
    "target_GT_parameter_choice": false,
    "source_weight_training": false,
    "source_reader": "R144positional_reference_Fpoints_Aimage_and_Fpoints_Bimage_full_native_competition",
    "source_prediction_seal_sha256": "beffcf4dc2fbf0610dc725918210be04c952b13f74733363d0f7cfbb2efa3a36",
    "FINAL_access": false,
    "utc": "2026-11-17T19:08:32.174787+00:00"
  },
  "image_audit": {
    "status": "PASS",
    "n": 1798,
    "new_backfilled": 166,
    "max_CPU_full_reader_error": 1.104635878101945e-06,
    "max_graph_weight_error": 0.0,
    "max_CPU_position_MLP_error": 7.62939453125e-06,
    "max_engineering_CPU_B_interpolation_error": 0.0,
    "max_B_native_to_exactsample_distance": 0.0,
    "all_parent_predictions_and_remaining_noB_exact": true,
    "source_parameters_updated": 0,
    "fullTransformer_PointNet_not_CPU_recomputed": true,
    "FINAL_access": false
  },
  "TTA_audit": {
    "status": "PASS",
    "n": 1798,
    "fit_count": 327,
    "max_CPU_probability_error": 5.681276884494935e-07,
    "max_state_error": 0.0,
    "max_Ffeature_error": 1.430511474609375e-06,
    "max_objective_error": 4.656617313969491e-10,
    "full_encoder_gradients_not_CPU_regenerated": true,
    "evidence_C_target_and_direct_reconstructed": true,
    "original_rho0_3steps_Up4896_and_source_only_initialization": true,
    "unchanged_parent1632_predictions_and_evidence_exact": true,
    "incremental_no_NEWupdate_controls_exact": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "task_values": 88102,
    "max_error": 0.0,
    "all_negative_noB_and_identitywrong_kept": true
  },
  "runtime": {
    "image": {
      "n": 1798,
      "backfilled": 166,
      "new_full_forwards": 1162,
      "fresh_backbone_calls": 0,
      "max_completeA_bridge": 0.0,
      "max_engineering_rawB_bridge_error": 0.0,
      "parameters_updated": 0,
      "GT_read": false,
      "FINAL_access": false
    },
    "method": {
      "n": 1798,
      "fits": 327,
      "GT_read": false,
      "primary": "bf_TTA",
      "source_kappas_sha256": "1c9066c93e4ea7bd70b2f57a585ca659f6574d45f57a044914b0c59e1a4b6ba6",
      "FINAL_access": false
    }
  },
  "parameter_updates": "same Up4896 on only new late-B cases; old1632 predictions exactparent reuse",
  "source_weight_training": false,
  "FINAL_access": false
}

EngineeringfreshBbackbone/interpolation/sample/projection/position/completeAbridge checks precedeformal. IndependentCPUauditsrebuildphysicalBgraphs/fixedpositionMLP/nativefinalmatching/livecompetition fromsavedfeatures andupdate deltas/objectives/featurekeep/directmeans. Actualengineeringfinite differences checkthefull differentiablepath. FullTransformer/PointNet/backprop areNOTfullyregeneratedCPU. Oldcompatiblefitsreuseprevioussealedandauditedstates; all1798parent sampleandevidencefiles independentlymatchpinnedoriginalR145seals. No claimofnewrerunning. Predictionsseal beforetaskGTscore; scalarreadback isindependent.

Firstengineeringattemptfailed theinherited18GiBspacepreflight BEFOREmodelinstantiation/newpredictions. Measured166case/332maximumfit scopebudgetwaslocked beforefirstactualmodel, with3GiBnewpayloadcap/12GiBfreefloor/1GiBsafety; olddata/weights/resultswereneverdeleted. Failedattempt/source retainedlocally andreceipt/costexported. Evidenceattempt001 wroteall1798arraysbutfailedwhile serializingaNumPyintegercounter; castingcounters toPythonintfixedmetadataonly. Independentrepairauditconfirmedall1798arrays unchanged beforeanygradient; bothfailedattemptandreceiptretained. Alljobcosts areincrementalresearch, notcoldstartdeploylatency: oldBPointNet/registration, originalprefixes/routing and1632results are reused, notfree. No newmodel/training/taskscoring after09:00BeijingNov18. Rawmedia/descriptions/coordinates/IDs/features/weights/keys/thirdpartysnapshots excluded.

## Pre-task-score coverage metadata

The following fixed descriptive statistics were computed beforeR146taskIoUscoring, fromsealedfrozen/evidence arrays only. They do notcontrolupdates orchooseanyinput. B2/B3selection maychange evidencequality despiteunchangedgeometry/sourceκ.
{
  "status": "PRE_TASK_SCORE_DESCRIPTIVE_ONLY",
  "rows": [
    {
      "source": "drone",
      "target": "quad",
      "kind": "original_B",
      "n": 178,
      "valid_fraction": 0.7077444698033708,
      "supported_groups": 155.2191011235955,
      "retained_groups": 32.30898876404494,
      "retained_fraction": 0.19960241279559798,
      "conditional_teacher_entropy": 2.8489191703052286,
      "teacher_top_mass": 0.15064382824820594,
      "teacher_group_equals_zero_group": 0.550561797752809,
      "zero_group_in_C": 0.6853932584269663
    },
    {
      "source": "drone",
      "target": "quad",
      "kind": "late_B",
      "n": 33,
      "valid_fraction": 0.5717625473484849,
      "supported_groups": 113.78787878787878,
      "retained_groups": 21.272727272727273,
      "retained_fraction": 0.17792197469836962,
      "conditional_teacher_entropy": 2.5774519026308815,
      "teacher_top_mass": 0.1676424486120682,
      "teacher_group_equals_zero_group": 0.3333333333333333,
      "zero_group_in_C": 0.3939393939393939
    },
    {
      "source": "drone",
      "target": "waymo",
      "kind": "original_B",
      "n": 50,
      "valid_fraction": 0.77267578125,
      "supported_groups": 167.12,
      "retained_groups": 38.78,
      "retained_fraction": 0.23551710915796062,
      "conditional_teacher_entropy": 2.824019932032365,
      "teacher_top_mass": 0.18354147533366486,
      "teacher_group_equals_zero_group": 0.44,
      "zero_group_in_C": 0.52
    },
    {
      "source": "drone",
      "target": "waymo",
      "kind": "late_B",
      "n": 12,
      "valid_fraction": 0.7858072916666666,
      "supported_groups": 170.83333333333334,
      "retained_groups": 20.333333333333332,
      "retained_fraction": 0.11365445551899912,
      "conditional_teacher_entropy": 2.40473969187055,
      "teacher_top_mass": 0.2625316013554217,
      "teacher_group_equals_zero_group": 0.5,
      "zero_group_in_C": 0.5
    },
    {
      "source": "quad",
      "target": "drone",
      "kind": "original_B",
      "n": 154,
      "valid_fraction": 0.6961051643668831,
      "supported_groups": 151.53896103896105,
      "retained_groups": 9.662337662337663,
      "retained_fraction": 0.06640814540310849,
      "conditional_teacher_entropy": 1.8922752008124375,
      "teacher_top_mass": 0.27596770794730235,
      "teacher_group_equals_zero_group": 0.38961038961038963,
      "zero_group_in_C": 0.7337662337662337
    },
    {
      "source": "quad",
      "target": "drone",
      "kind": "late_B",
      "n": 38,
      "valid_fraction": 0.6605417351973685,
      "supported_groups": 152.6315789473684,
      "retained_groups": 10.552631578947368,
      "retained_fraction": 0.07122176702818724,
      "conditional_teacher_entropy": 1.8652129642409825,
      "teacher_top_mass": 0.3208212218234605,
      "teacher_group_equals_zero_group": 0.18421052631578946,
      "zero_group_in_C": 0.7105263157894737
    },
    {
      "source": "quad",
      "target": "waymo",
      "kind": "original_B",
      "n": 50,
      "valid_fraction": 0.77267578125,
      "supported_groups": 106.8,
      "retained_groups": 9.32,
      "retained_fraction": 0.11189238349154892,
      "conditional_teacher_entropy": 1.4438396954005586,
      "teacher_top_mass": 0.44868075203756946,
      "teacher_group_equals_zero_group": 0.22,
      "zero_group_in_C": 0.34
    },
    {
      "source": "quad",
      "target": "waymo",
      "kind": "late_B",
      "n": 12,
      "valid_fraction": 0.7858072916666666,
      "supported_groups": 125.41666666666667,
      "retained_groups": 6.083333333333333,
      "retained_fraction": 0.09677188765805439,
      "conditional_teacher_entropy": 1.046556600173018,
      "teacher_top_mass": 0.5851573591530325,
      "teacher_group_equals_zero_group": 0.3333333333333333,
      "zero_group_in_C": 0.3333333333333333
    },
    {
      "source": "waymo",
      "target": "drone",
      "kind": "original_B",
      "n": 154,
      "valid_fraction": 0.6961051643668831,
      "supported_groups": 167.9935064935065,
      "retained_groups": 31.805194805194805,
      "retained_fraction": 0.1889118589061999,
      "conditional_teacher_entropy": 3.016299896570028,
      "teacher_top_mass": 0.14085160821519815,
      "teacher_group_equals_zero_group": 0.6168831168831169,
      "zero_group_in_C": 0.935064935064935
    },
    {
      "source": "waymo",
      "target": "drone",
      "kind": "late_B",
      "n": 38,
      "valid_fraction": 0.6605417351973685,
      "supported_groups": 165.05263157894737,
      "retained_groups": 33.473684210526315,
      "retained_fraction": 0.20407173388839694,
      "conditional_teacher_entropy": 2.9544936154331047,
      "teacher_top_mass": 0.1613359251015657,
      "teacher_group_equals_zero_group": 0.42105263157894735,
      "zero_group_in_C": 0.9473684210526315
    },
    {
      "source": "waymo",
      "target": "quad",
      "kind": "original_B",
      "n": 178,
      "valid_fraction": 0.7077444698033708,
      "supported_groups": 157.29775280898878,
      "retained_groups": 26.202247191011235,
      "retained_fraction": 0.2353430045846629,
      "conditional_teacher_entropy": 2.675438641001032,
      "teacher_top_mass": 0.188501859056735,
      "teacher_group_equals_zero_group": 0.47191011235955055,
      "zero_group_in_C": 0.6853932584269663
    },
    {
      "source": "waymo",
      "target": "quad",
      "kind": "late_B",
      "n": 33,
      "valid_fraction": 0.5717625473484849,
      "supported_groups": 133.6969696969697,
      "retained_groups": 26.242424242424242,
      "retained_fraction": 0.21640517772008905,
      "conditional_teacher_entropy": 2.663106555283549,
      "teacher_top_mass": 0.17321141064631723,
      "teacher_group_equals_zero_group": 0.5757575757575758,
      "zero_group_in_C": 0.8181818181818182
    }
  ],
  "GT_read": false,
  "no_gate_or_predictor_change": true,
  "conditioning": "existing R145 thresholds retained, late selection covariate shift not guaranteed calibrated"
}


## Postseal measured component path (no predictor changes)

{
  "status": "PASS_POSTSEAL_DESCRIPTIVE_ONLY",
  "new_n": 166,
  "fixed_order_path_not_all_interactions_or_universal_causal_effect": true,
  "old1632_parent_predictions_exact": true,
  "task_GT_scoring_not_repeated": true,
  "new_percase_or_direction_choice": false,
  "rows": [
    {
      "component": "new_point_mixture_before_newRGB_or_position",
      "method": "bf_raw_F_A",
      "control": "Frozen_A",
      "whole_cohort_weighted_contribution_pp": -0.14089943486675818
    },
    {
      "component": "new_RGB_after_point_mixture",
      "method": "bf_raw_joint",
      "control": "bf_raw_F_A",
      "whole_cohort_weighted_contribution_pp": 0.012760653411647572
    },
    {
      "component": "new_position_conditioning_after_newRGB",
      "method": "bf_zero",
      "control": "bf_raw_joint",
      "whole_cohort_weighted_contribution_pp": 0.06048783348040022
    },
    {
      "component": "new_Up_update_after_new_observation",
      "method": "bf_TTA",
      "control": "bf_zero",
      "whole_cohort_weighted_contribution_pp": -0.04569283717241083
    }
  ],
  "main_minus_parent_pp": -0.11334378514711929
}

This is the fixed actual path on the166backfilledinputs, expressed withoriginalwholecohort designweights/denominators. The differences telescope exactlytoR146minusR145 becauseold1632predictions areunchanged. The firstcomparison usesoriginalcompleteA ontheseoriginalnoBcases. It separates rawpointfeature averaging, addedRGBbank, fixedpositionalconditioning andUpupdate alongthispath, butdoesnotmeasureallinteractions orprove badregistration/physicalobjectidentity. Allendpoints were alreadysealedandscalaraudited; no newtaskIoU query, task-basedfilter oroutputswitch isintroduced.
