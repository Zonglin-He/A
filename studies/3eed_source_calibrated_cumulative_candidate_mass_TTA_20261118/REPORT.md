# R148 — Source-calibrated cumulative candidate mass with fixed R145 adaptation

## Decision

**do_not_adopt_cumulative_mass_keep_R145**. All1798 exposedinputs/sixdirections/designweights/physicalclusters andallnegativecomparisons retained. Methodchoice follows exposeddevelopment, notindependentconfirmation. Source-only numericalcalibration doesNOTmake theoverallresearch targetblind. Parent>=2pp overintactFrozenA13.394463% remainsaseparategoal, notcountedfromGTdiagnostic/weakerreset.

## Fixed versus changed

SameR145intactthreepreparedsource models/originalA256/pointF1024/actualRGB A/Bimages/physicalgraphs/sourceArole-routing/R144positionreference/fullnativefinalmatching andlivecompetition. Original764B used,1034originalnoB remainexactcompleteFrozenA. R146lateB166 andR147Apointentry wereNOTadopted; theirallnegativepredictions remainreferencecolumns, neverGT-chosen perinput. No newmodel/PointNet/media/view/expert/sourceweighttraining/architecture/regression/EMA/updateposition.

The ONLYscientificchange isCconstruction. WithinoriginalsupportedgroupsS, normalizegroupMEANcandidateprobabilities; take smallestdescendingprefix whosecumulativeconditionalmass>=q_source, includingallprobabilitytiesatboundary; unionA/Bsets. SameR145partial(S,C)+KL(frozenBgroupMAX|C || adaptedjointgroupMAX|C)+source-adaptedFfeaturekeep1, Up4896only/3steps/rho0/9Armijo/sourceinitializationeachinput. Assignment/S/teacherfunctionalform remainunchanged; C andteacher fixedbeforefit. Entirelyflatgroupmasses canretainallS; neverforcedelete. Thenpartialtermiszero; existingBconditionalKLmaystillbenonzero, notclaimallunknownmeanswholelosszero.

## Source-only calibration and relation to APS

Same313sealedR145nativeimage-reader sourceinputs/77eligiblegood25cases/154correlatedA/Bviews, no newsourceforward. Per-own-sourceq90 is thecumulativeconditionalgroupMEANmass throughfirstsourceAcc25-capablegroup. Sameq=.9 andsourcequalificationasbefore; noalpha/temperature/RAPS/randomization/grid. Thisisinspiredby generalizedcumulativequantile sets in [Classification with Valid and Adaptive Coverage](https://arxiv.org/html/2006.02544v1) andthe [official conformal classification implementation](https://github.com/aangelopoulos/conformal_classification/blob/master/conformal.py). Our deterministicweak-positiveperinputgroups/calibrationprotocol differfromformalAPS. Source/targetshift, correlatedviews, checkpoint-exposedsourceVAL andvariablecandidateweakgroups precludeformalconformal targetcoverage claims. No thirdpartycode snapshot redistributed.

{
  "status": "SOURCE_ONLY_CUMULATIVE_MASS_Q90_LOCKED",
  "q90": {
    "waymo": 0.8353701879195964,
    "drone": 0.5523609791217727,
    "quad": 0.8846554024312622
  },
  "source_counts": {
    "waymo": 22,
    "drone": 18,
    "quad": 37
  },
  "summary": [
    {
      "source": "waymo",
      "source_inputs_eligible": 22,
      "correlated_view_cases": 44,
      "q90_cumulative": 0.8353701879195964,
      "empirical_view_good25_covered_fraction": 0.9090909090909091,
      "mean_retained_group_fraction": 0.18250962440316654,
      "mean_old_loggap_retained_group_fraction": 0.12259164105179368,
      "mean_top_mass": 0.1567576671666018
    },
    {
      "source": "drone",
      "source_inputs_eligible": 18,
      "correlated_view_cases": 36,
      "q90_cumulative": 0.5523609791217727,
      "empirical_view_good25_covered_fraction": 0.9166666666666666,
      "mean_retained_group_fraction": 0.2072399597701343,
      "mean_old_loggap_retained_group_fraction": 0.16523213374506887,
      "mean_top_mass": 0.06400151654815558
    },
    {
      "source": "quad",
      "source_inputs_eligible": 37,
      "correlated_view_cases": 74,
      "q90_cumulative": 0.8846554024312622,
      "empirical_view_good25_covered_fraction": 0.9594594594594594,
      "mean_retained_group_fraction": 0.2121182614284815,
      "mean_old_loggap_retained_group_fraction": 0.020181980024330186,
      "mean_top_mass": 0.4692310491393946
    }
  ],
  "source_labels_used": true,
  "target_GT_choice": false,
  "source_weight_training": false,
  "new_models": 0,
  "no_formal_target_conformal_guarantee": true,
  "same_R145_reader": true,
  "source_prediction_seal_sha256": "beffcf4dc2fbf0610dc725918210be04c952b13f74733363d0f7cfbb2efa3a36",
  "utc": "2026-11-17T20:36:29.861278+00:00"
}

The groupMEANleader isalwaysretained; groupMAXteacherleader neednotbeidentical mathematically. InTHIS764inputcohort, bothold/newC retainallBgroupMAXleaders andteacherTopCchanges0. Thus thisexperiment testsfalseexclusion/conditionalmass/gradientdistribution, notnewteacheridentityinformation. No claimthatwideningC discoverscorrecttargets.

## Evidence prerequisite and compatible reuse

{
  "n": 1798,
  "eligible": 764,
  "changed_C": 741,
  "compatible_parent": 1057,
  "new_correct_fits": 741,
  "new_wrong_fits": 741,
  "new_C_equal_S": 1,
  "partial_active": 763,
  "GT_read": false,
  "source_weight_training": false,
  "new_models": 0,
  "summary": [
    {
      "source": "drone",
      "target": "quad",
      "n_B": 178,
      "changed_C": 172,
      "mean_old_C_fraction": 0.19960241279559798,
      "mean_new_C_fraction": 0.2545164278227926,
      "old_B_MAX_leader_missing": 0,
      "new_B_MAX_leader_missing": 0,
      "teacher_top_group_changed": 0
    },
    {
      "source": "drone",
      "target": "waymo",
      "n_B": 50,
      "changed_C": 48,
      "mean_old_C_fraction": 0.23551710915796062,
      "mean_new_C_fraction": 0.2602415794559933,
      "old_B_MAX_leader_missing": 0,
      "new_B_MAX_leader_missing": 0,
      "teacher_top_group_changed": 0
    },
    {
      "source": "quad",
      "target": "drone",
      "n_B": 154,
      "changed_C": 154,
      "mean_old_C_fraction": 0.06640814540310849,
      "mean_new_C_fraction": 0.566560329957009,
      "old_B_MAX_leader_missing": 0,
      "new_B_MAX_leader_missing": 0,
      "teacher_top_group_changed": 0
    },
    {
      "source": "quad",
      "target": "waymo",
      "n_B": 50,
      "changed_C": 50,
      "mean_old_C_fraction": 0.11189238349154892,
      "mean_new_C_fraction": 0.5995841099159801,
      "old_B_MAX_leader_missing": 0,
      "new_B_MAX_leader_missing": 0,
      "teacher_top_group_changed": 0
    },
    {
      "source": "waymo",
      "target": "drone",
      "n_B": 154,
      "changed_C": 150,
      "mean_old_C_fraction": 0.1889118589061999,
      "mean_new_C_fraction": 0.23616759028553302,
      "old_B_MAX_leader_missing": 0,
      "new_B_MAX_leader_missing": 0,
      "teacher_top_group_changed": 0
    },
    {
      "source": "waymo",
      "target": "quad",
      "n_B": 178,
      "changed_C": 167,
      "mean_old_C_fraction": 0.2353430045846629,
      "mean_new_C_fraction": 0.3313992596831349,
      "old_B_MAX_leader_missing": 0,
      "new_B_MAX_leader_missing": 0,
      "teacher_top_group_changed": 0
    }
  ],
  "prerequisite_nontrivial": true
}

IndependentCPU reconstructssourceq/targettiecompleteC/teacher/permutation/direct/compatibilitybeforemodel. 741changedC inputs need1482correct/wrongfits;1057compatibleinputs(including1034noB) exactlyreuseR145predictions andearlierauditedfitresults. Correct/wrongteacherweights permuteonlywithinC/equalquerymultiplicity, retainingconditionalmass/entropy/updatebudget. All65R147/R146/R145historicalreferences copiedfrompinnedparentseals andbytechecked.

## Same-information comparisons

Globalzero, newCdirect, frozenBteacher/twocanonicalhalfmeans andwrongconditionalTTA reported. Stronginc_mass_* controls retainoldR145TTAoncompatibleinputs andusezero/direct/Bteacher/means/wrongnewfits onchangedCinputs. Theseincrementalcontrols areNOTgloballyno-update: they preventinheritedoldpositivesfromhidingchangedCupdateharm. PrimarymustbeatcoherentparentR145 plusallstrongsameinfocontrols BOTHraw/designweighted; noeachdirectionpositive orpercase/dirbestoutputrule.

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
| raw | mass_direct | 13.716921227108832 | 26.558414597831185 | 5.48324893792589 |
| raw | mass_wrong_direct | 12.259251149208898 | 23.387590257165954 | 5.110914211685667 |
| raw | mass_TTA | 13.804276342512814 | 26.880798270624695 | 5.48324893792589 |
| raw | mass_wrong_TTA | 13.62297458589404 | 26.515351545486553 | 5.202958605200649 |
| raw | inc_mass_zero | 13.718215732223712 | 26.608900514999842 | 5.249643946003637 |
| raw | inc_mass_direct | 13.697831441741764 | 26.372821234802913 | 5.48324893792589 |
| raw | inc_mass_Bimage | 13.705390982239946 | 26.332232470732958 | 5.48324893792589 |
| raw | inc_mass_FA_FB_mean | 13.60956019682808 | 26.23753550103599 | 5.343192915516926 |
| raw | inc_mass_FAB_FB_mean | 13.635865653097289 | 26.283072841655294 | 5.389878256319915 |
| raw | inc_mass_wrong_TTA | 13.633388765529068 | 26.562036886289544 | 5.202958605200649 |
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
| design_weighted | mass_direct | 13.737278882449862 | 26.965650459289723 | 5.029399635265412 |
| design_weighted | mass_wrong_direct | 12.224173616401814 | 23.436454692436573 | 4.695056867097374 |
| design_weighted | mass_TTA | 13.817991313461796 | 27.35227930922902 | 5.018149647119095 |
| design_weighted | mass_wrong_TTA | 13.65172517264289 | 27.0046363704314 | 4.753600177489358 |
| design_weighted | inc_mass_zero | 13.741090890108099 | 27.091093465574257 | 4.793982350866708 |
| design_weighted | inc_mass_direct | 13.718172605749574 | 26.77928028901691 | 5.029399635265412 |
| design_weighted | inc_mass_Bimage | 13.726921463688074 | 26.798516220753772 | 5.029399635265411 |
| design_weighted | inc_mass_FA_FB_mean | 13.627356514098821 | 26.70494601300616 | 4.89027865861377 |
| design_weighted | inc_mass_FAB_FB_mean | 13.65344737047071 | 26.75048335362547 | 4.936367008664006 |
| design_weighted | inc_mass_wrong_TTA | 13.662197113522646 | 27.05158064698257 | 4.753600177489358 |

{
  "primary": "mass_TTA",
  "same_R142_loss_steps_and_Up": true,
  "R144_sourceonly_positionformula_fixed": true,
  "only_C_builder_changed_to_source_cumulative_mass": true,
  "source_numerical_calibration_only": true,
  "FINAL_access": false,
  "no_target_GT_numeric_parameter_or_per_direction_choice": true,
  "exposed_development_not_independent_confirmation": true,
  "endpoints": {
    "mass_TTA": {
      "comparisons": {
        "raw": {
          "norm_TTA": 0.0023616859636330645,
          "inc_mass_zero": 0.08606061028910261,
          "inc_mass_direct": 0.1064449007710504,
          "inc_mass_Bimage": 0.09888536027286854,
          "inc_mass_FA_FB_mean": 0.19471614568473505,
          "inc_mass_FAB_FB_mean": 0.16841068941552528,
          "inc_mass_wrong_TTA": 0.17088757698374657,
          "norm_joint_zero": 0.08743064429787317,
          "mass_direct": 0.087355115403982,
          "mass_wrong_TTA": 0.18130175661877423,
          "norm_Bimage": 0.09233404634760056,
          "norm_FA_FB_mean": 0.20562312270502758,
          "norm_FAB_FB_mean": 0.1793176664358196,
          "Frozen_A": 0.4012357032519649,
          "Frozen_F": 0.19842256862713725
        },
        "design_weighted": {
          "norm_TTA": -0.0016118534192930412,
          "inc_mass_zero": 0.07690042335369718,
          "inc_mass_direct": 0.09981870771222212,
          "inc_mass_Bimage": 0.09106984977372257,
          "inc_mass_FA_FB_mean": 0.19063479936297512,
          "inc_mass_FAB_FB_mean": 0.1645439429910862,
          "inc_mass_wrong_TTA": 0.1557941999391499,
          "norm_joint_zero": 0.0782780561241978,
          "mass_direct": 0.08071243101193382,
          "mass_wrong_TTA": 0.166266140818907,
          "norm_Bimage": 0.0845715879399549,
          "norm_FA_FB_mean": 0.2016022708812546,
          "norm_FAB_FB_mean": 0.17551141450936392,
          "Frozen_A": 0.42352784830764456,
          "Frozen_F": 0.23713007221443405
        }
      },
      "retained_development_candidate": false
    }
  },
  "disposition": "do_not_adopt_cumulative_mass_keep_R145",
  "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
}

Theseareinternalcontrolled comparisons, notpublishedTTA baselinereproductions. AllremainingnoB/unknown/negativecases stayinoriginaldenominators.

## Improvements and harms

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| mass_TTA | Frozen_A | 118 | 101 | 0.42352784830764784 | 0.053304207197498364 | -0.04909164122994028 | 118 | 182 | 16 |
| mass_TTA | norm_TTA | 16 | 13 | -0.001611853419291434 | 0.01691070741112714 | -0.021990333210120238 | 26 | 27 | 5 |
| mass_TTA | norm_joint_zero | 50 | 42 | 0.07827805612420011 | 0.030260612572096602 | -0.03206973218682594 | 69 | 89 | 7 |
| mass_TTA | mass_direct | 69 | 77 | 0.08071243101193501 | 0.043932121024601284 | -0.03206973218682594 | 93 | 129 | 13 |
| mass_TTA | mass_wrong_TTA | 71 | 50 | 0.16626614081890861 | 0.030260612572096602 | -0.03206973218682594 | 84 | 111 | 9 |
| mass_TTA | norm_Bimage | 77 | 81 | 0.08457158793995745 | 0.03014500654045807 | -0.03205222459287011 | 101 | 139 | 12 |
| mass_TTA | norm_FA_FB_mean | 63 | 58 | 0.20160227088125687 | 0.030260612572096602 | -0.03205222459287011 | 86 | 114 | 11 |
| mass_TTA | norm_FAB_FB_mean | 62 | 59 | 0.17551141450936397 | 0.030260612572096602 | -0.03205222459287011 | 86 | 114 | 11 |
| mass_TTA | inc_mass_zero | 48 | 41 | 0.07690042335369883 | 0.030260612572096602 | -0.03206973218682594 | 68 | 86 | 7 |
| mass_TTA | inc_mass_direct | 69 | 72 | 0.09981870771222393 | 0.043932121024601284 | -0.03206973218682594 | 91 | 126 | 13 |
| mass_TTA | inc_mass_Bimage | 76 | 76 | 0.09106984977372344 | 0.03014500654045807 | -0.03205222459287011 | 99 | 136 | 12 |
| mass_TTA | inc_mass_FA_FB_mean | 61 | 56 | 0.19063479936297872 | 0.030260612572096602 | -0.03205222459287011 | 84 | 111 | 11 |
| mass_TTA | inc_mass_FAB_FB_mean | 60 | 57 | 0.16454394299108582 | 0.030260612572096602 | -0.03205222459287011 | 84 | 111 | 11 |
| mass_TTA | inc_mass_wrong_TTA | 67 | 50 | 0.1557941999391488 | 0.030260612572096602 | -0.03206973218682594 | 82 | 107 | 9 |
| norm_joint_zero | norm_Aimage | 6 | 2 | 0.0805431848822385 | 0.04124910190536532 | -0.0014437455841802508 | 7 | 8 | 4 |
| norm_joint_zero | norm_joint_wrong | 14 | 7 | 0.1514091362954546 | 0.053304207197498364 | -0.0018507138090753316 | 19 | 21 | 5 |
| norm_Bimage | norm_B_wrong | 20 | 11 | 0.049292761492592554 | 0.031481163809365754 | -0.012802773570742389 | 24 | 30 | 5 |

Counts describeboxIoU, notindependentlyverifiedobjectidentity. Keepallchangedgood/badrecords; sharedframes/source directions/descriptions arenotindependentscenereplicates; no singleDrone recordingpopulationCI or largestharmremoval.

## Postseal scope and ranking diagnostics

| source | target | n | eligible | recoverable_zero_errors | recoverable_with_good25_in_S | recoverable_with_good25_in_C | changed | zero_best_already | best_outside_S | best_outside_C | C_internal_order |
|---|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | 357 | 178 | 202 | 67 | 61 | 22 | 1 | 5 | 1 | 15 |
| drone | waymo | 176 | 50 | 71 | 14 | 10 | 3 | 0 | 1 | 1 | 1 |
| quad | drone | 366 | 154 | 276 | 116 | 114 | 20 | 1 | 3 | 0 | 16 |
| quad | waymo | 176 | 50 | 106 | 25 | 19 | 5 | 1 | 2 | 0 | 2 |
| waymo | drone | 366 | 154 | 179 | 65 | 32 | 22 | 0 | 0 | 10 | 12 |
| waymo | quad | 357 | 178 | 153 | 59 | 55 | 20 | 0 | 3 | 0 | 17 |

GTusedonlyafterpredictionseal/nativeaudit forscalarsanddescriptiveanalysis. Good25 isageometricboxproxy, notidentitytruth; allcohortdiagnostics includeoriginalnoB, so theycannotbecomparedwitheligible-onlycountswithoutmatchingdenominators. No GTnumericparameterchoice orGTupdate/warmstart.

## Numerical audit and measured resources

{
  "decision": {
    "primary": "mass_TTA",
    "same_R142_loss_steps_and_Up": true,
    "R144_sourceonly_positionformula_fixed": true,
    "only_C_builder_changed_to_source_cumulative_mass": true,
    "source_numerical_calibration_only": true,
    "FINAL_access": false,
    "no_target_GT_numeric_parameter_or_per_direction_choice": true,
    "exposed_development_not_independent_confirmation": true,
    "endpoints": {
      "mass_TTA": {
        "comparisons": {
          "raw": {
            "norm_TTA": 0.0023616859636330645,
            "inc_mass_zero": 0.08606061028910261,
            "inc_mass_direct": 0.1064449007710504,
            "inc_mass_Bimage": 0.09888536027286854,
            "inc_mass_FA_FB_mean": 0.19471614568473505,
            "inc_mass_FAB_FB_mean": 0.16841068941552528,
            "inc_mass_wrong_TTA": 0.17088757698374657,
            "norm_joint_zero": 0.08743064429787317,
            "mass_direct": 0.087355115403982,
            "mass_wrong_TTA": 0.18130175661877423,
            "norm_Bimage": 0.09233404634760056,
            "norm_FA_FB_mean": 0.20562312270502758,
            "norm_FAB_FB_mean": 0.1793176664358196,
            "Frozen_A": 0.4012357032519649,
            "Frozen_F": 0.19842256862713725
          },
          "design_weighted": {
            "norm_TTA": -0.0016118534192930412,
            "inc_mass_zero": 0.07690042335369718,
            "inc_mass_direct": 0.09981870771222212,
            "inc_mass_Bimage": 0.09106984977372257,
            "inc_mass_FA_FB_mean": 0.19063479936297512,
            "inc_mass_FAB_FB_mean": 0.1645439429910862,
            "inc_mass_wrong_TTA": 0.1557941999391499,
            "norm_joint_zero": 0.0782780561241978,
            "mass_direct": 0.08071243101193382,
            "mass_wrong_TTA": 0.166266140818907,
            "norm_Bimage": 0.0845715879399549,
            "norm_FA_FB_mean": 0.2016022708812546,
            "norm_FAB_FB_mean": 0.17551141450936392,
            "Frozen_A": 0.42352784830764456,
            "Frozen_F": 0.23713007221443405
          }
        },
        "retained_development_candidate": false
      }
    },
    "disposition": "do_not_adopt_cumulative_mass_keep_R145",
    "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
  },
  "source_calibration": {
    "status": "SOURCE_ONLY_CUMULATIVE_MASS_Q90_LOCKED",
    "q90": {
      "waymo": 0.8353701879195964,
      "drone": 0.5523609791217727,
      "quad": 0.8846554024312622
    },
    "source_counts": {
      "waymo": 22,
      "drone": 18,
      "quad": 37
    },
    "summary": [
      {
        "source": "waymo",
        "source_inputs_eligible": 22,
        "correlated_view_cases": 44,
        "q90_cumulative": 0.8353701879195964,
        "empirical_view_good25_covered_fraction": 0.9090909090909091,
        "mean_retained_group_fraction": 0.18250962440316654,
        "mean_old_loggap_retained_group_fraction": 0.12259164105179368,
        "mean_top_mass": 0.1567576671666018
      },
      {
        "source": "drone",
        "source_inputs_eligible": 18,
        "correlated_view_cases": 36,
        "q90_cumulative": 0.5523609791217727,
        "empirical_view_good25_covered_fraction": 0.9166666666666666,
        "mean_retained_group_fraction": 0.2072399597701343,
        "mean_old_loggap_retained_group_fraction": 0.16523213374506887,
        "mean_top_mass": 0.06400151654815558
      },
      {
        "source": "quad",
        "source_inputs_eligible": 37,
        "correlated_view_cases": 74,
        "q90_cumulative": 0.8846554024312622,
        "empirical_view_good25_covered_fraction": 0.9594594594594594,
        "mean_retained_group_fraction": 0.2121182614284815,
        "mean_old_loggap_retained_group_fraction": 0.020181980024330186,
        "mean_top_mass": 0.4692310491393946
      }
    ],
    "source_labels_used": true,
    "target_GT_choice": false,
    "source_weight_training": false,
    "new_models": 0,
    "no_formal_target_conformal_guarantee": true,
    "same_R145_reader": true,
    "source_prediction_seal_sha256": "beffcf4dc2fbf0610dc725918210be04c952b13f74733363d0f7cfbb2efa3a36",
    "utc": "2026-11-17T20:36:29.861278+00:00"
  },
  "preparation": {
    "n": 1798,
    "eligible": 764,
    "changed_C": 741,
    "compatible_parent": 1057,
    "new_correct_fits": 741,
    "new_wrong_fits": 741,
    "new_C_equal_S": 1,
    "partial_active": 763,
    "GT_read": false,
    "source_weight_training": false,
    "new_models": 0,
    "summary": [
      {
        "source": "drone",
        "target": "quad",
        "n_B": 178,
        "changed_C": 172,
        "mean_old_C_fraction": 0.19960241279559798,
        "mean_new_C_fraction": 0.2545164278227926,
        "old_B_MAX_leader_missing": 0,
        "new_B_MAX_leader_missing": 0,
        "teacher_top_group_changed": 0
      },
      {
        "source": "drone",
        "target": "waymo",
        "n_B": 50,
        "changed_C": 48,
        "mean_old_C_fraction": 0.23551710915796062,
        "mean_new_C_fraction": 0.2602415794559933,
        "old_B_MAX_leader_missing": 0,
        "new_B_MAX_leader_missing": 0,
        "teacher_top_group_changed": 0
      },
      {
        "source": "quad",
        "target": "drone",
        "n_B": 154,
        "changed_C": 154,
        "mean_old_C_fraction": 0.06640814540310849,
        "mean_new_C_fraction": 0.566560329957009,
        "old_B_MAX_leader_missing": 0,
        "new_B_MAX_leader_missing": 0,
        "teacher_top_group_changed": 0
      },
      {
        "source": "quad",
        "target": "waymo",
        "n_B": 50,
        "changed_C": 50,
        "mean_old_C_fraction": 0.11189238349154892,
        "mean_new_C_fraction": 0.5995841099159801,
        "old_B_MAX_leader_missing": 0,
        "new_B_MAX_leader_missing": 0,
        "teacher_top_group_changed": 0
      },
      {
        "source": "waymo",
        "target": "drone",
        "n_B": 154,
        "changed_C": 150,
        "mean_old_C_fraction": 0.1889118589061999,
        "mean_new_C_fraction": 0.23616759028553302,
        "old_B_MAX_leader_missing": 0,
        "new_B_MAX_leader_missing": 0,
        "teacher_top_group_changed": 0
      },
      {
        "source": "waymo",
        "target": "quad",
        "n_B": 178,
        "changed_C": 167,
        "mean_old_C_fraction": 0.2353430045846629,
        "mean_new_C_fraction": 0.3313992596831349,
        "old_B_MAX_leader_missing": 0,
        "new_B_MAX_leader_missing": 0,
        "teacher_top_group_changed": 0
      }
    ],
    "prerequisite_nontrivial": true
  },
  "preparation_audit": {
    "status": "PASS",
    "source_n": 313,
    "target_n": 1798,
    "compatible_parent": 1057,
    "source_q90_independently_reconstructed": true,
    "target_C_teacher_wrong_direct_reconstructed": true,
    "tie_complete_prefix": true,
    "no_target_GT_read": true,
    "new_models": 0,
    "FINAL_access": false
  },
  "resource_profile": {
    "status": "PASS_BEFORE_MODEL",
    "samples": [
      {
        "source": "drone",
        "features_probe_bytes": 674776,
        "state_bytes": 74117,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "quad",
        "features_probe_bytes": 674316,
        "state_bytes": 73969,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "drone",
        "features_probe_bytes": 656851,
        "state_bytes": 74099,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "quad",
        "features_probe_bytes": 550389,
        "state_bytes": 74026,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "drone",
        "features_probe_bytes": 674691,
        "state_bytes": 74090,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "quad",
        "features_probe_bytes": 616638,
        "state_bytes": 74250,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "waymo",
        "features_probe_bytes": 635484,
        "state_bytes": 74055,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "waymo",
        "features_probe_bytes": 620063,
        "state_bytes": 74119,
        "full_adapted_array_uncompressed_bytes": 1179648
      },
      {
        "source": "waymo",
        "features_probe_bytes": 621234,
        "state_bytes": 74228,
        "full_adapted_array_uncompressed_bytes": 1179648
      }
    ],
    "max_fresh_fits": 1482,
    "conservative_estimate_bytes": 1372003426,
    "cap_bytes": 2147483648,
    "observed_free_bytes": 16210014208,
    "probe_indices": [
      0,
      33,
      66,
      99,
      132,
      165,
      198,
      231,
      264,
      297,
      330,
      363,
      396,
      429,
      462,
      495,
      528,
      561,
      594,
      627,
      660,
      693,
      726,
      759,
      792,
      825,
      858,
      891,
      924,
      957,
      990,
      1023
    ],
    "full_engineering_adapted_A_retained": true,
    "formal_full_adapted_A_not_stored_or_fully_compared": true,
    "formal_full_keep_reconstructed_and_native_final_heads_states_objectives_audited": true,
    "no_old_artifact_deleted": true
  },
  "source_audit": {
    "status": "PASS",
    "n": 313,
    "B_inputs": 113,
    "source_views": 626,
    "max_CPU_full_reader_probability_error": 1.0001617029575804e-06,
    "max_independent_B_graph_weight_error": 0.0,
    "max_CPU_position_MLP_error": 1.1444091796875e-05,
    "source_Fpoints_Aimage_Bimage_reader": true,
    "same_qualification_as_R132": true,
    "GT_read": false,
    "fullTransformer_not_CPU_regenerated": true,
    "FINAL_access": false
  },
  "image_audit": {
    "status": "PASS",
    "n": 1798,
    "B_inputs": 764,
    "effective_wrong_B_inputs": 754,
    "max_CPU_full_reader_error": 1.6016652828465539e-06,
    "max_independent_graph_weight_error": 0.0,
    "max_CPU_position_MLP_error": 1.1444091796875e-05,
    "native_A256_noB_and_reused_references_exact": true,
    "source_parameters_updated": 0,
    "fullTransformer_not_independently_CPU_recomputed": true,
    "FINAL_access": false
  },
  "TTA_audit": {
    "status": "PASS",
    "n": 1798,
    "fit_count": 1482,
    "max_CPU_probability_error": 1.4544682016559562e-06,
    "max_state_error": 0.0,
    "max_formal_Ffeature_probe_error": 1.430511474609375e-06,
    "max_engineering_full_Ffeature_error": 0.0,
    "formal_full_adapted_A_not_saved_or_fully_compared": true,
    "full1024point_featurekeep_independently_reconstructed": true,
    "max_objective_error": 4.656618424192516e-10,
    "full_encoder_gradients_not_CPU_regenerated": true,
    "evidence_C_target_and_direct_reconstructed": true,
    "original_rho0_3steps_Up4896_and_source_only_initialization": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "task_values": 134850,
    "max_error": 0.0,
    "all_negative_noB_and_identitywrong_kept": true
  },
  "runtime": {
    "method": {
      "n": 1798,
      "fits": 1482,
      "GT_read": false,
      "primary": "mass_TTA",
      "source_mass_sha256": "be304de4a74e7b035479415af493222f932576439ad3f60517ab04858a900e1b",
      "full_formal_adapted_A_not_stored": true,
      "FINAL_access": false
    }
  },
  "parameter_updates": "only original Up4896, sourceonlyinitialize perinput",
  "source_weight_training": false,
  "FINAL_access": false,
  "ancillary_metadata": {}
}

Engineeringactualfullpathfinite differences/zeroFbridge/sourcehashes plusindependentCPU sourceq/C/head/readout/states/rho/Armijotrace/fullfeaturekeep/objectivechecks. Sourceweights andallhistoricalA256 unchanged. FullTransformer/backprop notCPUregenerated. To staywithinmeasured2GiBscopebudget/12GiBfloor/1GiBsafety, FORMAL stores32fixedpointrowprobesinsteadentireadapted_A arrays, plusallnativeheadfeatures/gradients/deltas/traces. CPU independentlyreconstructsALL1024pointadaptationandfeaturekeep fromexactFinput/sourceUp+delta; allformalprobescompared. Engineeringretainsandcomparesfulladapted_A. DoNOTclaimfullformaladaptedtensorwasstoredorfullycompared. Thisstorageimplementationchangenevermodifiespredictions/loss. Nineoldfits profile beforemodels estimated1,372,003,426bytes; noolddata/results/weightsdeleted.

Firstengineeringattemptfailedstrictstatus-stringcomparison PASSvsPASS_BEFORE_MODEL beforemodel/rootcreation; oldmethod/lock/log/receipt preserved. Exactstatusfixedbeforefirstmodel; sourcecalibration/evidence/mathunchanged. Allfailedattemptcosts reported. SubsequentactualengineeringPASS isseparatefromthatpreflightfailure. PredictionssealbeforenewtaskGTscore; allscalarsindependentlyrecalculated; publicexport excludesrawmedia/words/coordinates/IDs/features/weights/keys/thirdpartysnapshots. Incrementalresearchtimings reuseoldnativeprefixes/camera registration/routing/sourcepredictions andarenotcoldstartdeploymentlatency. UniqueGPU/deadlineguard; no newmodel/training/scoringafter09:00BeijingNov18. FINALuntouched.


## Matched eligible-only scope and gradient diagnostic

{
  "status": "POSTSEAL_DESCRIPTIVE_ONLY",
  "target_models_or_GTgeometry_not_rerun": true,
  "rows": [
    {
      "source": "drone",
      "target": "quad",
      "eligible_n": 178,
      "recoverable_zero_errors_eligible": 91,
      "recoverable_good_in_S": 67,
      "recoverable_old_good_C": 60,
      "recoverable_new_good_C": 61,
      "old_excluded_good_zero": 14,
      "new_excluded_good_zero": 14,
      "old_teacher_good25": 18,
      "new_teacher_good25": 18,
      "teacher_query_changed": 0,
      "changed_C_first_gradient_n": 172,
      "mean_first_gradient_cosine": 0.9265886854581701,
      "min_first_gradient_cosine": -0.6128640703005823
    },
    {
      "source": "drone",
      "target": "waymo",
      "eligible_n": 50,
      "recoverable_zero_errors_eligible": 23,
      "recoverable_good_in_S": 14,
      "recoverable_old_good_C": 7,
      "recoverable_new_good_C": 10,
      "old_excluded_good_zero": 5,
      "new_excluded_good_zero": 5,
      "old_teacher_good25": 16,
      "new_teacher_good25": 16,
      "teacher_query_changed": 0,
      "changed_C_first_gradient_n": 48,
      "mean_first_gradient_cosine": 0.87826192756906,
      "min_first_gradient_cosine": -0.681759658402471
    },
    {
      "source": "quad",
      "target": "drone",
      "eligible_n": 154,
      "recoverable_zero_errors_eligible": 126,
      "recoverable_good_in_S": 116,
      "recoverable_old_good_C": 47,
      "recoverable_new_good_C": 114,
      "old_excluded_good_zero": 6,
      "new_excluded_good_zero": 6,
      "old_teacher_good25": 21,
      "new_teacher_good25": 21,
      "teacher_query_changed": 0,
      "changed_C_first_gradient_n": 154,
      "mean_first_gradient_cosine": 0.5844040214475287,
      "min_first_gradient_cosine": -0.901716106258056
    },
    {
      "source": "quad",
      "target": "waymo",
      "eligible_n": 50,
      "recoverable_zero_errors_eligible": 35,
      "recoverable_good_in_S": 25,
      "recoverable_old_good_C": 13,
      "recoverable_new_good_C": 19,
      "old_excluded_good_zero": 9,
      "new_excluded_good_zero": 9,
      "old_teacher_good25": 7,
      "new_teacher_good25": 7,
      "teacher_query_changed": 0,
      "changed_C_first_gradient_n": 50,
      "mean_first_gradient_cosine": 0.6664676811011019,
      "min_first_gradient_cosine": -0.5599743536376904
    },
    {
      "source": "waymo",
      "target": "drone",
      "eligible_n": 154,
      "recoverable_zero_errors_eligible": 67,
      "recoverable_good_in_S": 65,
      "recoverable_old_good_C": 32,
      "recoverable_new_good_C": 32,
      "old_excluded_good_zero": 0,
      "new_excluded_good_zero": 0,
      "old_teacher_good25": 40,
      "new_teacher_good25": 40,
      "teacher_query_changed": 0,
      "changed_C_first_gradient_n": 150,
      "mean_first_gradient_cosine": 0.8766956033206587,
      "min_first_gradient_cosine": -0.8519770927648176
    },
    {
      "source": "waymo",
      "target": "quad",
      "eligible_n": 178,
      "recoverable_zero_errors_eligible": 73,
      "recoverable_good_in_S": 59,
      "recoverable_old_good_C": 51,
      "recoverable_new_good_C": 55,
      "old_excluded_good_zero": 25,
      "new_excluded_good_zero": 25,
      "old_teacher_good25": 58,
      "new_teacher_good25": 58,
      "teacher_query_changed": 0,
      "changed_C_first_gradient_n": 167,
      "mean_first_gradient_cosine": 0.8940002796948204,
      "min_first_gradient_cosine": -0.43600859707253964
    }
  ],
  "new_outputs_or_runtime_filters": false,
  "same_eligible_denominators_for_old_and_new_C": true,
  "teacher_good25_boxproxy_not_instance_identity": true
}

These useexistingsealedcandidateIoUscalars onlyafterpredictions/scalarsaudited. Allrowsbelowsharethe764eligibleinputs, avoidingmixingeligiblecoveragewithall1798/noBerrorcounts. FirstgradientcosinescompareactualR145/R148initialgradients withidenticalreader/input/sourceinit/budget; onlyC/conditionalmasschanged. Cosineisdescriptive, notGTalignment oradeploymentgate. TeacherIoUthresholdsareboxqualityproxies, notverifiedidentity.
