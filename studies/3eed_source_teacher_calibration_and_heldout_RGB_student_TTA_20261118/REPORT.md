# R149 — Source-own teacher calibration and withheld B-image student

## Decision

**do_not_adopt_Aimage_student_keep_R145**. Same1798 exposedinputs/sixdirections/originaldesignweights/allnegative/noB/UNKNOWNcases andphysicalclusters. The parent2ppgoal overcompleteFrozenA13.394463% remainsaseparategoal. Methodchoice uses exposeddevelopment; thisisnotindependentconfirmation.

## Two predeclared factors and actual collapse

Frozen architecture/sourcepreparedmodels/originalA256/R145positionreference/F1024pointinput/S/C/registeredB/Up4896/3steps/rho0/9Armijo/fullsourceinitialization arefixed. Teacher isfrozenFpoint+BindependentRGB, groupMAXconditionalonoriginalR145C. Studentfinalreadout remainsunitnativebase+livelearnedcompetition; neitherispoint-only, competitionresidual isnotclippedto.2. No extra sourceweights, expert, regression, newview, EMA orbackbonechanges.

Factor1 wasfiniteper-own-source teacher-onlycompetition scaling alpha in{.5,1,2}. Matching-head replays ofsealedsource313nativefeatures were sealedandindependentlycheckedinNumPy BEFOREnewsource-labelselection. Selecteachownsource bymean candidateIoU ofBteacherwithinC onALL113geometryeligiblepairedsourcecases(32W/25D/56Q), noGTgood-casefilter; tiesprefer1. Thisdiffersfromthe77good25casesusedforoldkappa; kappaandCthemselvesunchanged. Itisfiniteposthocsource-onlynumericalcalibration, NOTnewsourceweighttraining. Differentoutgoingdirections useonlytheirownsourcealpha, never targetlabels. Allthree selected1, so stopthisfactorwithoutawidergrid. No targetalpha0.5/2 predictions or targetbestchoice.

Factor2 withholdsBIMAGEfromthestudent,butretainsSAMEfrozenFpointfeatures, whichcontainregisteredBpoints. StudentstillusesphysicalAimagegraph/RGBandthecompletebidirectionalfusion/nativefinalcompetitionreadout. Thus itdoesnotclaimfullyindependentobservations ornewRGBencoder. Bimageisavailableonlyviafrozenteacherduringadaptation, insteadalreadybeinginbothteacherandstudentRGBpaths. Canonicalzeroisexactoldnorm_Aimage; teacherandallqualificationstillreferenceoriginalR145. NoB1034inputsfall backexactcompleteFrozenA.

Becausealpha1wasselectedforallsources,cal_A_TTA==plain_A_TTAexactandcal_AB_TTA==parentR145norm_TTAexact. TheseareALIASES, notindependentreplications. OnlyAimage-studentcorrect/wrong fitsarenew. All75R148historicalarms retainedbyteexact. No parentreruns.

## Same-information controls

ComparewithAimagezero, ABimagezero, calibrateddirect(AandAB), fullfrozenBteacher,twofixedhalfmeans, same-budgetwrongconditionalteacherandR145strongreference. Wrong permutesconditionalweightswithinC among equalduplicate-groupmultiplicities; entropy/support/updatebudgetpreserved, ineffectivepermutationsrecorded. CalibratedB/direct/meansareoriginalteachercontrolswhenalpha1. Theseareinternalcontrolledcomparisons, notexternalpublishedbaselines.

## Source calibration

{
  "status": "SOURCE_OWN_FINITE_CALIBRATION_LOCKED",
  "alpha": {
    "waymo": 1.0,
    "drone": 1.0,
    "quad": 1.0
  },
  "summary": [
    {
      "source": "waymo",
      "alpha": 0.5,
      "n": 32,
      "mean_teacher_C_IoU": 23.47389014069582
    },
    {
      "source": "waymo",
      "alpha": 1.0,
      "n": 32,
      "mean_teacher_C_IoU": 23.47389014069582
    },
    {
      "source": "waymo",
      "alpha": 2.0,
      "n": 32,
      "mean_teacher_C_IoU": 22.99133493319485
    },
    {
      "source": "drone",
      "alpha": 0.5,
      "n": 25,
      "mean_teacher_C_IoU": 13.324603632414721
    },
    {
      "source": "drone",
      "alpha": 1.0,
      "n": 25,
      "mean_teacher_C_IoU": 13.324603632414721
    },
    {
      "source": "drone",
      "alpha": 2.0,
      "n": 25,
      "mean_teacher_C_IoU": 13.324603632414721
    },
    {
      "source": "quad",
      "alpha": 0.5,
      "n": 56,
      "mean_teacher_C_IoU": 31.468108501686014
    },
    {
      "source": "quad",
      "alpha": 1.0,
      "n": 56,
      "mean_teacher_C_IoU": 31.468108501686014
    },
    {
      "source": "quad",
      "alpha": 2.0,
      "n": 56,
      "mean_teacher_C_IoU": 31.468108501686014
    }
  ],
  "counts": {
    "waymo": 32,
    "drone": 25,
    "quad": 56
  },
  "source_labels_only": true,
  "no_target_selection": true,
  "source_weight_training": false,
  "all_eligible_not_GT_prefiltered": true,
  "source_seal_sha256": "22c2722d3242358d0c3cfce46dcdb93955deaf221aff4a71f7ea83b937a104c3",
  "utc": "2026-11-17T21:25:26.800375+00:00"
}

## Whole-cohort task scores

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
| raw | plain_A_TTA | 13.750143472443916 | 26.642244702153093 | 5.528786278545199 |
| raw | plain_A_wrong_TTA | 13.636785551082772 | 26.40863971023084 | 5.294033286439266 |
| raw | cal_A_TTA | 13.750143472443916 | 26.642244702153093 | 5.528786278545199 |
| raw | cal_A_wrong_TTA | 13.636785551082772 | 26.40863971023084 | 5.294033286439266 |
| raw | cal_AB_TTA | 13.801914656549181 | 26.834112929821703 | 5.528786278545199 |
| raw | cal_AB_wrong_TTA | 13.671675192005827 | 26.552496309005473 | 5.294033286439266 |
| raw | cal_A_direct | 13.61923556074858 | 26.46848791677625 | 5.48324893792589 |
| raw | cal_AB_direct | 13.61923556074858 | 26.46848791677625 | 5.48324893792589 |
| raw | cal_B | 13.711942296165214 | 26.471140492958238 | 5.48324893792589 |
| raw | cal_FA_FB_mean | 13.598653219807787 | 26.23753550103599 | 5.343192915516926 |
| raw | cal_FAB_FB_mean | 13.624958676076995 | 26.283072841655294 | 5.389878256319915 |
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
| design_weighted | plain_A_TTA | 13.742629038388102 | 27.024954478713067 | 5.063686987738404 |
| design_weighted | plain_A_wrong_TTA | 13.635455193421494 | 26.80078718246068 | 4.8446748587279735 |
| design_weighted | cal_A_TTA | 13.742629038388102 | 27.024954478713067 | 5.063686987738404 |
| design_weighted | cal_A_wrong_TTA | 13.635455193421494 | 26.80078718246068 | 4.8446748587279735 |
| design_weighted | cal_AB_TTA | 13.81960316688109 | 27.305335032677846 | 5.063686987738404 |
| design_weighted | cal_AB_wrong_TTA | 13.686396172517528 | 27.007052866504143 | 4.8446748587279735 |
| design_weighted | cal_A_direct | 13.643483266302125 | 26.90261572620692 | 5.029399635265412 |
| design_weighted | cal_AB_direct | 13.643483266302125 | 26.90261572620692 | 5.029399635265412 |
| design_weighted | cal_B | 13.733419725521841 | 26.93794211447542 | 5.029399635265411 |
| design_weighted | cal_FA_FB_mean | 13.616389042580542 | 26.70494601300616 | 4.89027865861377 |
| design_weighted | cal_FAB_FB_mean | 13.642479898952432 | 26.75048335362547 | 4.936367008664006 |

{
  "primary": "cal_A_TTA",
  "same_R142_loss_steps_and_Up": true,
  "R144_sourceonly_positionformula_fixed": true,
  "teacher_scale_sourcecal_all1_and_Aimage_student_factor": true,
  "source_numerical_calibration_only": true,
  "FINAL_access": false,
  "no_target_GT_numeric_parameter_or_per_direction_choice": true,
  "exposed_development_not_independent_confirmation": true,
  "endpoints": {
    "cal_A_TTA": {
      "comparisons": {
        "raw": {
          "norm_TTA": -0.05177118410526482,
          "norm_Aimage": 0.08859874890984365,
          "norm_joint_zero": 0.033297774228975285,
          "cal_A_direct": 0.13090791169533667,
          "cal_A_wrong_TTA": 0.11335792136114442,
          "cal_B": 0.038201176278702675,
          "cal_FA_FB_mean": 0.1514902526361297,
          "cal_FAB_FB_mean": 0.1251847963669217,
          "Frozen_A": 0.347102833183067,
          "Frozen_F": 0.14428969855823937
        },
        "design_weighted": {
          "norm_TTA": -0.07697412849298679,
          "norm_Aimage": 0.08345896593274382,
          "norm_joint_zero": 0.0029157810505040516,
          "cal_A_direct": 0.09914577208597741,
          "cal_A_wrong_TTA": 0.1071738449666082,
          "cal_B": 0.009209312866261143,
          "cal_FA_FB_mean": 0.12623999580756085,
          "cal_FAB_FB_mean": 0.10014913943567016,
          "Frozen_A": 0.3481655732339508,
          "Frozen_F": 0.1617677971407403
        }
      },
      "retained_development_candidate": false
    },
    "cal_AB_TTA": {
      "comparisons": {
        "raw": {
          "norm_TTA": 0.0,
          "norm_joint_zero": 0.08506895833424011,
          "cal_AB_direct": 0.1826790958006015,
          "cal_AB_wrong_TTA": 0.13023946454335444,
          "cal_B": 0.0899723603839675,
          "cal_FA_FB_mean": 0.20326143674139452,
          "cal_FAB_FB_mean": 0.17695598047218652,
          "Frozen_A": 0.39887401728833183,
          "Frozen_F": 0.1960608826635042
        },
        "design_weighted": {
          "norm_TTA": 0.0,
          "norm_joint_zero": 0.07988990954349084,
          "cal_AB_direct": 0.1761199005789642,
          "cal_AB_wrong_TTA": 0.13320699436356165,
          "cal_B": 0.08618344135924794,
          "cal_FA_FB_mean": 0.20321412430054764,
          "cal_FAB_FB_mean": 0.17712326792865696,
          "Frozen_A": 0.4251397017269376,
          "Frozen_F": 0.2387419256337271
        }
      },
      "retained_development_candidate": false
    }
  },
  "disposition": "do_not_adopt_Aimage_student_keep_R145",
  "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
}

## Good and bad changes

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| cal_A_TTA | Frozen_A | 116 | 103 | 0.34816557323395003 | 0.053304207197498364 | -0.04909164122994028 | 119 | 180 | 18 |
| cal_A_TTA | norm_TTA | 4 | 6 | -0.0769741284929892 | 0.011308447493548368 | -0.04124910190536532 | 10 | 10 | 5 |
| cal_A_TTA | norm_Aimage | 51 | 45 | 0.08345896593274084 | 0.030260612572096602 | -0.03206973218682594 | 71 | 90 | 7 |
| cal_A_TTA | norm_joint_zero | 53 | 50 | 0.0029157810505023446 | 0.030260612572096602 | -0.04124910190536532 | 75 | 96 | 9 |
| cal_A_TTA | cal_A_direct | 85 | 89 | 0.09914577208597655 | 0.043932121024601284 | -0.04124910190536532 | 108 | 150 | 15 |
| cal_A_TTA | cal_A_wrong_TTA | 46 | 47 | 0.10717384496660595 | 0.043932121024601284 | -0.027670598778255808 | 72 | 86 | 8 |
| cal_A_TTA | cal_B | 79 | 92 | 0.009209312866259617 | 0.034984151162616633 | -0.04124910190536532 | 108 | 147 | 13 |
| cal_A_TTA | cal_FA_FB_mean | 68 | 69 | 0.1262399958075591 | 0.034984151162616633 | -0.04124910190536532 | 98 | 126 | 13 |
| cal_A_TTA | cal_FAB_FB_mean | 67 | 70 | 0.10014913943566621 | 0.034984151162616633 | -0.04124910190536532 | 98 | 126 | 13 |
| norm_Aimage | norm_joint_zero | 2 | 6 | -0.0805431848822385 | 0.0014437455841802508 | -0.04124910190536532 | 7 | 8 | 4 |

Allchangedrecords areexported, noGT-selectedcasefilter. CandidateIoU/Acc25 arebox-qualityproxies, notverifiedinstanceidentity. Sharedphysicalframes/recordings/directions arenotindependentreplications. No per-directiondeploymentwinner, no singleDroneclusterpopulationCI.

## Frozen-input and update interaction decomposition

| aggregation | source | target | frozen_A_minus_AB_pp | update_A_gain_pp | update_AB_gain_pp | update_interaction_pp | total_A_TTA_minus_AB_TTA_pp |
|---|---|---|---|---|---|---|---|
| raw | ALL | ALL | -0.055300974680868364 | 0.08859874890984365 | 0.08506895833424011 | 0.003529790575603542 | -0.05177118410526482 |
| raw | drone | quad | -0.00016958545694301108 | -0.07229537902306582 | -0.07683055662539218 | 0.004535177602326357 | 0.004365592145383346 |
| raw | drone | waymo | 0.0 | -0.011745866552502093 | -0.011745866552502093 | 0.0 | 0.0 |
| raw | quad | drone | 0.0 | 0.18754526080176603 | 0.06618108819292345 | 0.12136417260884258 | 0.12136417260884258 |
| raw | quad | waymo | -0.3391748258205194 | -0.20400904889373628 | -0.10499955292376661 | -0.09900949596996966 | -0.43818432179048905 |
| raw | waymo | drone | -0.007401604688315189 | 0.3806134411713167 | 0.3732118364830015 | 0.007401604688315189 | 0.0 |
| raw | waymo | quad | 0.014940167880563848 | 0.2514840859552727 | 0.26459680143116415 | -0.013112715475891434 | 0.0018274524046724139 |
| design_weighted | ALL | ALL | -0.08054318488223977 | 0.08345896593274382 | 0.07988990954349084 | 0.003569056389252978 | -0.07697412849298679 |
| design_weighted | drone | quad | -0.0001705260463520375 | -0.07269635838265742 | -0.07725668991093393 | 0.004560331528276507 | 0.004389805481924469 |
| design_weighted | drone | waymo | 0.0 | -0.01467445614457219 | -0.01467445614457219 | 0.0 | 0.0 |
| design_weighted | quad | drone | 0.0 | 0.18754526080176603 | 0.06618108819292345 | 0.12136417260884258 | 0.12136417260884258 |
| design_weighted | quad | waymo | -0.49071001064212005 | -0.23291301007606613 | -0.1341866834759866 | -0.09872632660007952 | -0.5894363372421996 |
| design_weighted | waymo | drone | -0.007401604688315189 | 0.3806134411713149 | 0.3732118364829997 | 0.007401604688315189 | 0.0 |
| design_weighted | waymo | quad | 0.015023032083359311 | 0.25287891822666 | 0.2660643621165235 | -0.013185443889863535 | 0.001837588193495776 |

The fixed factorial difference separates changing the frozen student bank from changing its adaptation increment. It is not an independent correctness label or proof of teacher independence; pointF remains shared.

## Postseal diagnostics

| source | target | n | eligible | recoverable_Azero_errors | recoverable_Azero_with_good_C | changed | zero_best_already | best_outside_S | best_outside_C | C_internal_order |
|---|---|---|---|---|---|---|---|---|---|---|
| drone | quad | 357 | 178 | 202 | 60 | 18 | 1 | 4 | 1 | 12 |
| drone | waymo | 176 | 50 | 71 | 7 | 2 | 0 | 0 | 1 | 1 |
| quad | drone | 366 | 154 | 276 | 47 | 20 | 1 | 1 | 12 | 6 |
| quad | waymo | 176 | 50 | 108 | 14 | 7 | 1 | 3 | 3 | 0 |
| waymo | drone | 366 | 154 | 179 | 32 | 24 | 0 | 0 | 10 | 14 |
| waymo | quad | 357 | 178 | 151 | 49 | 25 | 0 | 4 | 0 | 21 |

GTdiagnosis onlyafterpredictionseal/nativeaudit. Theseclassifications usetheAstudentzero, nottheABstudentzero, andall1798includingnoB; doNOTcomparedenominatorswithoutmatchingeligibility. CorrectcandidatesinCdonotprovefrozenteacherselectsthem.

## Numerical validity and storage

{
  "decision": {
    "primary": "cal_A_TTA",
    "same_R142_loss_steps_and_Up": true,
    "R144_sourceonly_positionformula_fixed": true,
    "teacher_scale_sourcecal_all1_and_Aimage_student_factor": true,
    "source_numerical_calibration_only": true,
    "FINAL_access": false,
    "no_target_GT_numeric_parameter_or_per_direction_choice": true,
    "exposed_development_not_independent_confirmation": true,
    "endpoints": {
      "cal_A_TTA": {
        "comparisons": {
          "raw": {
            "norm_TTA": -0.05177118410526482,
            "norm_Aimage": 0.08859874890984365,
            "norm_joint_zero": 0.033297774228975285,
            "cal_A_direct": 0.13090791169533667,
            "cal_A_wrong_TTA": 0.11335792136114442,
            "cal_B": 0.038201176278702675,
            "cal_FA_FB_mean": 0.1514902526361297,
            "cal_FAB_FB_mean": 0.1251847963669217,
            "Frozen_A": 0.347102833183067,
            "Frozen_F": 0.14428969855823937
          },
          "design_weighted": {
            "norm_TTA": -0.07697412849298679,
            "norm_Aimage": 0.08345896593274382,
            "norm_joint_zero": 0.0029157810505040516,
            "cal_A_direct": 0.09914577208597741,
            "cal_A_wrong_TTA": 0.1071738449666082,
            "cal_B": 0.009209312866261143,
            "cal_FA_FB_mean": 0.12623999580756085,
            "cal_FAB_FB_mean": 0.10014913943567016,
            "Frozen_A": 0.3481655732339508,
            "Frozen_F": 0.1617677971407403
          }
        },
        "retained_development_candidate": false
      },
      "cal_AB_TTA": {
        "comparisons": {
          "raw": {
            "norm_TTA": 0.0,
            "norm_joint_zero": 0.08506895833424011,
            "cal_AB_direct": 0.1826790958006015,
            "cal_AB_wrong_TTA": 0.13023946454335444,
            "cal_B": 0.0899723603839675,
            "cal_FA_FB_mean": 0.20326143674139452,
            "cal_FAB_FB_mean": 0.17695598047218652,
            "Frozen_A": 0.39887401728833183,
            "Frozen_F": 0.1960608826635042
          },
          "design_weighted": {
            "norm_TTA": 0.0,
            "norm_joint_zero": 0.07988990954349084,
            "cal_AB_direct": 0.1761199005789642,
            "cal_AB_wrong_TTA": 0.13320699436356165,
            "cal_B": 0.08618344135924794,
            "cal_FA_FB_mean": 0.20321412430054764,
            "cal_FAB_FB_mean": 0.17712326792865696,
            "Frozen_A": 0.4251397017269376,
            "Frozen_F": 0.2387419256337271
          }
        },
        "retained_development_candidate": false
      }
    },
    "disposition": "do_not_adopt_Aimage_student_keep_R145",
    "normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls": true
  },
  "source_alpha": {
    "status": "SOURCE_OWN_FINITE_CALIBRATION_LOCKED",
    "alpha": {
      "waymo": 1.0,
      "drone": 1.0,
      "quad": 1.0
    },
    "summary": [
      {
        "source": "waymo",
        "alpha": 0.5,
        "n": 32,
        "mean_teacher_C_IoU": 23.47389014069582
      },
      {
        "source": "waymo",
        "alpha": 1.0,
        "n": 32,
        "mean_teacher_C_IoU": 23.47389014069582
      },
      {
        "source": "waymo",
        "alpha": 2.0,
        "n": 32,
        "mean_teacher_C_IoU": 22.99133493319485
      },
      {
        "source": "drone",
        "alpha": 0.5,
        "n": 25,
        "mean_teacher_C_IoU": 13.324603632414721
      },
      {
        "source": "drone",
        "alpha": 1.0,
        "n": 25,
        "mean_teacher_C_IoU": 13.324603632414721
      },
      {
        "source": "drone",
        "alpha": 2.0,
        "n": 25,
        "mean_teacher_C_IoU": 13.324603632414721
      },
      {
        "source": "quad",
        "alpha": 0.5,
        "n": 56,
        "mean_teacher_C_IoU": 31.468108501686014
      },
      {
        "source": "quad",
        "alpha": 1.0,
        "n": 56,
        "mean_teacher_C_IoU": 31.468108501686014
      },
      {
        "source": "quad",
        "alpha": 2.0,
        "n": 56,
        "mean_teacher_C_IoU": 31.468108501686014
      }
    ],
    "counts": {
      "waymo": 32,
      "drone": 25,
      "quad": 56
    },
    "source_labels_only": true,
    "no_target_selection": true,
    "source_weight_training": false,
    "all_eligible_not_GT_prefiltered": true,
    "source_seal_sha256": "22c2722d3242358d0c3cfce46dcdb93955deaf221aff4a71f7ea83b937a104c3",
    "utc": "2026-11-17T21:25:26.800375+00:00"
  },
  "source_head_audit": {
    "status": "PASS",
    "n": 313,
    "max_numpy_base_error": 5.602646353963792e-06,
    "max_numpy_probability_error": 9.288015435648944e-07,
    "GT_read": false,
    "original_native_source_reader_audit": "R145/SOURCE_AUDIT.json"
  },
  "preparation": {
    "n": 1798,
    "eligible": 764,
    "changed_teacher": 0,
    "teacher_top_group_changed": 0,
    "new_plain_A_fits": 764,
    "new_cal_A_fits": 0,
    "new_cal_AB_fits": 0,
    "GT_read": false,
    "source_alpha": {
      "waymo": 1.0,
      "drone": 1.0,
      "quad": 1.0
    },
    "source_calibration_sha256": "fb032286a866cc16ed238f19ed2c0bdb7420e220ba4650945818ab1042258686",
    "metadata_serialization_repair_only": true
  },
  "preparation_audit": {
    "status": "PASS",
    "n": 1798,
    "original_R145_C_S_eligibility_exact": true,
    "max_independent_teacher_probability_error": 4.440892098500626e-16,
    "alpha1_exact_rows": 1798,
    "target_GT_read": false
  },
  "resource_profile": {
    "status": "PASS_BEFORE_MODEL",
    "profile_inputs": 12,
    "measured_mean_bytes": {
      "state": 135964.0,
      "feature": 3119909.1666666665,
      "record": 4776.75,
      "prediction": 84335.41666666667
    },
    "estimated_peak_bytes": 445191593.97499996,
    "newfits_max": 1528,
    "existing_payload_bytes": 31405057,
    "free_bytes": 14944743424,
    "scope_cap_GiB": 0.75,
    "streamed_chunk": 32,
    "old_files_deleted": false
  },
  "TTA_audit": {
    "status": "PASS",
    "n": 1798,
    "fit_count": 1506,
    "max_CPU_probability_error": 9.031470392106833e-07,
    "max_state_error": 0.0,
    "max_pointfeature_error": 1.430511474609375e-06,
    "max_objective_error": 4.656617313969491e-10,
    "all_native_chunks_audited_before_temporaryfeature_deletion": true,
    "predictions_states_gradients_traces_and_hashreceipts_retained": true,
    "engineering_full_features_retained": false,
    "formal_full_adapted_A_not_saved_or_fully_compared": true,
    "full1024point_keep_reconstructed": true,
    "fullTransformer_gradients_not_CPU_regenerated": true,
    "source_alpha_all1_exact_aliases": true,
    "original_R145_teacher_C_S_and_Up_budget": true,
    "student_A_RGB_only_teacher_B_RGB_only_but_same_Fpoint_input": true,
    "FINAL_access": false
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "task_values": 154628,
    "max_error": 0.0,
    "all_negatives_noB_and_aliases_kept": true
  },
  "runtime": {
    "n": 1798,
    "newfits": 1506,
    "GT_read": false,
    "primary": "cal_A_TTA",
    "alpha1_exact_aliases": true,
    "temporary_native_features_CPU_audited_then_deleted": true,
    "source_alpha_lock_sha256": "fb032286a866cc16ed238f19ed2c0bdb7420e220ba4650945818ab1042258686",
    "FINAL_access": false
  },
  "source_weight_training": false,
  "FINAL_access": false
}

Engineeringactualfullpathfinite-difference/zeroAimagebridge/actualFpoint-inputhook/sourcehash checks; independentCPUfullnativefinalmatching/competition, delta/radius/Armijo/objective/probes/all1024pointkeep checks. FORMAL retains32fixedadaptedpointprobes perfit untilnativeaudited, notentireadaptedpointtensor; engineeringcomparesfulltensor. FullTransformer/backpropnotindependentlyCPUregenerated.

Newtemporarynativefinalfeatures aresealedinchunks32, independentlyCPUchecked BEFOREanytaskGT, andONLYafterPASSremovedfromTHISroundtemporarydirectory. Allpredictions, states, gradients, traces, chunkmanifestsandnativehashreceiptspersist. Finalauditverifiesall1798receipts/persistentbytes/evidence/sourcebindings BEFOREGT. Thisboundsnewpayloadat.75GiBwith12GiBfreefloor+1GiBsafety; noolddata/results/weightsdeleted. Fullformalnativefeaturearraysarenotretainedafterstreamingaudit, so cannotclaimtheycouldbelateCPUreplayedwithoutregeneratingfeaturesinanewauthorizedrun. Auditsummaryandcodearepublic; privatechunkmanifestshaveinputpaths/IDs andarenotredistributed.

Two pre-model/pre-gradient engineeringfailuresarepreserved: registrationcopiedanonexistentR145positionmetadatafilename(reusedactualR148copy); preparationnumpyintegercounterfailedJSONmetadataafterall1798arrayswerematerialized. Finalizerrepairedmetadatawithoutreplayingthehead/changinganyarray, thenindependentauditcheckedallbeforegradients. Neitherisalgorithmicnegative. Allguardedfailedattemptcostsincluded; registrationpreflighthasnomeasuredmodelruntime. Incrementaltimingreusesnativeprefixes/mediaregistration andisnotcolddeploymentcost. NoFINAL/sourceweighttraining/newexpert/newarchitectureaccess.09:00BeijingNov18harddeadline remainsunchanged.


## Matched eligible teacher diagnostic

{
  "status": "POSTSEAL_DESCRIPTIVE_ONLY",
  "rows": [
    {
      "source": "drone",
      "target": "quad",
      "eligible": 178,
      "recoverable_Azero_errors": 91,
      "good_C_recoverable": 60,
      "teacher_C_good_recoverable": 2,
      "TTA_good_on_teacher_recoverable": 0,
      "teacher_C_harms_Azero_good": 13,
      "A_zero_outside_S": 56,
      "TTA_outside_S": 53,
      "insideS_to_outsideS": 1,
      "outsideS_to_insideS": 4,
      "parent_Cgroup_contains_good_but_selected_bad": 2,
      "teacher_member_better_parent": 3,
      "teacher_member_worse_parent": 1
    },
    {
      "source": "drone",
      "target": "waymo",
      "eligible": 50,
      "recoverable_Azero_errors": 23,
      "good_C_recoverable": 7,
      "teacher_C_good_recoverable": 1,
      "TTA_good_on_teacher_recoverable": 0,
      "teacher_C_harms_Azero_good": 4,
      "A_zero_outside_S": 24,
      "TTA_outside_S": 23,
      "insideS_to_outsideS": 0,
      "outsideS_to_insideS": 1,
      "parent_Cgroup_contains_good_but_selected_bad": 0,
      "teacher_member_better_parent": 0,
      "teacher_member_worse_parent": 0
    },
    {
      "source": "quad",
      "target": "drone",
      "eligible": 154,
      "recoverable_Azero_errors": 126,
      "good_C_recoverable": 47,
      "teacher_C_good_recoverable": 10,
      "TTA_good_on_teacher_recoverable": 3,
      "teacher_C_harms_Azero_good": 11,
      "A_zero_outside_S": 41,
      "TTA_outside_S": 39,
      "insideS_to_outsideS": 4,
      "outsideS_to_insideS": 6,
      "parent_Cgroup_contains_good_but_selected_bad": 0,
      "teacher_member_better_parent": 1,
      "teacher_member_worse_parent": 1
    },
    {
      "source": "quad",
      "target": "waymo",
      "eligible": 50,
      "recoverable_Azero_errors": 37,
      "good_C_recoverable": 14,
      "teacher_C_good_recoverable": 4,
      "TTA_good_on_teacher_recoverable": 0,
      "teacher_C_harms_Azero_good": 7,
      "A_zero_outside_S": 33,
      "TTA_outside_S": 31,
      "insideS_to_outsideS": 0,
      "outsideS_to_insideS": 2,
      "parent_Cgroup_contains_good_but_selected_bad": 0,
      "teacher_member_better_parent": 0,
      "teacher_member_worse_parent": 0
    },
    {
      "source": "waymo",
      "target": "drone",
      "eligible": 154,
      "recoverable_Azero_errors": 67,
      "good_C_recoverable": 32,
      "teacher_C_good_recoverable": 5,
      "TTA_good_on_teacher_recoverable": 3,
      "teacher_C_harms_Azero_good": 3,
      "A_zero_outside_S": 10,
      "TTA_outside_S": 10,
      "insideS_to_outsideS": 1,
      "outsideS_to_insideS": 1,
      "parent_Cgroup_contains_good_but_selected_bad": 0,
      "teacher_member_better_parent": 1,
      "teacher_member_worse_parent": 3
    },
    {
      "source": "waymo",
      "target": "quad",
      "eligible": 178,
      "recoverable_Azero_errors": 71,
      "good_C_recoverable": 49,
      "teacher_C_good_recoverable": 7,
      "TTA_good_on_teacher_recoverable": 1,
      "teacher_C_harms_Azero_good": 27,
      "A_zero_outside_S": 55,
      "TTA_outside_S": 51,
      "insideS_to_outsideS": 2,
      "outsideS_to_insideS": 6,
      "parent_Cgroup_contains_good_but_selected_bad": 3,
      "teacher_member_better_parent": 2,
      "teacher_member_worse_parent": 3
    }
  ],
  "totals": {
    "eligible": 764,
    "recoverable_Azero_errors": 415,
    "good_C_recoverable": 209,
    "teacher_C_good_recoverable": 29,
    "TTA_good_on_teacher_recoverable": 7,
    "teacher_C_harms_Azero_good": 65,
    "A_zero_outside_S": 219,
    "TTA_outside_S": 207,
    "insideS_to_outsideS": 8,
    "outsideS_to_insideS": 20,
    "parent_Cgroup_contains_good_but_selected_bad": 5,
    "teacher_member_better_parent": 7,
    "teacher_member_worse_parent": 8
  },
  "no_new_predictor_or_task_geometry": true,
  "GT_IoU_not_verified_instance_identity": true,
  "denominator": "original B/eligible764; sourceAimage zero is reference for recoverable counts",
  "no_GT_selected_deployment_filter": true
}

These postseal counts are descriptive, not new method outputs. A poor teacher conditional leader does not imply every probability component is wrong. Outside-S outputs are not automatically update-created errors; before/after transitions are explicitly separated. Within-group member information is weak on this cohort; do not launch a broader member KL merely because the omitted term is computable.
