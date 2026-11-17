# R150 — Image-visibility-qualified teacher prerequisite

## Decision

**stop_visibility_routing_evidence_branch_no_TTA**. Both predeclared whole-cohort evidence gates fail. No new source calibration or parameter TTA was launched. Keep R145 best13.819603% versus intactFrozenA13.394463%; >=2ppgoal NOTmet. Same1798 exposedinputs/sixdirections/originalweights/allnegatives andphysicalclusters, notindependentconfirmation.

## What the new metadata found

OriginalS checksregisteredpointcoverage, notcamera-specificRGBsupport. Original1024seed/group geometry andpinnedR145 A/Bvisiblemasks wereused; Bprojections/sparsegraphs independentlyreproducedbefore anytaskGT. Totals across764 eligibleinputs: Bconditionalteacherleader has0visible supporteduniqueAseedpoints274times and<4points512times;474ofthesehave>=4supportpoints inAimage. Thisdoesnotprove theobjectisinvisibleineverypixel, thatcandidategroupsareinstances, orthatthemodelcannotusecontext. It does show thatpointregistrationqualification andlocalimageavailability aredifferentmeasurements.

NativeRGB384x256 ->3stride2convolutions ->48x32tokens. Prespecified3highestjoint-scoredsupportedgroupswithvisibleBpoints have small projectedshort-sideextents(medians.67–1.68cells bydirection); selectedsupport-disjointgroupssharecenterRGBcellsonlyabout2–5%ofthetime. Thesearesparsepredictedgroupsupportextents, NOTGTobjectpixelwidths. ThusdoNOTclaimallobjectsaretoosmallorcellcollisionisthemainfailure. ActualnativeROI wasnotpreviouslytested:R27usesQwen, R128baselineuses24x24cropsfromalreadydownsizedframes. No cropmodel/pixelswerecreatedinthisround.

{
  "status": "COMPLETE_NO_MODELS_NO_TASK_GT",
  "n": 764,
  "summary": [
    {
      "source": "drone",
      "target": "quad",
      "n": 178,
      "teacher_B_zero_visible": 65,
      "teacher_B_below4": 116,
      "teacher_A_ge4_B_lt4": 108,
      "mean_S_RGB_B_qualified_fraction": 0.4619955678223833,
      "mean_S_RGB_either_qualified_fraction": 0.9500144552842231,
      "selected_regions": 529,
      "median_short_cells": 1.2474393844604492,
      "fraction_short_lt3": 0.8771266540642723,
      "fraction_short_lt1": 0.4706994328922495,
      "fraction_with_shared_center_cell": 0.024574669187145556,
      "mean_shared_center_cell_fraction": 0.00981636510937078
    },
    {
      "source": "drone",
      "target": "waymo",
      "n": 50,
      "teacher_B_zero_visible": 3,
      "teacher_B_below4": 27,
      "teacher_A_ge4_B_lt4": 27,
      "mean_S_RGB_B_qualified_fraction": 0.6022946928595594,
      "mean_S_RGB_either_qualified_fraction": 0.9841944963349909,
      "selected_regions": 150,
      "median_short_cells": 1.67840576171875,
      "fraction_short_lt3": 0.64,
      "fraction_short_lt1": 0.37333333333333335,
      "fraction_with_shared_center_cell": 0.05333333333333334,
      "mean_shared_center_cell_fraction": 0.019002886002886
    },
    {
      "source": "quad",
      "target": "drone",
      "n": 154,
      "teacher_B_zero_visible": 77,
      "teacher_B_below4": 130,
      "teacher_A_ge4_B_lt4": 107,
      "mean_S_RGB_B_qualified_fraction": 0.19250297542546996,
      "mean_S_RGB_either_qualified_fraction": 0.8707007766508882,
      "selected_regions": 424,
      "median_short_cells": 0.6657823324203491,
      "fraction_short_lt3": 0.9009433962264151,
      "fraction_short_lt1": 0.5849056603773585,
      "fraction_with_shared_center_cell": 0.04245283018867924,
      "mean_shared_center_cell_fraction": 0.02090071877807727
    },
    {
      "source": "quad",
      "target": "waymo",
      "n": 50,
      "teacher_B_zero_visible": 2,
      "teacher_B_below4": 26,
      "teacher_A_ge4_B_lt4": 24,
      "mean_S_RGB_B_qualified_fraction": 0.4725811825524903,
      "mean_S_RGB_either_qualified_fraction": 0.9694719345075751,
      "selected_regions": 150,
      "median_short_cells": 1.2406425476074219,
      "fraction_short_lt3": 0.8333333333333334,
      "fraction_short_lt1": 0.43333333333333335,
      "fraction_with_shared_center_cell": 0.02666666666666667,
      "mean_shared_center_cell_fraction": 0.008333333333333333
    },
    {
      "source": "waymo",
      "target": "drone",
      "n": 154,
      "teacher_B_zero_visible": 69,
      "teacher_B_below4": 104,
      "teacher_A_ge4_B_lt4": 102,
      "mean_S_RGB_B_qualified_fraction": 0.26461655234241815,
      "mean_S_RGB_either_qualified_fraction": 0.9382957193238274,
      "selected_regions": 433,
      "median_short_cells": 1.3396975994110107,
      "fraction_short_lt3": 0.7090069284064665,
      "fraction_short_lt1": 0.4618937644341801,
      "fraction_with_shared_center_cell": 0.023094688221709007,
      "mean_shared_center_cell_fraction": 0.015832691814216066
    },
    {
      "source": "waymo",
      "target": "quad",
      "n": 178,
      "teacher_B_zero_visible": 58,
      "teacher_B_below4": 109,
      "teacher_A_ge4_B_lt4": 106,
      "mean_S_RGB_B_qualified_fraction": 0.45791823282713034,
      "mean_S_RGB_either_qualified_fraction": 0.9682114864896489,
      "selected_regions": 512,
      "median_short_cells": 1.205350399017334,
      "fraction_short_lt3": 0.91015625,
      "fraction_short_lt1": 0.484375,
      "fraction_with_shared_center_cell": 0.033203125,
      "mean_shared_center_cell_fraction": 0.01466703869047619
    }
  ],
  "graph_B_reproduces_pinned_R145": true,
  "only_original_1024seed_support_not_instance_identity": true,
  "feature_grid": [
    48,
    32
  ],
  "no_new_crop_pixels_or_model_inference": true
}

## Evidence routing, no new networks

FrozenAimage/Bimageprobabilities arepinnedR145 nativecomplete-readeroutputs onexactlythesameFpoint1024/normalizedpositions/originalA256/sourcepreparedweights. For eachoriginalSgroup countuniquephysicalseedpoints withvisiblepixelcorrespondenceinAandB. Qualifyaviewonlyifcount>=4(theexistingminimumphysicalsupportnumber, prespecifiedforRGB,noGTtuning). Primaryequalroute averagesqualifiedviews; ifonlyone qualifies useit; ifneitherqualifiesretainAimagevalue asUNKNOWN, notazero/negative. Normalizecandidatevectoronceglobally. OutsideoriginalS alsokeepAimage. Secondarycount routeweightsqualifiedviewsbytheirvisiblecounts, a diagnostic notautomaticTTAfamilyselection.

MatchedwrongroutepermutesA/Bweightvectorsamongqualifiedgroupswithidenticalquerymultiplicity, preservingweightmultiset/groupcardinality/unknownscope;699/764inputshaveeffectiveequal-weightmisbinding, othersrecordidentity. Globalrouteusesoneinput-levelmeanweightforallqualifiedgroupswithsameunknownfallback. Fixedhalfmean, bothsingleviews, originaljointzeroandR145remainstrongcontrols. Thistestsnewteachereligibility, notformalprobabilitycalibrationortrueinstanceidentity.

Per-instanceviewselectionhasaprimaryprecedentin [OpenMask3D](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d77b5482e38339a8068791d939126be2-Abstract-Conference.html): physicalvisibility informswhichRGBviews contribute. The [official feature extractor](https://github.com/OpenMask3D/openmask3d/blob/3bc3fc52693b25668d0e91d55a2ea714544a4749/openmask3d/mask_features_computation/features_extractor.py) computesdepth-consistentvisiblepointsbeforemask-specificviewselection. We reuseonlythismeasurementprinciple; noCLIP/SAM/Mask3D/extramodel, no certified3Dinstancemask, noformalOpenMask3D reproduction ornoveltyclaim. Thirdpartysnapshotnotredistributed.

## Frozen and conditional controls

AllnewpredictionssealedbeforetaskGT. Fullroutedteacherisapracticalno-updatecandidate. old-CteacherconditionsallmassonORIGINALC todiagnose teacherlabelqualitywithmatchedfixedscope; itdoesNOTcertifythattrueanswerliesinC andisnotclaimedasanabstainingdeploymentpolicy. old-Cdirect usesexactoriginalR136groupMAXprojectionwithroutedteacher, preservingwithin-grouprelativequeryscores. SourceS/C/kappa/decoder/boxcoords unchanged throughoutthisprerequisite. NativeoldCdirectmustnotbeconfusedwithgroupMEANmassremapping.

Primaryevidencegatepassesif fullroutedteacher beatsoldBteacher,matchedwrongandglobalroutesinbothraw/designweightedmetrics, OR fixed-Croutedteacherbeatsoldfixed-Cteacher,matchedwrongandglobalfixed-Croutesinbothmetrics. ThisdoesNOTrequireeverydirectionpositiveortheteacheritselftobeatR145beforeTTA. StrongR145andjointzeroresultsarestillreported. Bothgatesfail; source-readerrecalibration/newRGBscopeC/UpupdateswerethereforeNOTexecuted.

## Task scores

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
| raw | vis_equal | 13.65703618154393 | 26.560888886105857 | 5.344340915700607 |
| raw | vis_count | 13.649543009630477 | 26.46848791677625 | 5.296329286806626 |
| raw | vis_wrong_equal | 13.660833014777301 | 26.46504391622521 | 5.389878256319914 |
| raw | vis_wrong_count | 13.634328721556736 | 26.46619191640889 | 5.295181286622945 |
| raw | vis_global_equal | 13.640956371886118 | 26.373969234986593 | 5.344340915700607 |
| raw | vis_global_count | 13.662574875840301 | 26.42065457578958 | 5.249643946003637 |
| raw | vis_equal_Cteacher | 12.763515151439991 | 24.635388184086352 | 4.22124016024691 |
| raw | vis_equal_oldC_direct | 13.705118374308812 | 26.6963529077801 | 5.3910262565035945 |
| raw | vis_wrong_equal_Cteacher | 12.79320173547219 | 24.679599236614663 | 4.266777500866218 |
| raw | vis_wrong_equal_oldC_direct | 13.725741566917904 | 26.647193278702435 | 5.389878256319914 |
| raw | vis_global_equal_Cteacher | 12.784189001171619 | 24.63653618427003 | 4.22124016024691 |
| raw | vis_global_equal_oldC_direct | 13.732781400328463 | 26.697500907963775 | 5.3910262565035945 |
| raw | vis_count_Cteacher | 12.777234040290159 | 24.6843695252567 | 4.173228531352929 |
| raw | vis_count_oldC_direct | 13.71800715711103 | 26.69864890814746 | 5.343014627609613 |
| raw | old_Cteacher | 12.699847063612438 | 24.496301873953755 | 4.313462841669206 |
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
| design_weighted | vis_equal | 13.671428910183812 | 27.01697859565294 | 4.890829668044698 |
| design_weighted | vis_count | 13.647308520049684 | 26.86241557710243 | 4.840926627417876 |
| design_weighted | vis_wrong_equal | 13.656168097648843 | 26.85819476930685 | 4.936367008664006 |
| design_weighted | vis_wrong_count | 13.629239931265863 | 26.859601705238713 | 4.839519691486016 |
| design_weighted | vis_global_equal | 13.638807039199476 | 26.76712008806823 | 4.890829668044698 |
| design_weighted | vis_global_count | 13.660367264333294 | 26.814064364619398 | 4.793982350866708 |
| design_weighted | vis_equal_Cteacher | 12.812393982661655 | 25.118773435565473 | 3.7660442690763065 |
| design_weighted | vis_equal_oldC_direct | 13.733525300555037 | 27.153039608079936 | 4.936918018094935 |
| design_weighted | vis_wrong_equal_Cteacher | 12.829841055700792 | 25.100822438872882 | 3.811581609695614 |
| design_weighted | vis_wrong_equal_oldC_direct | 13.742953373539969 | 27.042055984785947 | 4.936367008664006 |
| design_weighted | vis_global_equal_Cteacher | 12.83303626679944 | 25.12018037149733 | 3.7660442690763065 |
| design_weighted | vis_global_equal_oldC_direct | 13.76184637896255 | 27.1544465440118 | 4.936918018094935 |
| design_weighted | vis_count_Cteacher | 12.825926992600202 | 25.168531583980364 | 3.7161412284494855 |
| design_weighted | vis_count_oldC_direct | 13.74622360034236 | 27.15585347994366 | 4.887014977468113 |
| design_weighted | old_Cteacher | 12.743227737909109 | 24.95661036306735 | 3.8585258862467833 |

{
  "phase": "frozen_evidence_prerequisite",
  "primary": "vis_equal",
  "secondary": "vis_count_diagnostic_only",
  "new_network_forwards": 0,
  "FINAL_access": false,
  "target_exposed_development": true,
  "endpoints": {
    "vis_equal": {
      "comparisons": {
        "raw": {
          "norm_Bimage": -0.05490611462128392,
          "vis_wrong_equal": -0.0037968332333715438,
          "vis_global_equal": 0.016079809657812305,
          "norm_joint_zero": -0.05980951667101131,
          "norm_Aimage": -0.004508541990142945,
          "norm_FA_FB_mean": 0.0583829617361431,
          "norm_TTA": -0.14487847500525142,
          "Frozen_A": 0.2539955422830804
        },
        "design_weighted": {
          "norm_Bimage": -0.06199081533802975,
          "vis_wrong_equal": 0.015260812534968693,
          "vis_global_equal": 0.032621870984336,
          "norm_joint_zero": -0.06828434715378684,
          "norm_Aimage": 0.012258837728452932,
          "norm_FA_FB_mean": 0.055039867603269954,
          "norm_TTA": -0.14817425669727768,
          "Frozen_A": 0.2769654450296599
        }
      }
    },
    "vis_equal_Cteacher": {
      "comparisons": {
        "raw": {
          "old_Cteacher": 0.06366808782755307,
          "vis_wrong_equal_Cteacher": -0.029686584032198482,
          "vis_global_equal_Cteacher": -0.02067384973162767
        },
        "design_weighted": {
          "old_Cteacher": 0.06916624475254629,
          "vis_wrong_equal_Cteacher": -0.017447073039136285,
          "vis_global_equal_Cteacher": -0.020642284137784017
        }
      }
    },
    "vis_count": {
      "comparisons": {
        "raw": {
          "norm_Bimage": -0.062399286534736476,
          "vis_wrong_count": 0.015214288073741145,
          "vis_global_count": -0.013031866209823662,
          "norm_joint_zero": -0.06730268858446387,
          "norm_TTA": -0.15237164691870397,
          "Frozen_A": 0.24650237036962785
        },
        "design_weighted": {
          "norm_Bimage": -0.08611120547215734,
          "vis_wrong_count": 0.018068588783821227,
          "vis_global_count": -0.013058744283609869,
          "norm_joint_zero": -0.09240473728791443,
          "norm_TTA": -0.17229464683140527,
          "Frozen_A": 0.25284505489553233
        }
      }
    },
    "vis_equal_oldC_direct": {
      "comparisons": {
        "raw": {
          "norm_direct": 0.08588281356023231,
          "vis_wrong_equal_oldC_direct": -0.020623192609091845,
          "vis_global_equal_oldC_direct": -0.02766302601965087,
          "norm_joint_zero": -0.011727323906129072,
          "norm_TTA": -0.09679628224036918,
          "Frozen_A": 0.30207773504796265
        },
        "design_weighted": {
          "norm_direct": 0.09004203425291202,
          "vis_wrong_equal_oldC_direct": -0.009428072984931646,
          "vis_global_equal_oldC_direct": -0.02832107840751341,
          "norm_joint_zero": -0.0061879567825613435,
          "norm_TTA": -0.08607786632605219,
          "Frozen_A": 0.3390618354008854
        }
      }
    }
  },
  "full_teacher_evidence_gate": false,
  "conditional_oldC_evidence_gate": false,
  "proceed_sourcecal_and_TTA": false,
  "disposition": "stop_visibility_routing_evidence_branch_no_TTA"
}

## All positive and negative changes

| method | control | better | worse | weighted_gain_pp | largest_positive_pp | largest_negative_pp | changed_content_groups | changed_frames | changed_scene_groups |
|---|---|---|---|---|---|---|---|---|---|
| vis_equal | norm_TTA | 54 | 58 | -0.14817425669727943 | 0.03206973218682594 | -0.040535899868321686 | 79 | 103 | 9 |
| vis_equal | norm_joint_zero | 26 | 25 | -0.06828434715378788 | 0.011988499523091839 | -0.040535899868321686 | 41 | 49 | 6 |
| vis_equal | norm_Aimage | 28 | 24 | 0.012258837728450628 | 0.04124910190536532 | -0.02121788318256053 | 41 | 50 | 6 |
| vis_equal | norm_Bimage | 58 | 77 | -0.06199081533803059 | 0.03206973218682594 | -0.040535899868321686 | 91 | 121 | 10 |
| vis_equal | norm_FA_FB_mean | 40 | 47 | 0.05503986760326886 | 0.03206973218682594 | -0.040535899868321686 | 65 | 81 | 9 |
| vis_equal | vis_wrong_equal | 29 | 36 | 0.015260812534968138 | 0.04124910190536532 | -0.017628455792287553 | 46 | 60 | 5 |
| vis_equal | vis_global_equal | 15 | 23 | 0.03262187098433474 | 0.04124910190536532 | -0.016561661845913917 | 30 | 37 | 3 |
| vis_equal | Frozen_A | 117 | 104 | 0.2769654450296598 | 0.053304207197498364 | -0.04909164122994028 | 124 | 181 | 18 |
| vis_equal_Cteacher | old_Cteacher | 49 | 66 | 0.06916624475254594 | 0.05417292030843174 | -0.030260612572096602 | 80 | 106 | 9 |
| vis_equal_Cteacher | vis_wrong_equal_Cteacher | 29 | 39 | -0.01744707303913579 | 0.02441521068576204 | -0.017628455792287553 | 49 | 62 | 6 |
| vis_equal_Cteacher | vis_global_equal_Cteacher | 11 | 24 | -0.020642284137783018 | 0.010564449576207823 | -0.01054303565995613 | 28 | 34 | 2 |
| vis_equal_oldC_direct | norm_direct | 49 | 60 | 0.09004203425291381 | 0.05417292030843174 | -0.030260612572096602 | 76 | 99 | 9 |
| vis_equal_oldC_direct | vis_wrong_equal_oldC_direct | 29 | 38 | -0.009428072984930883 | 0.02479833880958189 | -0.017628455792287553 | 48 | 62 | 7 |
| vis_equal_oldC_direct | vis_global_equal_oldC_direct | 12 | 23 | -0.028321078407513113 | 0.010564449576207823 | -0.011599796703079502 | 28 | 34 | 3 |
| vis_equal_oldC_direct | norm_joint_zero | 57 | 48 | -0.0061879567825604 | 0.03205222459287011 | -0.040535899868321686 | 82 | 97 | 13 |

IoUisabox-qualityproxy, notverifiedinstanceidentity. Sharedframes/directions/recordings arenotindependentreplicates; no singleDroneclusterpopulationCI, no targetcase/directionselection andnomaximalharmdeletion.

## Measurement and resources

{
  "decision": {
    "phase": "frozen_evidence_prerequisite",
    "primary": "vis_equal",
    "secondary": "vis_count_diagnostic_only",
    "new_network_forwards": 0,
    "FINAL_access": false,
    "target_exposed_development": true,
    "endpoints": {
      "vis_equal": {
        "comparisons": {
          "raw": {
            "norm_Bimage": -0.05490611462128392,
            "vis_wrong_equal": -0.0037968332333715438,
            "vis_global_equal": 0.016079809657812305,
            "norm_joint_zero": -0.05980951667101131,
            "norm_Aimage": -0.004508541990142945,
            "norm_FA_FB_mean": 0.0583829617361431,
            "norm_TTA": -0.14487847500525142,
            "Frozen_A": 0.2539955422830804
          },
          "design_weighted": {
            "norm_Bimage": -0.06199081533802975,
            "vis_wrong_equal": 0.015260812534968693,
            "vis_global_equal": 0.032621870984336,
            "norm_joint_zero": -0.06828434715378684,
            "norm_Aimage": 0.012258837728452932,
            "norm_FA_FB_mean": 0.055039867603269954,
            "norm_TTA": -0.14817425669727768,
            "Frozen_A": 0.2769654450296599
          }
        }
      },
      "vis_equal_Cteacher": {
        "comparisons": {
          "raw": {
            "old_Cteacher": 0.06366808782755307,
            "vis_wrong_equal_Cteacher": -0.029686584032198482,
            "vis_global_equal_Cteacher": -0.02067384973162767
          },
          "design_weighted": {
            "old_Cteacher": 0.06916624475254629,
            "vis_wrong_equal_Cteacher": -0.017447073039136285,
            "vis_global_equal_Cteacher": -0.020642284137784017
          }
        }
      },
      "vis_count": {
        "comparisons": {
          "raw": {
            "norm_Bimage": -0.062399286534736476,
            "vis_wrong_count": 0.015214288073741145,
            "vis_global_count": -0.013031866209823662,
            "norm_joint_zero": -0.06730268858446387,
            "norm_TTA": -0.15237164691870397,
            "Frozen_A": 0.24650237036962785
          },
          "design_weighted": {
            "norm_Bimage": -0.08611120547215734,
            "vis_wrong_count": 0.018068588783821227,
            "vis_global_count": -0.013058744283609869,
            "norm_joint_zero": -0.09240473728791443,
            "norm_TTA": -0.17229464683140527,
            "Frozen_A": 0.25284505489553233
          }
        }
      },
      "vis_equal_oldC_direct": {
        "comparisons": {
          "raw": {
            "norm_direct": 0.08588281356023231,
            "vis_wrong_equal_oldC_direct": -0.020623192609091845,
            "vis_global_equal_oldC_direct": -0.02766302601965087,
            "norm_joint_zero": -0.011727323906129072,
            "norm_TTA": -0.09679628224036918,
            "Frozen_A": 0.30207773504796265
          },
          "design_weighted": {
            "norm_direct": 0.09004203425291202,
            "vis_wrong_equal_oldC_direct": -0.009428072984931646,
            "vis_global_equal_oldC_direct": -0.02832107840751341,
            "norm_joint_zero": -0.0061879567825613435,
            "norm_TTA": -0.08607786632605219,
            "Frozen_A": 0.3390618354008854
          }
        }
      }
    },
    "full_teacher_evidence_gate": false,
    "conditional_oldC_evidence_gate": false,
    "proceed_sourcecal_and_TTA": false,
    "disposition": "stop_visibility_routing_evidence_branch_no_TTA"
  },
  "metadata_totals": {
    "n": 764,
    "teacher_B_zero_visible": 274,
    "teacher_B_below4": 512,
    "teacher_A_ge4_B_lt4": 474,
    "selected_regions": 2198
  },
  "preparation": {
    "RGB_scope_inputs_ge2": 764,
    "old_S_groups": 118532,
    "new_S_groups": 112020,
    "wrong_equal_effective": 699,
    "n": 1798,
    "new_models": 0,
    "labels_read": false
  },
  "audit": {
    "status": "PASS",
    "n": 1798,
    "eligible": 764,
    "max_routing_or_direct_error": 3.885780586188048e-16,
    "cached_native_reader_reference": "R145/IMAGE_AUDIT.json",
    "new_network_forwards": 0,
    "GT_read": false,
    "FINAL_access": false,
    "unknown_groups_keep_A_not_negative": true
  },
  "scalar_audit": {
    "status": "PASS",
    "n": 1798,
    "task_values": 181598,
    "max_error": 0.0,
    "all_historical_and_new_negative_cases_kept": true
  },
  "new_network_forwards": 0,
  "source_calibration_run": false,
  "TTA_run": false,
  "FINAL_access": false,
  "metadata_no_GT": true,
  "target_exposed_development": true
}

No networkforward/backprop/sourceweighttraining/newimagecrop/newpointencoding/newA256/FINAL. CPUmetadata usescameraallowlistandimageheaders, nosourceortargettasklabels. Frozenoutputs reuseauditednativepredictions; independentauditorreconstructsuniquegroupRGBcounts,eligibility,permutation,vectorrouting,UNKNOWNfallback,forced-CandnativegroupMAXdirectprojection. NewpredictionarraysaresealedandhashcheckedbeforeGT; all86parentR149armsarekeptbypinnedparentreferencewithoutduplicatingtheirrawmatrices. Publictablesincludeall101arms. SourcefullTransformer/backpropnotCPUrerun. Allscalarsindependentlyrecalculated.

Frozenaudit/scorercodewaspinnedafterpredictioncodeexecutionbutBEFOREanytaskGT/scoring; theirlockskeepactualorderexplicit,notretroactivelypresentedasprediction-prelockedcode. Coreproducer/designwerepinnedbeforeexecution. Aninitialnonexecutedauditoralgebraexpressionwascorrectedfromgroup-sizeweightingtothenativegroupMAXmassmappingbeforeauditexecution; no predictions changed. Nofailedmodelattempt. Resourcecap.45GiB/floor12GiB+safety1; nooldartifactsdeleted. Method/routingcacheisnotacolddeploymentlatencybenchmark. Harddeadline09:00BeijingNov18 unchanged.


## Post-write resource measurement

{
  "status": "PASS_POSTWRITE_MEASURED",
  "payload_bytes": 39286553,
  "free_bytes": 14548664320,
  "cap_GiB": 0.45,
  "free_floor_GiB": 12,
  "safety_GiB": 1,
  "new_models": 0,
  "new_optimization": 0,
  "no_old_artifact_deletion": true,
  "per_iteration_storage_polling_not_claimed": true
}

This CPU-only frozen cache round measured the cap/floor after materialization; it did not run a model or claim per-iteration model-storage polling.
