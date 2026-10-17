# Frozen paper T2+B integrated256

This is a frozen development evaluation with all no-fit/unknown outcomes retained. Target GT was read only after complete prediction/root seals. It is not independent confirmation. Historical failed confirmation and T2/T4 noninferiority findings remain unchanged.

| Domain | Arm | vIoU % | tIoU % | dense sIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|30.468750|7.812500|
|hc2|T2_only|26.331915|49.232269|53.809342|41.406250|14.062500|
|hc2|S_B_Native|22.131307|39.546835|54.903673|32.812500|8.593750|
|hc2|Full_T2_B|26.725237|49.232269|54.840045|42.968750|12.500000|
|hc2|T2_NativeB_evidence_diagnostic|26.893742|49.232269|54.903673|41.406250|14.843750|
|hc2|Full_T4_B_captionfixed|26.646800|49.042562|54.671292|42.187500|11.718750|
|vidstg|Frozen|9.843861|32.374371|30.729497|10.156250|1.562500|
|vidstg|T2_only|10.286840|33.278160|30.729497|10.937500|1.562500|
|vidstg|S_B_Native|10.262778|32.374371|31.360643|9.375000|3.125000|
|vidstg|Full_T2_B|10.633082|33.278160|31.205598|11.718750|1.562500|
|vidstg|T2_NativeB_evidence_diagnostic|10.704679|33.278160|31.360643|10.937500|2.343750|
|vidstg|Full_T4_B_captionfixed|11.059517|33.602694|31.977050|13.281250|3.125000|

| Domain | Contrast | Metric | Delta pp | Source-paired t95 pp | + / - / neutral / unknown |
|---|---|---|---:|---|---|
|hc2|T2_only_minus_Frozen|m_vIoU|4.680755653103774|[3.1010778893282684, 6.26043341687928]|78/26/24/0|
|hc2|T2_only_minus_Frozen|m_tIoU|9.685433961021994|[6.878881006782061, 12.491986915261927]|83/23/22/0|
|hc2|T2_only_minus_Frozen|m_sIoU_dense_GT|0.0|[0.0, 0.0]|0/0/128/0|
|hc2|S_B_Native_minus_Frozen|m_vIoU|0.48014707208687807|[0.1939885392295284, 0.7663056049442277]|66/38/24/0|
|hc2|S_B_Native_minus_Frozen|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|hc2|S_B_Native_minus_Frozen|m_sIoU_dense_GT|1.094331117415543|[0.6082367814204064, 1.5804254534106796]|79/41/8/0|
|hc2|S_B_Native_minus_T2_only|m_vIoU|-4.200608581016897|[-5.856705897678488, -2.5445112643553056]|37/76/15/0|
|hc2|S_B_Native_minus_T2_only|m_tIoU|-9.685433961021994|[-12.491986915261927, -6.878881006782061]|23/83/22/0|
|hc2|S_B_Native_minus_T2_only|m_sIoU_dense_GT|1.094331117415543|[0.6082367814204064, 1.5804254534106796]|79/41/8/0|
|hc2|Full_T2_B_minus_Frozen|m_vIoU|5.074077331331191|[3.4914796888399664, 6.656674973822415]|88/26/14/0|
|hc2|Full_T2_B_minus_Frozen|m_tIoU|9.685433961021994|[6.878881006782061, 12.491986915261927]|83/23/22/0|
|hc2|Full_T2_B_minus_Frozen|m_sIoU_dense_GT|1.0307024961209295|[0.5165365593204154, 1.5448684329214435]|83/39/6/0|
|hc2|Full_T2_B_minus_T2_only|m_vIoU|0.39332167822741654|[0.07546025151516828, 0.7111831049396649]|78/34/16/0|
|hc2|Full_T2_B_minus_T2_only|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|hc2|Full_T2_B_minus_T2_only|m_sIoU_dense_GT|1.0307024961209295|[0.5165365593204154, 1.5448684329214435]|83/39/6/0|
|hc2|Full_T2_B_minus_S_B_Native|m_vIoU|4.5939302592443125|[2.962826379466997, 6.225034139021628]|81/25/22/0|
|hc2|Full_T2_B_minus_S_B_Native|m_tIoU|9.685433961021994|[6.878881006782061, 12.491986915261927]|83/23/22/0|
|hc2|Full_T2_B_minus_S_B_Native|m_sIoU_dense_GT|-0.0636286212946135|[-0.5906095850318791, 0.46335234244265205]|65/52/11/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_Frozen|m_vIoU|5.242582552659269|[3.6628679827706767, 6.8222971225478615]|88/26/14/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_Frozen|m_tIoU|9.685433961021994|[6.878881006782061, 12.491986915261927]|83/23/22/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_Frozen|m_sIoU_dense_GT|1.094331117415543|[0.6082367814204064, 1.5804254534106796]|79/41/8/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_T2_only|m_vIoU|0.5618268995554948|[0.25205231419301255, 0.8716014849179771]|69/41/18/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_T2_only|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_T2_only|m_sIoU_dense_GT|1.094331117415543|[0.6082367814204064, 1.5804254534106796]|79/41/8/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_S_B_Native|m_vIoU|4.762435480572391|[3.1575462496620457, 6.367324711482736]|79/26/23/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_S_B_Native|m_tIoU|9.685433961021994|[6.878881006782061, 12.491986915261927]|83/23/22/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_S_B_Native|m_sIoU_dense_GT|0.0|[0.0, 0.0]|0/0/128/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_Full_T2_B|m_vIoU|0.1685052213280783|[-0.13148568490160953, 0.46849612755776615]|47/58/23/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_Full_T2_B|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|hc2|T2_NativeB_evidence_diagnostic_minus_Full_T2_B|m_sIoU_dense_GT|0.0636286212946135|[-0.46335234244265205, 0.5906095850318791]|52/65/11/0|
|hc2|Full_T4_B_captionfixed_minus_Frozen|m_vIoU|4.995640042014948|[3.31343375618716, 6.677846327842736]|89/24/15/0|
|hc2|Full_T4_B_captionfixed_minus_Frozen|m_tIoU|9.495726954454163|[6.650433820910646, 12.34102008799768]|82/21/25/0|
|hc2|Full_T4_B_captionfixed_minus_Frozen|m_sIoU_dense_GT|0.8619498507100185|[0.3513772367442146, 1.3725224646758223]|77/42/9/0|
|hc2|Full_T4_B_captionfixed_minus_T2_only|m_vIoU|0.3148843889111738|[-0.40440199117338854, 1.0341707689957362]|69/43/16/0|
|hc2|Full_T4_B_captionfixed_minus_T2_only|m_tIoU|-0.18970700656782993|[-1.311749499973843, 0.9323354868381832]|32/38/58/0|
|hc2|Full_T4_B_captionfixed_minus_T2_only|m_sIoU_dense_GT|0.8619498507100185|[0.3513772367442146, 1.3725224646758223]|77/42/9/0|
|hc2|Full_T4_B_captionfixed_minus_S_B_Native|m_vIoU|4.51549296992807|[2.7853311532673977, 6.245654786588743]|77/26/25/0|
|hc2|Full_T4_B_captionfixed_minus_S_B_Native|m_tIoU|9.495726954454163|[6.650433820910646, 12.34102008799768]|82/21/25/0|
|hc2|Full_T4_B_captionfixed_minus_S_B_Native|m_sIoU_dense_GT|-0.23238126670552456|[-0.6904429087731189, 0.22568037536206975]|62/52/14/0|
|hc2|Full_T4_B_captionfixed_minus_Full_T2_B|m_vIoU|-0.07843728931624276|[-0.769975291063787, 0.6131007124313016]|33/40/55/0|
|hc2|Full_T4_B_captionfixed_minus_Full_T2_B|m_tIoU|-0.18970700656782993|[-1.311749499973843, 0.9323354868381832]|32/38/58/0|
|hc2|Full_T4_B_captionfixed_minus_Full_T2_B|m_sIoU_dense_GT|-0.16875264541091106|[-0.5877688835370739, 0.2502635927152518]|37/40/51/0|
|hc2|Full_T4_B_captionfixed_minus_T2_NativeB_evidence_diagnostic|m_vIoU|-0.24694251064432104|[-0.9607732729700378, 0.4668882516813957]|51/54/23/0|
|hc2|Full_T4_B_captionfixed_minus_T2_NativeB_evidence_diagnostic|m_tIoU|-0.18970700656782993|[-1.311749499973843, 0.9323354868381832]|32/38/58/0|
|hc2|Full_T4_B_captionfixed_minus_T2_NativeB_evidence_diagnostic|m_sIoU_dense_GT|-0.23238126670552456|[-0.6904429087731189, 0.22568037536206975]|62/52/14/0|
|vidstg|T2_only_minus_Frozen|m_vIoU|0.44297962581364503|[-0.02543153251891561, 0.9113907841462057]|48/26/54/0|
|vidstg|T2_only_minus_Frozen|m_tIoU|0.9037890610976052|[-0.14820723156951088, 1.9557853537647212]|53/30/45/0|
|vidstg|T2_only_minus_Frozen|m_sIoU_dense_GT|0.0|[0.0, 0.0]|0/0/128/0|
|vidstg|S_B_Native_minus_Frozen|m_vIoU|0.4189169935770362|[-0.0597510518792474, 0.8975850390333198]|43/29/56/0|
|vidstg|S_B_Native_minus_Frozen|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|vidstg|S_B_Native_minus_Frozen|m_sIoU_dense_GT|0.6311450888571346|[-0.5154048553346022, 1.7776950330488714]|55/39/34/0|
|vidstg|S_B_Native_minus_T2_only|m_vIoU|-0.024062632236608806|[-0.6722011584945378, 0.6240758940213202]|46/44/38/0|
|vidstg|S_B_Native_minus_T2_only|m_tIoU|-0.9037890610976052|[-1.9557853537647212, 0.14820723156951088]|30/53/45/0|
|vidstg|S_B_Native_minus_T2_only|m_sIoU_dense_GT|0.6311450888571346|[-0.5154048553346022, 1.7776950330488714]|55/39/34/0|
|vidstg|Full_T2_B_minus_Frozen|m_vIoU|0.7892212390942297|[0.0006448944658892231, 1.5777975837225702]|57/32/39/0|
|vidstg|Full_T2_B_minus_Frozen|m_tIoU|0.9037890610976052|[-0.14820723156951088, 1.9557853537647212]|53/30/45/0|
|vidstg|Full_T2_B_minus_Frozen|m_sIoU_dense_GT|0.4761001189202059|[-0.7764158171417088, 1.7286160549821206]|61/32/35/0|
|vidstg|Full_T2_B_minus_T2_only|m_vIoU|0.3462416132805846|[-0.19296002700695664, 0.8854432535681258]|48/25/55/0|
|vidstg|Full_T2_B_minus_T2_only|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|vidstg|Full_T2_B_minus_T2_only|m_sIoU_dense_GT|0.4761001189202059|[-0.7764158171417088, 1.7286160549821206]|61/32/35/0|
|vidstg|Full_T2_B_minus_S_B_Native|m_vIoU|0.37030424551719343|[-0.39115906170089776, 1.1317675527352846]|48/31/49/0|
|vidstg|Full_T2_B_minus_S_B_Native|m_tIoU|0.9037890610976052|[-0.14820723156951088, 1.9557853537647212]|53/30/45/0|
|vidstg|Full_T2_B_minus_S_B_Native|m_sIoU_dense_GT|-0.15504496993692865|[-1.541739984807935, 1.2316500449340775]|53/39/36/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_Frozen|m_vIoU|0.860818020252052|[0.18686937966953754, 1.5347666608345665]|59/31/38/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_Frozen|m_tIoU|0.9037890610976052|[-0.14820723156951088, 1.9557853537647212]|53/30/45/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_Frozen|m_sIoU_dense_GT|0.6311450888571346|[-0.5154048553346022, 1.7776950330488714]|55/39/34/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_T2_only|m_vIoU|0.4178383944384069|[-0.04807534664323565, 0.8837521355200495]|45/29/54/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_T2_only|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_T2_only|m_sIoU_dense_GT|0.6311450888571346|[-0.5154048553346022, 1.7776950330488714]|55/39/34/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_S_B_Native|m_vIoU|0.44190102667501574|[-0.027395583340421525, 0.9111976366904531]|47/29/52/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_S_B_Native|m_tIoU|0.9037890610976052|[-0.14820723156951088, 1.9557853537647212]|53/30/45/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_S_B_Native|m_sIoU_dense_GT|0.0|[0.0, 0.0]|0/0/128/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_Full_T2_B|m_vIoU|0.07159678115782231|[-0.46659807753292615, 0.6097916398485708]|33/39/56/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_Full_T2_B|m_tIoU|0.0|[0.0, 0.0]|0/0/128/0|
|vidstg|T2_NativeB_evidence_diagnostic_minus_Full_T2_B|m_sIoU_dense_GT|0.15504496993692865|[-1.2316500449340775, 1.541739984807935]|39/53/36/0|
|vidstg|Full_T4_B_captionfixed_minus_Frozen|m_vIoU|1.2156566778776277|[0.36961845247585956, 2.061694903279396]|54/28/46/0|
|vidstg|Full_T4_B_captionfixed_minus_Frozen|m_tIoU|1.2283231025354617|[0.11935380429058351, 2.33729240078034]|46/27/55/0|
|vidstg|Full_T4_B_captionfixed_minus_Frozen|m_sIoU_dense_GT|1.2475522551889742|[0.2610407459282186, 2.2340637644497296]|58/33/37/0|
|vidstg|Full_T4_B_captionfixed_minus_T2_only|m_vIoU|0.7726770520639826|[0.12287556338965411, 1.4224785407383111]|49/33/46/0|
|vidstg|Full_T4_B_captionfixed_minus_T2_only|m_tIoU|0.3245340414378565|[-0.3540354012708207, 1.0031034841465338]|25/39/64/0|
|vidstg|Full_T4_B_captionfixed_minus_T2_only|m_sIoU_dense_GT|1.2475522551889742|[0.2610407459282186, 2.2340637644497296]|58/33/37/0|
|vidstg|Full_T4_B_captionfixed_minus_S_B_Native|m_vIoU|0.7967396843005914|[-0.012087742259050516, 1.6055671108602332]|40/28/60/0|
|vidstg|Full_T4_B_captionfixed_minus_S_B_Native|m_tIoU|1.2283231025354617|[0.11935380429058351, 2.33729240078034]|46/27/55/0|
|vidstg|Full_T4_B_captionfixed_minus_S_B_Native|m_sIoU_dense_GT|0.6164071663318395|[-0.41167238116072846, 1.6444867138244073]|43/40/45/0|
|vidstg|Full_T4_B_captionfixed_minus_Full_T2_B|m_vIoU|0.42643543878339796|[-0.24983016474948483, 1.1027010423162809]|27/35/66/0|
|vidstg|Full_T4_B_captionfixed_minus_Full_T2_B|m_tIoU|0.3245340414378565|[-0.3540354012708207, 1.0031034841465338]|25/39/64/0|
|vidstg|Full_T4_B_captionfixed_minus_Full_T2_B|m_sIoU_dense_GT|0.7714521362687681|[-0.42559272489772426, 1.9684969974352604]|30/41/57/0|
|vidstg|Full_T4_B_captionfixed_minus_T2_NativeB_evidence_diagnostic|m_vIoU|0.35483865762557565|[-0.2566401488531973, 0.9663174641043486]|37/46/45/0|
|vidstg|Full_T4_B_captionfixed_minus_T2_NativeB_evidence_diagnostic|m_tIoU|0.3245340414378565|[-0.3540354012708207, 1.0031034841465338]|25/39/64/0|
|vidstg|Full_T4_B_captionfixed_minus_T2_NativeB_evidence_diagnostic|m_sIoU_dense_GT|0.6164071663318395|[-0.41167238116072846, 1.6444867138244073]|43/40/45/0|

Full_T2_B uses actual T2-conditioned observations, candidate pixels, Caption policy and WHO references, followed by original sparse Gaussian OPD. Historical T4 Full is explicitly labeled; NativeB evidence attached to final T2 time is a readout diagnostic. T2 was selected by human simplicity preference, not a relabeled noninferiority claim. B is fixed, with s=.75mean-.125disagreement and original MMR scale; it is not a calibrated posterior or universal superiority claim. No C/Global32/noCoverage/annealing/Tube/Decoder-Aligned or online writeback. Source/T2/cached feature work is reused only after exact dependency checks and is not fresh zero-cost work. Official full roster, scientifically qualified second Backbone and fresh cost comparison are separate actual execution tasks, not completed by this256 result.

All source-paired signs and unknowns are public. Pixels are separately root inspected; these engineering views are not independent semantic labels. Private media/query/Caption text, GT coordinates, raw Teacher/tensors/weights, credentials and personal conversations are excluded.
