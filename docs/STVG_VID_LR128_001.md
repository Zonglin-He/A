# Vid128 fixed-reference Spatial learning-rate development comparison

Same exposed core256 Vid128 parents. T2 clock, Caption, WHO references, Gaussian32 samples and query/round seeds, sigma=.4, tau=.5, steps3, response mu=.125 and original fullstep acceptance/rollback remain fixed. Current LR=.027 is exact whole-query sealed reuse; only .0135/.00675 are newly fit. No new Teacher/Temporal. GT read only after both complete prediction/root seals. This is development tuning, not new confirmation; all128 including no-fit are retained.

| Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |
|---|---:|---:|---:|---:|---:|
|Frozen|9.843861|30.729497|32.374371|10.156250|1.562500|
|T2_only|10.286840|30.729497|33.278160|10.937500|1.562500|
|LR_CURRENT|10.633082|31.205598|33.278160|11.718750|1.562500|
|LR_HALF|10.633534|31.255549|33.278160|10.937500|1.562500|
|LR_QUARTER|10.536393|31.150537|33.278160|10.937500|1.562500|

| Contrast | vIoU delta pp | Paired source t95 pp | + / - / neutral / unknown |
|---|---:|---|---|
|T2_only_minus_Frozen|0.44297962581364503|[-0.02543153251891561, 0.9113907841462057]|48/26/54/0|
|LR_CURRENT_minus_Frozen|0.7892212390942297|[0.0006448944658892231, 1.5777975837225702]|57/32/39/0|
|LR_CURRENT_minus_T2_only|0.3462416132805846|[-0.19296002700695664, 0.8854432535681258]|48/25/55/0|
|LR_HALF_minus_Frozen|0.7896734306988826|[0.0816511671249378, 1.4976956942728274]|57/32/39/0|
|LR_HALF_minus_T2_only|0.34669380488523754|[-0.09837139240615639, 0.7917590021766314]|50/26/52/0|
|LR_HALF_minus_LR_CURRENT|0.0004521916046528732|[-0.16917330578556602, 0.17007768899487177]|34/42/52/0|
|LR_QUARTER_minus_Frozen|0.692531891155237|[0.06738465711587138, 1.3176791251946025]|63/27/38/0|
|LR_QUARTER_minus_T2_only|0.24955226534159186|[-0.07672129234250752, 0.5758258230256912]|54/26/48/0|
|LR_QUARTER_minus_LR_CURRENT|-0.09668934793899278|[-0.3814638609853742, 0.18808516510738868]|36/44/48/0|
|LR_QUARTER_minus_LR_HALF|-0.09714153954364567|[-0.24495999443789712, 0.0506769153506058]|30/50/48/0|

Development selection (all candidates disclosed): {"selected": "LR_HALF", "lr": 0.0135, "means": {"LR_CURRENT": 0.10633082003503429, "LR_HALF": 0.10633534195108083, "LR_QUARTER": 0.10536392655564436}, "not_independent_confirmation": true, "paper_registry_unchanged": true}

Selection uses the already-exposed development set. Post-selection confidence intervals are descriptive, not unbiased confirmation of the chosen arm. No frozen paper or historical CURRENT registry is overwritten; no automatic follow-up search. Full updates may still follow imperfect Teacher references. Actual engineering pixel panels do not complete independent semantic review. Raw media, GT coordinates, actual query/Caption text, tensors and weights remain private.

Observed interpretation: the half LR is the nominal mean winner by only +0.000452pp versus current (paired source t95 -0.169173 to +0.170078pp), while quarter LR changes vIoU by -0.096689pp (t95 -0.381464 to +0.188085pp). Neither establishes an advantage over current. Half LR also reduces R@.3 by one source (15/128 to14/128). Zero-accepted fit sources decrease25 to18 to13 and accepted steps increase241 to283 to310 across current/half/quarter, but more accepted pseudo-reference updates do not establish better task localization. The frozen paper retains its original configuration; the prelocked development ranking is still disclosed without claiming a validated optimum.
