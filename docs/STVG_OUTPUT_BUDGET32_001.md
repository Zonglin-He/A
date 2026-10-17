# Independent output-KL-budget M0/M1/M2 development32

This is a frozen development evaluation with all no-fit/unknown outcomes retained. Target GT was read only after complete prediction/root seals. It is not independent confirmation. Historical failed confirmation and T2/T4 noninferiority findings remain unchanged.

| Domain | Arm | vIoU % | tIoU % | dense sIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Source_T|26.625819|52.919754|50.497117|37.500000|12.500000|
|hc2|M0|27.955626|52.919754|52.796755|37.500000|12.500000|
|hc2|M1|26.699242|52.919754|50.618854|37.500000|12.500000|
|hc2|M2|26.698911|52.919754|50.610640|37.500000|12.500000|
|vidstg|Source_T|12.052470|35.947273|32.870500|12.500000|0.000000|
|vidstg|M0|12.222038|35.947273|31.290835|12.500000|6.250000|
|vidstg|M1|12.695640|35.947273|33.715222|12.500000|0.000000|
|vidstg|M2|12.292884|35.947273|33.261658|12.500000|0.000000|

| Domain | Contrast | Metric | Delta pp | Source-paired t95 pp | + / - / neutral / unknown |
|---|---|---|---:|---|---|
|hc2|M0_minus_Source_T|m_vIoU|1.329807250781646|[0.5972796095615184, 2.0623348920017737]|13/1/2/0|
|hc2|M0_minus_Source_T|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|hc2|M0_minus_Source_T|m_sIoU_dense_GT|2.299637720956448|[1.1766653245821783, 3.4226101173307177]|13/1/2/0|
|hc2|M1_minus_Source_T|m_vIoU|0.07342293079728407|[-0.022238166047286775, 0.16908402764185493]|5/1/10/0|
|hc2|M1_minus_Source_T|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|hc2|M1_minus_Source_T|m_sIoU_dense_GT|0.12173678718774891|[-0.03170953518516903, 0.2751831095606668]|6/1/9/0|
|hc2|M1_minus_M0|m_vIoU|-1.256384319984362|[-1.942863937856374, -0.5699047021123501]|1/13/2/0|
|hc2|M1_minus_M0|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|hc2|M1_minus_M0|m_sIoU_dense_GT|-2.177900933768699|[-3.230170170167335, -1.1256316973700629]|2/13/1/0|
|hc2|M2_minus_Source_T|m_vIoU|0.07309199170272015|[-0.011419648912474475, 0.1576036323179148]|9/0/7/0|
|hc2|M2_minus_Source_T|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|hc2|M2_minus_Source_T|m_sIoU_dense_GT|0.11352302493926564|[-0.007417604316176232, 0.2344636541947075]|10/0/6/0|
|hc2|M2_minus_M0|m_vIoU|-1.256715259078926|[-1.936836258575264, -0.5765942595825877]|1/13/2/0|
|hc2|M2_minus_M0|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|hc2|M2_minus_M0|m_sIoU_dense_GT|-2.1861146960171824|[-3.2520416359337485, -1.1201877561006162]|2/13/1/0|
|hc2|M2_minus_M1|m_vIoU|-0.00033093909456390937|[-0.043499321594157585, 0.04283744340502977]|6/4/6/0|
|hc2|M2_minus_M1|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|hc2|M2_minus_M1|m_sIoU_dense_GT|-0.00821376224848327|[-0.08233983385601032, 0.06591230935904377]|6/5/5/0|
|vidstg|M0_minus_Source_T|m_vIoU|0.16956881579796856|[-2.3777725520191804, 2.7169101836151177]|6/3/7/0|
|vidstg|M0_minus_Source_T|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|vidstg|M0_minus_Source_T|m_sIoU_dense_GT|-1.5796642714335574|[-8.301655532690186, 5.142326989823071]|7/4/5/0|
|vidstg|M1_minus_Source_T|m_vIoU|0.6431706181932535|[-0.3347456451569253, 1.6210868815434323]|4/1/11/0|
|vidstg|M1_minus_Source_T|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|vidstg|M1_minus_Source_T|m_sIoU_dense_GT|0.8447225128379013|[-0.5292073639565827, 2.2186523896323855]|4/2/10/0|
|vidstg|M1_minus_M0|m_vIoU|0.47360180239528493|[-1.6500115806836675, 2.597215185474237]|4/5/7/0|
|vidstg|M1_minus_M0|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|vidstg|M1_minus_M0|m_sIoU_dense_GT|2.4243867842714586|[-3.79552078615387, 8.644294354696786]|4/7/5/0|
|vidstg|M2_minus_Source_T|m_vIoU|0.24041429164593878|[-0.04678088645420672, 0.5276094697460842]|8/1/7/0|
|vidstg|M2_minus_Source_T|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|vidstg|M2_minus_Source_T|m_sIoU_dense_GT|0.391158076562339|[-0.03465865894784548, 0.8169748120725234]|10/2/4/0|
|vidstg|M2_minus_M0|m_vIoU|0.07084547584797016|[-2.425453310074239, 2.567144261770179]|4/6/6/0|
|vidstg|M2_minus_M0|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|vidstg|M2_minus_M0|m_sIoU_dense_GT|1.9708223479958966|[-4.504712517202714, 8.446357213194506]|6/7/3/0|
|vidstg|M2_minus_M1|m_vIoU|-0.4027563265473147|[-1.2656168165638024, 0.460104163469173]|5/5/6/0|
|vidstg|M2_minus_M1|m_tIoU|0.0|[0.0, 0.0]|0/0/16/0|
|vidstg|M2_minus_M1|m_sIoU_dense_GT|-0.4535644362755625|[-1.6387886677019676, 0.7316597951508428]|8/5/3/0|

M0 is the exact old original-response OPD. M1 projects raw Adam onto the fixed linearized output-KL budget; M2 realizes the OPD anchor target with the same budget and gamma=1. Epsilon=.01 is fixed without GT. All accepted M1/M2 steps also satisfy actual nonlinear decoded KL and original sparse reference acceptance, with original rollback/early-stop and no added shrink search. Invalid whole-trajectory logit charts are whole-query Source no-ops in the full32 table; the pre-adaptation chart-valid diagnostic never replaces it. Primary contrast is M2-M1. This independent branch does not change the frozen paper method. Dense-Jacobian implementation costs and unknown fresh/peak comparisons are explicit. No new Teacher/Temporal fits or policy/Prompt/LR/sigma search.

All source-paired signs and unknowns are public. Pixels are separately root inspected; these engineering views are not independent semantic labels. Private media/query/Caption text, GT coordinates, raw Teacher/tensors/weights, credentials and personal conversations are excluded.
