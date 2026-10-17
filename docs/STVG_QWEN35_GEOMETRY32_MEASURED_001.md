# Qwen3.5 independent sparse geometry32: original OPD versus same-reference propagation

Fixed exposed HC16/Vid16 parents, same128 T2 anchors, original unmarked context4 and standalone anchor image. Qwen3.5-9B BF16 uses one fixed non-thinking normalized1000 protocol. Four prelocked parent interface requests count among128 formal requests; no outcome retry, Prompt sweep or model substitution. Original R1 temporal feedback is retained: this is not a single-Teacher system. Native processor resize/clock and actual generation are bound by the qualification receipt. The historical uniform context-clock approximation is retained and disclosed; source timestamps are preserved in private input audits.

QWEN_OPD uses unchanged original1792-parameter episodic Gaussian/response/fullstep rollback/reject-stop. QWEN_D2 uses the exact same sparse Qwen references and unchanged Source-relative center/log-size interpolation in physical time inside T2, with no Student updates. Missing/invalid/abstained references remain in the denominator. Both complete prediction families and roots precede GT. Frozen, T2, WF, original and revised R1 Direct+OPD, revised R1 same-reference D2, R1 Heads and D0/D1/D2 are exact matched old predictions; their cost is not zero. All negative/neutral/unknown source deltas and extreme cases are retained. Development32 intervals quantify uncertainty, not a mandatory per-domain exclusion-of-zero selection gate. No domain splicing, new independent confirmation, production promotion or temporal-Teacher migration.

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|28.116723|50.497117|52.754115|31.250000|25.000000|
|hc2|T2|27.487135|50.497117|53.666875|37.500000|18.750000|
|hc2|Current_WF|28.641217|52.179161|53.666875|37.500000|25.000000|
|hc2|R1_DIRECT|27.972261|51.210459|53.666875|43.750000|18.750000|
|hc2|R1_FIXED_OPD|28.580257|52.108541|53.666875|37.500000|25.000000|
|hc2|R1_FIXED_D2|25.756999|44.559898|53.666875|31.250000|18.750000|
|hc2|R1_HEADS|28.769897|52.627071|53.666875|43.750000|25.000000|
|hc2|D0|28.440075|51.967586|53.666875|37.500000|25.000000|
|hc2|D1|28.600908|52.314891|53.666875|37.500000|25.000000|
|hc2|D2|32.632171|57.825914|53.666875|43.750000|31.250000|
|hc2|QWEN_OPD|28.840534|52.752743|53.666875|37.500000|25.000000|
|hc2|QWEN_D2|29.770524|51.192975|53.666875|43.750000|18.750000|
|vidstg|Frozen|12.185607|32.870500|35.923576|12.500000|0.000000|
|vidstg|T2|12.239461|32.870500|35.680949|12.500000|0.000000|
|vidstg|Current_WF|12.072499|32.573620|35.680949|12.500000|0.000000|
|vidstg|R1_DIRECT|14.323998|35.542683|35.680949|18.750000|0.000000|
|vidstg|R1_FIXED_OPD|14.572580|31.878854|35.680949|18.750000|0.000000|
|vidstg|R1_FIXED_D2|13.222003|36.900708|35.680949|12.500000|6.250000|
|vidstg|R1_HEADS|14.012450|31.388836|35.680949|25.000000|0.000000|
|vidstg|D0|15.368798|33.573892|35.680949|25.000000|6.250000|
|vidstg|D1|16.158575|34.885355|35.680949|25.000000|0.000000|
|vidstg|D2|15.046009|38.012582|35.680949|25.000000|6.250000|
|vidstg|QWEN_OPD|15.343443|37.946449|35.680949|25.000000|0.000000|
|vidstg|QWEN_D2|15.043710|38.624391|35.680949|18.750000|6.250000|

| Domain | Contrast | delta vIoU pp | paired t95 pp | + / - / neutral / unknown |
|---|---|---:|---|---|
|hc2|QWEN_OPD_minus_Current_WF|0.199317|[-0.5881623727013414, 0.9867960767210371]|9/6/1/0|
|hc2|QWEN_OPD_minus_R1_FIXED_OPD|0.260277|[-0.3399092822375049, 0.860463373261622]|10/3/3/0|
|hc2|QWEN_OPD_minus_R1_FIXED_D2|3.083536|[-9.530634479645519, 15.697705985545918]|12/2/2/0|
|hc2|QWEN_OPD_minus_D0|0.400459|[-0.07184533006452609, 0.8727632114559007]|10/5/1/0|
|hc2|QWEN_OPD_minus_D1|0.239626|[-0.22231918712745347, 0.7015711402776179]|10/5/1/0|
|hc2|QWEN_D2_minus_R1_FIXED_D2|4.013525|[-0.6863838586299886, 8.713434034773293]|10/4/2/0|
|hc2|QWEN_D2_minus_D2|-2.861648|[-11.006783724917643, 5.283488666398149]|5/10/1/0|
|hc2|QWEN_D2_minus_QWEN_OPD|0.929989|[-10.447703225954399, 12.307681896197305]|4/10/2/0|
|vidstg|QWEN_OPD_minus_Current_WF|3.270944|[-0.41567759660168546, 6.9575659476228235]|9/2/5/0|
|vidstg|QWEN_OPD_minus_R1_FIXED_OPD|0.770862|[-0.4016444636307577, 1.9433692533297113]|4/6/6/0|
|vidstg|QWEN_OPD_minus_R1_FIXED_D2|2.121439|[-3.144924430969646, 7.387803366165592]|8/4/4/0|
|vidstg|QWEN_OPD_minus_D0|-0.025355|[-2.338235423932781, 2.2875255412658135]|5/7/4/0|
|vidstg|QWEN_OPD_minus_D1|-0.815132|[-2.1342545568824898, 0.503989802505681]|6/6/4/0|
|vidstg|QWEN_D2_minus_R1_FIXED_D2|1.821707|[-3.1968526506026897, 6.8402662269623224]|8/4/4/0|
|vidstg|QWEN_D2_minus_D2|-0.002299|[-2.9119964722717375, 2.907397953565604]|6/6/4/0|
|vidstg|QWEN_D2_minus_QWEN_OPD|-0.299733|[-3.3292025844911373, 2.7297372256548242]|4/7/5/0|

RESULTS.json includes all paired sources, nonanchor geometry and usage. REFERENCE_QUALITY.json separates overall known-position success from conditional IoU and unknowns. These geometric proxies do not replace semantic identity/role review; the original semantic512 panel remains44 model-assisted/468 pending.


Recovery003: the first four raw outputs and once-only claims are unchanged. Two envelope variants returned a unique bbox payload: one list, one object lacking only status. The declared normalized1000 coordinates were losslessly decoded under one GT-free unique-payload compatibility rule, without clamping, unit guessing or a new Prompt. Multi-box/duplicate-key/invalid/truncated output remains invalid. 124 remaining unique requests executed,128 total. Original scientific deadline and old failure retained. No response was regenerated.


Recovery005 corrects an assertion that mistakenly required approximate context timing. Twelve contexts are exactly uniformly spaced (flag False), which is legal; processor inputs/fps/timestamps are unchanged. All128 inputs were CPU preflighted, and every26 existing actual tensor hash matches. One claim had stopped in preprocessing before model.generate; it resumes once. Only102 remaining unique generations executed. All failures and previous outputs/deadlines preserved.
