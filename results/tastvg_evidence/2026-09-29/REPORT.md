# TA-EV Round1 Full64 — completed / independently read back

**Scope:** 64 fixed historical development parents, 32 per cross-domain direction; strict scientific workers opened no GT, no expert, no model parameter update. PTD remains paused. Round2/3 not run.

## Main readout

|rho|nonzero feasible selection|strong shifted & preserved|terminal strict|random strict|selected mean self-vIoU|
|---|---:|---:|---:|---:|---:|
|0.005|64/64|4/64|58/64|63/64|0.99043903|
|0.01|64/64|12/64|53/64|63/64|0.98660183|
|0.02|64/64|22/64|40/64|61/64|0.98494211|

Unique strong-preserved parents: **25/64**; by direction: `{'hcstvg1_test': 7, 'vidstg_test': 18}`. Strong cases beating feasible random: **25**. Pilot rule on full panel (descriptive): **True**.

The selected result maximizes the declared drift surrogate among strictly feasible visited states including untouched baseline; it is a finite-search lower bound on vulnerability. The terminal is separately reported. High selected preservation alone is tautological and is NOT attack success.

## Evidence components (mean across offsets, then equal parents)

|Metric|rho .005|rho .01|rho .02|
|---|---:|---:|---:|
|TTS_app_JSD|0.00000057|0.00000197|0.00000586|
|TTS_motion_JSD|0.00000001|0.00000020|0.00000220|
|selected_stage1_jaccard|0.99975586|0.99897461|0.99919782|
|selected_stage2_jaccard|0.99708452|0.97750856|0.98583764|
|ASA1_app_cosine_drift|0.01381171|0.02778643|0.05131790|
|ASA1_motion_cosine_drift|0.00093847|0.00701135|0.01797590|
|ASA2_app_cosine_drift|0.00569738|0.01408061|0.03597918|
|ASA2_motion_cosine_drift|0.00009853|0.00125406|0.00765826|
|ASA1_app_top20_iou|0.98145946|0.96780534|0.95499258|
|ASA1_motion_top20_iou|0.99819345|0.98415468|0.97055401|
|ASA2_app_top20_iou|0.99058955|0.98168173|0.96863030|
|ASA2_motion_top20_iou|0.99958689|0.99567960|0.98590926|
|Qs1_cosine_drift|0.00037334|0.00075992|0.00149822|
|Qt1_cosine_drift|0.00003674|0.00031636|0.00073342|
|Qs2_cosine_drift|0.00016188|0.00116113|0.00305533|
|Qt2_cosine_drift|0.00003074|0.00044200|0.00080457|

## Full anonymous per-query selected results

|Query|Direction|rho|selected step (-1 = original)|relative norm|self-vIoU|strong shift|terminal strict|
|---|---|---:|---:|---:|---:|---|---|
|Q01|Vid→HC1|0.005|9|0.0009668|0.9977698|False|True|
|Q01|Vid→HC1|0.01|8|0.0019155|0.9977815|True|True|
|Q01|Vid→HC1|0.02|9|0.0054479|0.9978159|False|False|
|Q02|Vid→HC1|0.005|10|0.0006478|0.9982287|False|True|
|Q02|Vid→HC1|0.01|9|0.0018452|0.9962466|False|True|
|Q02|Vid→HC1|0.02|7|0.0049677|0.9823529|True|True|
|Q03|Vid→HC1|0.005|10|0.0006763|0.9955449|False|True|
|Q03|Vid→HC1|0.01|9|0.0020221|0.9650493|False|False|
|Q03|Vid→HC1|0.02|0|0.0002000|0.9999853|False|False|
|Q04|Vid→HC1|0.005|4|0.0011924|0.9729500|False|True|
|Q04|Vid→HC1|0.01|9|0.0027448|0.9673598|False|True|
|Q04|Vid→HC1|0.02|0|0.0002000|0.9999076|False|False|
|Q05|Vid→HC1|0.005|9|0.0013501|0.9687257|False|True|
|Q05|Vid→HC1|0.01|8|0.0035671|0.9650347|False|False|
|Q05|Vid→HC1|0.02|0|0.0002000|0.9999695|False|False|
|Q06|Vid→HC1|0.005|7|0.0009249|0.9943093|False|True|
|Q06|Vid→HC1|0.01|1|0.0020019|0.9914054|False|True|
|Q06|Vid→HC1|0.02|9|0.0035376|0.9810819|False|True|
|Q07|Vid→HC1|0.005|9|0.0011592|0.9981076|False|False|
|Q07|Vid→HC1|0.01|10|0.0025619|0.9986429|False|True|
|Q07|Vid→HC1|0.02|1|0.0040046|0.9959700|True|False|
|Q08|Vid→HC1|0.005|8|0.0012988|0.9952700|False|True|
|Q08|Vid→HC1|0.01|8|0.0032541|0.9943991|False|True|
|Q08|Vid→HC1|0.02|9|0.0068508|0.9923774|True|False|
|Q09|Vid→HC1|0.005|0|0.0000500|0.9999751|False|False|
|Q09|Vid→HC1|0.01|0|0.0001000|0.9999440|False|False|
|Q09|Vid→HC1|0.02|0|0.0002000|0.9998832|False|False|
|Q10|Vid→HC1|0.005|9|0.0013275|0.9994837|False|True|
|Q10|Vid→HC1|0.01|7|0.0028518|0.9992167|False|True|
|Q10|Vid→HC1|0.02|7|0.0068816|0.9990252|False|True|
|Q11|Vid→HC1|0.005|0|0.0000500|0.9999845|False|False|
|Q11|Vid→HC1|0.01|0|0.0001000|0.9999616|False|False|
|Q11|Vid→HC1|0.02|0|0.0002000|0.9999306|False|False|
|Q12|Vid→HC1|0.005|3|0.0008016|0.9739080|False|True|
|Q12|Vid→HC1|0.01|4|0.0013307|0.9796391|False|True|
|Q12|Vid→HC1|0.02|8|0.0046158|0.9545376|False|False|
|Q13|Vid→HC1|0.005|0|0.0000500|0.9999871|False|False|
|Q13|Vid→HC1|0.01|0|0.0001000|0.9999809|False|False|
|Q13|Vid→HC1|0.02|0|0.0002000|0.9999604|False|False|
|Q14|Vid→HC1|0.005|3|0.0009136|0.9929563|False|True|
|Q14|Vid→HC1|0.01|9|0.0019256|0.9907224|True|False|
|Q14|Vid→HC1|0.02|9|0.0046361|0.9856584|True|False|
|Q15|Vid→HC1|0.005|9|0.0008495|0.9951172|False|True|
|Q15|Vid→HC1|0.01|9|0.0017738|0.9888247|False|True|
|Q15|Vid→HC1|0.02|4|0.0019540|0.9968651|False|False|
|Q16|Vid→HC1|0.005|9|0.0008858|0.9776328|False|True|
|Q16|Vid→HC1|0.01|9|0.0018545|0.9592711|False|True|
|Q16|Vid→HC1|0.02|10|0.0028431|0.9697858|False|True|
|Q17|Vid→HC1|0.005|1|0.0010010|0.9986535|False|True|
|Q17|Vid→HC1|0.01|1|0.0020021|0.9973805|False|True|
|Q17|Vid→HC1|0.02|7|0.0054940|0.9943846|False|True|
|Q18|Vid→HC1|0.005|3|0.0008246|0.9812417|False|True|
|Q18|Vid→HC1|0.01|10|0.0019030|0.9794668|False|True|
|Q18|Vid→HC1|0.02|10|0.0056460|0.9845658|False|True|
|Q19|Vid→HC1|0.005|9|0.0008600|0.9959197|False|True|
|Q19|Vid→HC1|0.01|3|0.0019221|0.9922752|False|True|
|Q19|Vid→HC1|0.02|9|0.0033141|0.9971911|False|True|
|Q20|Vid→HC1|0.005|3|0.0008899|0.9806738|False|True|
|Q20|Vid→HC1|0.01|9|0.0018959|0.9937255|False|True|
|Q20|Vid→HC1|0.02|10|0.0054919|0.9759381|False|True|
|Q21|Vid→HC1|0.005|9|0.0010047|0.9918557|False|True|
|Q21|Vid→HC1|0.01|10|0.0022169|0.9873798|False|True|
|Q21|Vid→HC1|0.02|10|0.0055936|0.9685647|False|True|
|Q22|Vid→HC1|0.005|1|0.0010010|0.9959954|False|True|
|Q22|Vid→HC1|0.01|9|0.0019556|0.9932977|False|True|
|Q22|Vid→HC1|0.02|4|0.0032246|0.9882601|False|True|
|Q23|Vid→HC1|0.005|8|0.0018532|0.9722067|False|True|
|Q23|Vid→HC1|0.01|10|0.0045660|0.9669893|False|True|
|Q23|Vid→HC1|0.02|10|0.0089629|0.9676870|True|True|
|Q24|Vid→HC1|0.005|7|0.0012861|0.9955468|False|True|
|Q24|Vid→HC1|0.01|10|0.0027230|0.9943882|False|True|
|Q24|Vid→HC1|0.02|10|0.0067697|0.9866246|False|True|
|Q25|Vid→HC1|0.005|10|0.0012812|0.9978826|False|True|
|Q25|Vid→HC1|0.01|7|0.0026742|0.9962553|False|True|
|Q25|Vid→HC1|0.02|10|0.0078398|0.9900624|False|True|
|Q26|Vid→HC1|0.005|5|0.0008932|0.9902519|False|True|
|Q26|Vid→HC1|0.01|8|0.0021697|0.9814424|False|True|
|Q26|Vid→HC1|0.02|9|0.0059081|0.9844459|False|True|
|Q27|Vid→HC1|0.005|1|0.0010010|0.9698541|False|True|
|Q27|Vid→HC1|0.01|8|0.0017270|0.9632196|False|True|
|Q27|Vid→HC1|0.02|0|0.0002000|0.9999537|False|False|
|Q28|Vid→HC1|0.005|7|0.0011663|0.9678442|False|True|
|Q28|Vid→HC1|0.01|7|0.0028108|0.9568417|False|True|
|Q28|Vid→HC1|0.02|9|0.0072672|0.9718802|False|True|
|Q29|Vid→HC1|0.005|5|0.0009381|0.9815186|False|True|
|Q29|Vid→HC1|0.01|9|0.0017239|0.9738660|False|True|
|Q29|Vid→HC1|0.02|6|0.0015993|0.9872803|False|False|
|Q30|Vid→HC1|0.005|1|0.0010011|0.9987654|False|True|
|Q30|Vid→HC1|0.01|7|0.0021126|0.9962302|False|True|
|Q30|Vid→HC1|0.02|9|0.0051410|0.9949351|False|True|
|Q31|Vid→HC1|0.005|10|0.0015150|0.9989543|False|True|
|Q31|Vid→HC1|0.01|10|0.0042782|0.9937686|False|True|
|Q31|Vid→HC1|0.02|10|0.0081177|0.9742707|True|True|
|Q32|Vid→HC1|0.005|1|0.0010011|0.9988006|False|True|
|Q32|Vid→HC1|0.01|3|0.0020165|0.9966804|False|True|
|Q32|Vid→HC1|0.02|1|0.0040044|0.9977397|False|True|
|Q33|HC2→Vid|0.005|10|0.0025887|0.9970289|True|True|
|Q33|HC2→Vid|0.01|9|0.0030250|0.9977740|True|False|
|Q33|HC2→Vid|0.02|3|0.0042573|0.9940241|False|False|
|Q34|HC2→Vid|0.005|7|0.0008740|0.9584448|False|True|
|Q34|HC2→Vid|0.01|10|0.0013769|0.9778601|False|True|
|Q34|HC2→Vid|0.02|0|0.0002000|0.9999430|False|False|
|Q35|HC2→Vid|0.005|9|0.0009290|0.9924272|False|True|
|Q35|HC2→Vid|0.01|10|0.0007633|0.9968663|False|True|
|Q35|HC2→Vid|0.02|10|0.0023348|0.9947747|False|True|
|Q36|HC2→Vid|0.005|10|0.0006177|0.9947750|False|True|
|Q36|HC2→Vid|0.01|10|0.0015911|0.9916164|False|True|
|Q36|HC2→Vid|0.02|10|0.0046765|0.9679107|True|True|
|Q37|HC2→Vid|0.005|10|0.0009554|0.9961286|False|True|
|Q37|HC2→Vid|0.01|10|0.0022676|0.9943738|False|True|
|Q37|HC2→Vid|0.02|10|0.0055214|0.9811713|True|True|
|Q38|HC2→Vid|0.005|10|0.0002836|0.9989852|False|True|
|Q38|HC2→Vid|0.01|10|0.0007430|0.9978021|False|True|
|Q38|HC2→Vid|0.02|10|0.0028165|0.9905644|True|True|
|Q39|HC2→Vid|0.005|10|0.0009368|0.9933821|False|True|
|Q39|HC2→Vid|0.01|10|0.0018356|0.9828775|True|True|
|Q39|HC2→Vid|0.02|4|0.0021623|0.9771220|False|False|
|Q40|HC2→Vid|0.005|9|0.0009495|0.9896410|False|True|
|Q40|HC2→Vid|0.01|5|0.0018754|0.9805803|False|False|
|Q40|HC2→Vid|0.02|3|0.0037372|0.9612559|False|False|
|Q41|HC2→Vid|0.005|9|0.0012358|0.9917665|False|True|
|Q41|HC2→Vid|0.01|10|0.0026172|0.9872116|False|True|
|Q41|HC2→Vid|0.02|9|0.0056246|0.9830883|True|False|
|Q42|HC2→Vid|0.005|1|0.0010010|0.9907829|False|True|
|Q42|HC2→Vid|0.01|9|0.0018897|0.9835305|False|True|
|Q42|HC2→Vid|0.02|10|0.0024922|0.9918590|True|True|
|Q43|HC2→Vid|0.005|9|0.0028977|0.9741506|False|True|
|Q43|HC2→Vid|0.01|10|0.0050224|0.9744090|True|True|
|Q43|HC2→Vid|0.02|10|0.0099898|0.9730452|True|True|
|Q44|HC2→Vid|0.005|10|0.0014279|0.9886036|False|True|
|Q44|HC2→Vid|0.01|10|0.0026214|0.9678388|False|True|
|Q44|HC2→Vid|0.02|10|0.0066611|0.9720595|False|True|
|Q45|HC2→Vid|0.005|3|0.0008661|0.9806342|False|True|
|Q45|HC2→Vid|0.01|1|0.0020019|0.9507270|False|True|
|Q45|HC2→Vid|0.02|9|0.0040590|0.9535850|False|False|
|Q46|HC2→Vid|0.005|10|0.0011440|0.9987817|True|True|
|Q46|HC2→Vid|0.01|10|0.0021377|0.9968471|True|True|
|Q46|HC2→Vid|0.02|10|0.0045588|0.9936460|True|True|
|Q47|HC2→Vid|0.005|10|0.0011757|0.9990528|False|True|
|Q47|HC2→Vid|0.01|10|0.0031409|0.9949802|True|True|
|Q47|HC2→Vid|0.02|10|0.0070681|0.9923531|True|True|
|Q48|HC2→Vid|0.005|9|0.0015585|0.9860346|False|True|
|Q48|HC2→Vid|0.01|10|0.0032078|0.9907773|False|True|
|Q48|HC2→Vid|0.02|10|0.0076121|0.9644962|True|True|
|Q49|HC2→Vid|0.005|0|0.0000500|0.9999577|False|False|
|Q49|HC2→Vid|0.01|0|0.0001000|0.9999406|False|False|
|Q49|HC2→Vid|0.02|0|0.0002000|0.9996168|False|False|
|Q50|HC2→Vid|0.005|10|0.0010755|0.9951954|False|True|
|Q50|HC2→Vid|0.01|9|0.0022501|0.9872425|False|False|
|Q50|HC2→Vid|0.02|10|0.0045473|0.9920590|True|True|
|Q51|HC2→Vid|0.005|10|0.0009595|0.9959090|False|True|
|Q51|HC2→Vid|0.01|10|0.0021488|0.9908914|False|True|
|Q51|HC2→Vid|0.02|10|0.0046343|0.9751255|False|True|
|Q52|HC2→Vid|0.005|10|0.0013275|0.9943531|False|True|
|Q52|HC2→Vid|0.01|10|0.0029027|0.9878472|True|True|
|Q52|HC2→Vid|0.02|10|0.0059540|0.9784015|True|True|
|Q53|HC2→Vid|0.005|9|0.0008378|0.9927592|False|True|
|Q53|HC2→Vid|0.01|9|0.0016501|0.9868600|False|True|
|Q53|HC2→Vid|0.02|10|0.0035306|0.9858392|True|True|
|Q54|HC2→Vid|0.005|9|0.0010552|0.9917542|True|False|
|Q54|HC2→Vid|0.01|9|0.0022064|0.9843660|True|False|
|Q54|HC2→Vid|0.02|9|0.0037131|0.9941787|True|False|
|Q55|HC2→Vid|0.005|1|0.0010010|0.9552075|False|True|
|Q55|HC2→Vid|0.01|10|0.0022605|0.9732976|False|True|
|Q55|HC2→Vid|0.02|7|0.0048687|0.9565540|False|False|
|Q56|HC2→Vid|0.005|10|0.0006167|0.9970739|False|True|
|Q56|HC2→Vid|0.01|10|0.0016834|0.9956945|False|True|
|Q56|HC2→Vid|0.02|10|0.0045032|0.9853413|False|True|
|Q57|HC2→Vid|0.005|10|0.0010722|0.9974721|True|True|
|Q57|HC2→Vid|0.01|10|0.0024001|0.9941797|True|True|
|Q57|HC2→Vid|0.02|10|0.0058187|0.9927467|True|True|
|Q58|HC2→Vid|0.005|10|0.0004499|0.9974812|False|True|
|Q58|HC2→Vid|0.01|10|0.0012499|0.9929943|False|True|
|Q58|HC2→Vid|0.02|10|0.0032514|0.9901284|False|True|
|Q59|HC2→Vid|0.005|10|0.0003308|0.9995800|False|True|
|Q59|HC2→Vid|0.01|10|0.0012143|0.9984935|False|True|
|Q59|HC2→Vid|0.02|6|0.0019669|0.9973206|False|False|
|Q60|HC2→Vid|0.005|9|0.0008119|0.9886837|False|True|
|Q60|HC2→Vid|0.01|10|0.0017521|0.9820522|False|True|
|Q60|HC2→Vid|0.02|10|0.0043710|0.9693014|False|True|
|Q61|HC2→Vid|0.005|1|0.0010010|0.9888695|False|True|
|Q61|HC2→Vid|0.01|10|0.0019878|0.9924503|False|True|
|Q61|HC2→Vid|0.02|10|0.0049651|0.9728397|False|True|
|Q62|HC2→Vid|0.005|10|0.0011929|0.9980914|False|True|
|Q62|HC2→Vid|0.01|10|0.0026343|0.9977036|True|True|
|Q62|HC2→Vid|0.02|10|0.0066873|0.9974924|True|True|
|Q63|HC2→Vid|0.005|10|0.0008020|0.9921842|False|True|
|Q63|HC2→Vid|0.01|10|0.0017841|0.9665663|False|True|
|Q63|HC2→Vid|0.02|10|0.0047063|0.9566275|False|True|
|Q64|HC2→Vid|0.005|10|0.0011811|0.9969901|False|True|
|Q64|HC2→Vid|0.01|10|0.0026345|0.9891784|True|True|
|Q64|HC2→Vid|0.02|10|0.0057778|0.9829338|True|True|

## Validity and limitations

All 64 queries and 192 arms included; 1920 paired-offset optimization backwards. NumPy independently recomputed 15932 quantities; maximum metric error 5.33e-15. All selected text deltas exactly zero and all selected radii bounded. Baseline full-pipeline/historical predictions exact for every capture; selected full-pipeline reinsertions checked for three budgets on the first query of each direction.

GPU-process allocations including imports/loading/capture/failures total **583.785 s (9.73 min)**. CPU coding/tests/scoring/reporting excluded. The first capture failed an exact equality check before attacks; matching the historical second-spatial-pass zero-query materialization repaired it with no tolerance change. All receipts and original source retained.

The baseline uses the established FP16-prefix/FP32-suffix interface. Attack gradients reopen the official TTS/ASA input detach for representation interventions; hard selections and iterative decoder reference detach remain. All native routing is recomputed, and actual output tests decide preservation. This is a white-box latent diagnostic, not an image-space replication of X-Shift.

TTS scores, ASA maps and query vectors are different evidence objects, not calibrated correctness measures. Their drift does not demonstrate semantic misleadingness. If the expansion gate fails, this bounded attack configuration did not establish the desired strong vulnerability; it does not prove the model is robust or all such attacks impossible. No accuracy/TTA gains or vulnerability–correctability correlations were measured.

Primary source: [Right Predictions, Misleading Explanations](https://arxiv.org/html/2605.16651v1). TA-specific implementation and numeric contracts are in `../../../protocols/tastvg_evidence_vulnerability_v1.md`. Complete local per-query data: `ROWS_INDEX.json`; descriptive parent bootstrap: `SUMMARY.json`; independent readback: `AUDIT.json`.

The complete numerical rows and component summaries are included in this directory; the local plot can be regenerated with the published reporting script.

## Disjoint pilot and extension readout

All panels are historically exposed development data; extension48 was not used to choose the attack configuration. The prelocked pilot decision only controlled whether to pay for the remaining48.

|Subset|Unique strong optimized parents|Unique strong random parents|
|---|---:|---:|
|pilot16|9|0|
|extension48|16|0|
|all64|25|0|

Counts below require the same strict preservation and self-vIoU>.95; component categories overlap. TTS threshold=.01 JSD; ASA/query=.10 cosine distance; selection Jaccard<=.5.

|Subset|rho|Strong any|TTS|ASA|Query|Selection|Strong random|
|---|---:|---:|---:|---:|---:|---:|---:|
|pilot16|0.005|1|0|1|0|0|0|
|pilot16|0.01|3|0|2|0|1|0|
|pilot16|0.02|6|0|6|1|1|0|
|extension48|0.005|3|0|3|0|0|0|
|extension48|0.01|9|0|8|0|1|0|
|extension48|0.02|16|0|16|0|0|0|
|all64|0.005|4|0|4|0|0|0|
|all64|0.01|12|0|10|0|2|0|
|all64|0.02|22|0|22|1|1|0|

## Largest ASA changes (descriptive cases, selected post hoc without GT)

|Query|rho|Component/offset|Cosine drift|Top20 patch IoU|Self-vIoU|Selected step|
|---|---:|---|---:|---:|---:|---:|
|Q46|0.02|ASA1_app_cosine_drift/0|0.593079|0.555556|0.993646|10|
|Q47|0.02|ASA1_app_cosine_drift/0|0.517259|0.736842|0.992353|10|
|Q57|0.02|ASA1_app_cosine_drift/0|0.446221|0.711111|0.992747|10|
|Q38|0.02|ASA1_app_cosine_drift/0|0.423254|1.000000|0.990564|10|
|Q54|0.02|ASA1_app_cosine_drift/0|0.371383|0.333333|0.994179|9|

## Scope of the positive finding

The measured claim is existence of internal-evidence sensitivity with a stable final native tube under these finite latent attacks. Native correctness was not scored; stable predictions are not automatically correct predictions. Evidence drift is not by itself evidence that explanations became semantically wrong. High top20 overlap can coexist with large cosine drift because attention mass redistributes among mostly the same high-ranked patches. Report both, rather than interpreting cosine drift as wholesale relocation.

The random control uses the nominal radius and one direction. The optimized selection can have a smaller norm and searches 11 visited states; beating this control is a screening comparison, not proof of optimal attack efficiency or a statistically controlled ranking. Larger budgets are separate finite searches, so selected drift/strong-case counts need not be monotonic.

Additional independent random-control readback: 7872 quantities, max error 6.66e-16; all 1920 saved gradient norms finite, 0 exactly zero; all six saved full-pipeline reinsertion validations passed. All16 loader-isolated pilot baselines have bitwise identical H and evidence to the original loader attempt; numerical identity does not erase the earlier incidental file exposure.

## Constructor I/O correction and regeneration

The initial v1 engineering run incidentally deserialized the official constructor runtime annotation dictionaries. GT values were not used in losses, selection, scoring or threshold choice; the analyst inspected dictionary schema to locate the problem. The run was interrupted with all24 completed arms/failures retained. Every scientific pilot capture/attack was regenerated in no_gt_v2 with empty constructor annotation dictionaries and a process-wide open-denial audit. Only these strict pilot artifacts were reused here; the48 new captures/144 arms used the same guard. The whole-session claim is no GT used for optimization/evaluation, not no incidental annotation file ever opened. Original118.331861s remain charged in the cumulative GPU-process total.

## Timing and completion

The full64 set contains192 attack arms and1920 paired-offset backwards, with480 from the sealed pilot reused exactly once and1440 new backwards. The cumulative process total includes pilot regeneration, old aborted v1, imports/loading, capture, replay checks and attack. CPU development, audits and report generation are separate; the total is not CUDA-event kernel time. All current Round1 GPU workers exited; Round2/3 were not started and PTD remains paused.

## Public artifact scope

All64 are historical exposed development parents. Original source IDs are replaced by fixed Q01-Q64; pilot membership is Q01-Q08 and Q33-Q40. This export includes selected, terminal and random-control scalar metrics for all192 arms, but excludes private media, annotations, raw predictions/features and weights. Local runners require the existing authorized caches/checkpoints; public code alone is not a self-contained dataset download. Configuration/protocol and source/public SHA hashes identify the exact implementation; no new model inference was run during publication.

Public scalar check: `python scripts/audit_tastvg_evidence_public_results_v1.py results/tastvg_evidence/2026-09-29`. This rebuilds counts/mean preservation from the anonymous rows; it is not a replacement for the saved-tensor NumPy audits.
