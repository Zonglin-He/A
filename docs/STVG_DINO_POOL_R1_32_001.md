# Same cached DINO pool: Top1 versus native R1 selector32

Historical exposed HC16/Vid16;38 fixed common positions/19 supported sources,13 unsupported no-op sources retained. Exact cached processed_probe three-candidate pool per position and entity target scores. Top1 is ID1, then fixed descending target score/expert ID; possible position bias is disclosed. No new DINO. R1 sees native full1280 visual tokens, old crop Prompt/context4/local5 and original physical uniform-clock adapter; it is not compressed A or current B. Caption-independent, one fixed selector, no Prompt/Token search. All local pixels match historical SHA, center RGB also matches actual DINO input. Existing KLT only renders adjacent crops; actual center targets remain exact pool members.

The first A160 attempt completed5WHO/3short qualification calls/6encoders, then stopped on its sixth job: valid real crop cells were insufficient for its fixed quota. Those partial A decisions are not mixed into the native comparison. A second native attempt stopped on an old receipt collision after3short calls/1encoder and0WHO. Failure/pins/logs are retained and directories isolated; native1280 avoids inventing observations or a new Token allocation rule. All extra cost and the final6short qualification calls are separately counted. Earlier zero-WHO wording was corrected.

Same episodic Source/Adam/query RNG,1792 parameters,32Gaussian samples and original sparse-reference acceptance/rollback/reject-stop under both historical current5/original packages. Top1 cells exact whole-query reuse; identical entire R1 references also reuse Top1 exactly. Different/new R1 references are actually fit and full outputs recomputed. Invalid/abstained R1 references are omitted, not replaced with GT or Top1; all sources stay in the denominator. All predictions/ROOT/Global barriers preceded GT.

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|F|32.587386|53.922657|54.448394|50.000000|31.250000|
|hc2|T|34.066991|53.922657|60.145183|50.000000|31.250000|
|hc2|S_Top1_SHORT|33.241644|54.698718|54.448394|50.000000|37.500000|
|hc2|TS_Top1_SHORT|34.741339|54.698718|60.145183|50.000000|43.750000|
|hc2|S_Top1_ORIGINAL|31.844646|51.864243|54.448394|50.000000|31.250000|
|hc2|TS_Top1_ORIGINAL|32.799812|51.864243|60.145183|50.000000|31.250000|
|hc2|S_R1_SHORT|33.231038|54.688937|54.448394|50.000000|37.500000|
|hc2|TS_R1_SHORT|34.725325|54.688937|60.145183|50.000000|43.750000|
|hc2|S_R1_ORIGINAL|32.059457|52.101700|54.448394|50.000000|31.250000|
|hc2|TS_R1_ORIGINAL|32.983216|52.101700|60.145183|50.000000|31.250000|
|vidstg|F|9.701443|33.353589|33.628222|12.500000|0.000000|
|vidstg|T|9.966696|33.353589|34.185907|12.500000|0.000000|
|vidstg|S_Top1_SHORT|10.630311|37.009485|33.628222|12.500000|0.000000|
|vidstg|TS_Top1_SHORT|10.924565|37.009485|34.185907|12.500000|0.000000|
|vidstg|S_Top1_ORIGINAL|12.845797|39.955245|33.628222|18.750000|6.250000|
|vidstg|TS_Top1_ORIGINAL|13.001229|39.955245|34.185907|18.750000|6.250000|
|vidstg|S_R1_SHORT|10.410598|36.510420|33.628222|12.500000|0.000000|
|vidstg|TS_R1_SHORT|10.718390|36.510420|34.185907|12.500000|0.000000|
|vidstg|S_R1_ORIGINAL|11.615104|39.188778|33.628222|12.500000|6.250000|
|vidstg|TS_R1_ORIGINAL|11.899663|39.188778|34.185907|12.500000|6.250000|

| Domain | Fixed-package contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |
|---|---|---:|---|---|
|hc2|S_R1_SHORT_minus_S_Top1_SHORT|-0.010605|[-0.036903630640793154, 0.01569315281752958]|1/1/14/0|
|hc2|TS_R1_SHORT_minus_TS_Top1_SHORT|-0.016014|[-0.04392503767260127, 0.01189704802637822]|0/2/14/0|
|hc2|S_R1_ORIGINAL_minus_S_Top1_ORIGINAL|0.214811|[-0.11383428343303564, 0.5434556330022393]|2/0/14/0|
|hc2|TS_R1_ORIGINAL_minus_TS_Top1_ORIGINAL|0.183403|[-0.08447705029124819, 0.45128358202769236]|2/0/14/0|
|vidstg|S_R1_SHORT_minus_S_Top1_SHORT|-0.219713|[-0.6880191894056382, 0.24859381825040994]|0/1/15/0|
|vidstg|TS_R1_SHORT_minus_TS_Top1_SHORT|-0.206175|[-0.594566502040109, 0.1822169798916008]|0/2/14/0|
|vidstg|S_R1_ORIGINAL_minus_S_Top1_ORIGINAL|-1.230693|[-3.853853023535743, 1.3924670312237462]|0/1/15/0|
|vidstg|TS_R1_ORIGINAL_minus_TS_Top1_ORIGINAL|-1.101566|[-3.5310194889558355, 1.3278865236653703]|1/1/14/0|

Selector decisions (all38, including invalid/0) and per-source update/reuse counts are preserved in RESULTS.json. Identity/role correctness cannot be inferred solely from box IoU or vIoU; no independent human semantic annotation is asserted.

All pairwise source deltas, signs, unknowns and intervals for all metrics are in RESULTS.json; t95 is descriptive on this exposed small development sample. The support is conditional on historical DINO admission. This tests the additional effect of a fixed native R1 selector on the same expert candidate pool within each configuration package, not current-B superiority or long-stream online state. Sparse-reference acceptance is not GT correctness. Post-seal observed-position reference IoU and selection coverage are in REFERENCE_QUALITY.json; missing GT and abstention remain distinct, with no identity/role-label claim. Actual pixel review is engineering inspection, not completion of the512-window semantic panel. Production CURRENT and frozen paper registry remain unchanged.

Measured interpretation: this fixed development cohort does not establish a benefit from the native R1 selector over expert Top1 under either package. TS R1−Top1 is HC−.016014/Vid−.206175pp under short and HC+.183403/Vid−1.101566pp under original; all four paired t95 intervals include zero.38 valid parses yield29Top1 selections,7alternative selections and2abstentions.26whole-query outputs are exact Top1 reuse (including13unsupported sources), and only6changed-reference sources require12newfits. Among actually annotated reference positions, HC15 and Vid9, one alternative per domain lowers reference IoU and the rest are unchanged; remaining positions are unknown, not automatically incorrect. No independent identity/role labels are asserted. This is a bounded negative result for this pool/input/selector, not proof that all MLLM selection is useless. Do not add more selection modules or promote this selector; preserve the simpler geometry-expert comparison and R1 Temporal route, with production CURRENT and frozen paper unchanged.

Invocation-accounting correction:38 is the scientific job pack encoder count, not the total visual encodes including qualification. Two native model.generate qualification calls re-encode their inputs; the successful native attempt therefore uses40visual encodes. Earlier A partial and receipt-collision attempts use7 and2 respectively, for49total. The5partial A WHO and12total short qualification generations are separately disclosed (3+3+6). These counts are read back from the actual executed source paths and saved receipts; no fresh end-to-end speedup is claimed.
