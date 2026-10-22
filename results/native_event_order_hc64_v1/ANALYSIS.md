# Joint first/last subevent boundary decisions with narrative order

Fixed cache-only follow-up, exposed HC64. Same52 main-AND queries, unchanged native
positive clause outputs/full captions/space/expert8. No model/expert/optimizer call.
Prior unordered hull improves average time but short events lose5.7702pp. Concrete
RC014,RC054,RC026 show the standalone final action selecting an earlier occurrence,
which unordered min/max can incorrectly span. Marginal first-start error is NOT
inherently lower; this test depends on jointly resolving occurrences using order.

Condition interval pairs on startFirst<=startLast AND endFirst<=endLast, permitting
overlap. Sum all legal ordered pairs into[startFirst,endLast], retaining full probability
mass within the fixed relation. Primary mixes half original Full, half conditional
ordered posterior; original physical MBR/complete-box DP. Valid-order mass is logged,
never a rejection gate. Narrative AND is not guaranteed chronological; this is an
explicit semantic assumption tested here, not supplied by a new event expert.

DirectOrdered uses the same information. RoleOnly omits inner endpoint order;
SwappedOrder exchanges semantic first/last roles. SelfOrder uses two copies of the
original posterior and controls new structural/length effects. Prior unordered hull
and self-hull remain, as does exact no-composition. No weight or threshold sweep.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|OrderedClauseFusion|31.0088|57.9005|28/23/13|11|
|DirectOrdered|30.7308|57.4003|27/25/12|19|
|RoleOnlyFusion|30.9113|58.1135|27/20/17|9|
|SwappedOrderFusion|30.9381|57.8953|31/19/14|12|
|SelfOrderFusion|30.4658|57.5333|34/17/13|6|
|ClauseFusion|31.3453|58.4307|30/21/13|11|
|SelfHullFusion|30.5252|57.4780|33/18/13|7|
|NoComposition|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−NativeG6: v+1.3136pp CI[-0.6734722372481238, 3.1664546351572485], t+1.9392pp CI[-1.1823250240640015, 4.899512825626509]. Primary−Source: v+4.1150pp, t+3.0775pp. Primary−SelfOrderFusion: t+0.3672pp CI[-2.33309246937511, 2.8359291099916137]. Shortt-6.4844pp, longt+4.0863pp versusNativeG6.

Support52/64, all remaining outputs exactNative. Matched feedback accuracy56.8157% including unsupported ties, native52.1362%; supported-only0.5838853485357076.

Primary does not pass the broad-improvement development screen. Preserve positive means and all short-event/severe harms; no promotion or order-strength scan.

All64 parent aliases/Source/NativeG6 restore bitwise, no-composition exact. Official/dense checks1280, maxerror6.66e-16. Original48+434 sources intact. Worker3.81s CPU cached arithmetic, calls{'new_RGB': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_decoder': 0, 'new_expert': 0, 'new_optimizer': 0}; not end-to-end deployment time. All new outputs sealed before labels/scoring.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).
