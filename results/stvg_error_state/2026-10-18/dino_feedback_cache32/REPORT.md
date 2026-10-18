# Conditional DINO feedback scarcity: original32 cached diagnosis

Same exposed HC16/Vid16 as2d4a3fa4. Existing scalar outputs and reference actions only. No model/decoder/processor forward, pretrained weight load, video/image read, official GT read, GT scoring, new candidate, parameter change or alternative prediction. One-at-E24 repeated actions are collapsed to one reference per original anchor. The word admitted means passing the existing score gate, not verified correct geometry or identity.

## Direct census of the unchanged .35 gate

| Domain/arm | Active/16 | Candidate trajectories | Invalid | Legal | With admitted feedback | Returned anchor outputs | Score>=.35 | [.30,.35) | Score median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hc2/One_at_E | 16/16 | 16 | 2 | 14 | 4 | 56 | 4 | 0 | 0.004770 |
| hc2/Fixed24 | 16/16 | 384 | 78 | 306 | 8 | 1224 | 9 | 0 | 0.004298 |
| hc2/Dynamic3 | 16/16 | 384 | 77 | 307 | 8 | 1228 | 9 | 0 | 0.004282 |
| vidstg/One_at_E | 12/16 | 12 | 0 | 12 | 6 | 48 | 7 | 0 | 0.008824 |
| vidstg/Fixed24 | 12/16 | 288 | 12 | 276 | 86 | 1104 | 121 | 3 | 0.006851 |
| vidstg/Dynamic3 | 12/16 | 288 | 14 | 274 | 82 | 1096 | 110 | 5 | 0.006560 |

These are mechanical counts, not a temperature/threshold experiment. HC Dynamic3 has1219 rejected returned outputs and zero in[.30,.35); Vid has986 rejected outputs, only5 in that band. Most HC output scores are around.004 rather than just below.35. A modest cutoff adjustment is not supported by a near-cutoff pileup; no claim is made about the correctness of released low-score boxes. No thresholds or predicted results were changed.

## What the content initialization actually is

Installed snapshot config has embedding_init_target=true,two_stage=true,num_queries=900. HF creates a learned query_position_embeddings table and uses its rows as decoder target content. The encoder top900 proposal order is based on maximum logit across text tokens, not the target-token score. The experimental wrapper chooses the nearest initial native proposal by geometry and copies that slot's learned embedding. It does not copy a query-specific ROI visual descriptor, nor select by target semantic support. This pairing can be incompatible with the student reference; cached scores alone do not isolate the content vector from its associated proposal.

| Dynamic domain | Selected slot0-9 / returned | Passed in0-9 | Selected slot10-899 | Passed there | Known original Top3 slot matches | Originally >=.35 among matched | Still pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| hc2 | 20/1228 | 9 | 1208 | 0 | 16 | 14 | 9 |
| vidstg | 187/1096 | 110 | 909 | 0 | 113 | 105 | 103 |

Native Top3 IDs are reliable only for context_active=true. Fallback ID0 is a placeholder and was excluded from slot matching. A slot absent from Top3 is unknown, not proven wrong. These within-cache associations do not establish that slot>=10 embeddings are intrinsically defective: proposal rank, geometry and content assignment are coupled.

## One exact matching full900 cache: H018

The metadata join found one current-cohort query with old raw native900 output. All four RGB hashes, original text, input tensor hashes, target spans and retained probes exactly match. No other cohort data were added. At its four anchors the native maximum target scores are 0.576067, 0.769284, 0.768997, 0.688666; exactly one of900 native slots per anchor passes.35.

For H018 fixed24 and Dynamic3, all96 selected slots already have original native target score<.35 (maximum 0.033231). Their median original score is 0.003278, versus conditional median 0.003081. Thus this case starts from native slots with weak target support; it is not a case where many originally confident target slots uniformly collapse only after reanchoring. This is one case, not a cohort-wide causal proof. No arbitrary content-swap output exists in cache.

## Student reference position and legality

| Dynamic domain | Invalid trajectories | Individual invalid anchor refs | Otherwise legal anchors discarded with bad trajectory | Fully inside returned refs / admitted | Input IoU>=.75 with cached native final pool / admitted |
|---|---:|---:|---:|---:|---:|
| hc2 | 77 | 179 | 129 | 1041/4 | 44/6 |
| vidstg | 14 | 16 | 40 | 925/86 | 569/94 |

Legality tests every cx,cy,w,h in open(0,1), not whether every corner lies inside the image. The whole trajectory is discarded if any one of its four references fails;129 HC and40 Vid individually legal anchors therefore have no measured conditional output. Their hypothetical feedback is unknown. HC invalid coordinate counts are cx4,cy0,w12,h171 (reasons can overlap), Vid height16 only. No clipping or gate change was tested.

Position matters descriptively: farther from retained native final boxes, admitted feedback is rarer. Yet1041 fully contained HC inputs yield only4 admitted outputs, and38/44 HC inputs with native-pool overlap>=.75 still fail. Boundary violations or reference distance alone do not explain everything. Native retained boxes are detection proxies, not target GT; matching them is not an identity/correctness label.

Holding frame/text/selected content slot fixed, unique saved reference changes do alter scores; some groups cross.35. Same-input repeated scores vary at most 0.000002946, while within-slot ranges can be much larger. Group counts and score ranges are in WITHIN_FIXED_CONTENT.json. These are dependent samples; no causal variance share or independent-sample confidence interval is claimed.

## What is and is not established

The exact loss chain is: no E support -> invalid reference -> low conditional target score -> admitted correction. The gate can be counted exactly. The strongest unresolved interface issue is how geometrically selected native slot/content and the student reference are paired, given their weak native target support. Original DINO decodes its full query pool then selects by text; selecting a geometric slot first changes which evidence is available. This does not prove that a different initialization would work.

Initial900 reference coordinates, copied content vectors, full native scores for31 other queries and same-reference alternative-content outputs are absent. Therefore content versus position cannot be cleanly divided into causal percentages. All-original clone qualification established engineering replay only, not arbitrary-box target correction. No lowered threshold, new content initialization or new teacher pass was tested; current interface remains stopped.

## Integrity and cost

This is CPU-only cache diagnosis, with no new GT metrics or localization performance results. Independent audit verifies every1372 candidate-trajectory census entry,4756 returned scalar references and196 selected-slot matches against the one native900 cache, plus all6 domain/arm summaries. All32 queries and unknown states retained. Source/current method/old prediction pins are unchanged. Public export excludes raw input boxes, original Query/parent strings, GT coordinates, images, weights and original tensors.

## Stronger geometry check against already high-scored native candidates

Using only the original .35 support cutoff and the prelocked .75 overlap bin,35 HC Dynamic3 student references overlap a cached high-scored native final box by>=.75. Of these,6 select native slots0-9 and all6 pass;29 select later slots and none pass. Vid has495 such references:136 select0-9 (94 pass),359 later slots (none pass). This reduces the explanation that every low score occurs only because the student is far from every plausible native detection. It still does not hold the exact reference constant while swapping content, so it does not prove content causality or that alternative content would improve grounding. No GT was used.

The native900 queries are still decoded as frozen context inside the modified interface. What changes is which external query output supplies the feedback: the wrapper copies a geometrically chosen slot and reads that added query's result, rather than selecting a different high target-scoring native query as the correction. Thus this is not a claim that only one query was computationally decoded.
