# Sparse STVG-R1 candidate verification: finite P0 screen

The fixed eight-case / 48-request input screen completed, but did not qualify for P1/P2 expansion. The frozen STVG-R1-7B evaluator used student-generated candidate IDs, with no Grounding DINO, YOLO or SAM2. This result applies to one sparse input protocol, not to every MLLM or OPD method.

## Actual scope

- Cold cross-domain TA-STVG source predictions were reused. Four HC-STVG2 and four VidSTG cases were selected by existing source order, before current teacher outputs or GT diagnostics.
- Each request used eight uniformly sampled frames over the complete legal student input frame grid (448 x 448 letterboxed). Two comparisons per query: four temporal intervals with fixed Native boxes, and four spatial trajectories with fixed Projection interval.
- Temporal candidates: Native, Actionness Projection, and two bounded interval perturbations. Spatial candidates: Native and three controlled nearby box perturbations. This does not add a different object missing from the student's candidate pool.
- Each branch ran base, geometry-preserving ID permutation, and a deterministic next-query control. 8 x 2 x 3 = 48 actual teacher calls. Wrong queries are distribution-shift controls, not automatically false annotations.
- STVG-R1-7B revision c0db00d: frozen, unquantized BF16, SDPA, greedy, 4096 input cap / 1024 output cap. The output cap was increased before the first actual output to accommodate the requested structured JSON. No prompt or hyperparameter sweep followed the results.
- All responses and source/runtime bindings were sealed before any current GT score was opened. The model selected candidates without GT. Offline candidate Oracle was diagnostic only.
- This is a historically exposed development cohort. Teacher target-training overlap is unresolved. It is not a fresh unseen confirmation set.

## Format readback and ID controls

The original strict bare-JSON parser accepted 0/48; all raw outputs had exact JSON fences. Lossless fence removal plus the complete originally required schema accepted 12/48. A separate minimal complete-JSON, legal top-choice readback accepted 40/48, without completing missing candidate fields or repairing candidate IDs. It was declared after the output-format issue and before GT scoring; original strict failures remain preserved. These tiers are distinct, not a relabeling of full-schema qualification.

16 geometry-preserving ID-swap pairs: 12 comparable, 9 identical top geometries, 3 changed, 4 unavailable. HC-STVG2 had 5/8 identical choices and VidSTG 4/8, below the prelocked requirement of 6/8 per domain. HC temporal was 4/4 consistent; HC spatial had 1 same / 2 changed / 1 unavailable. Missing evidence-frame slot units and omitted evaluations remain unknown. Sparse-frame visual rationales do not establish dense grounding truth.

## Actual base-query quality

Changes use only the matched subset with an actual legal teacher choice; unavailable cases are not assigned zero. Pool headroom uses all four cases per domain and is an offline GT upper bound.

| Dataset | Branch | Legal choices | vIoU change vs matched Projection (pp) | Pool Oracle headroom (pp) |
| --- | --- | ---: | ---: | ---: |
| HC-STVG2 | Temporal | 4/4 | -2.48 | +3.70 |
| HC-STVG2 | Spatial | 4/4 | -1.62 | +0.58 |
| VidSTG | Temporal | 3/4 | -1.23 | +3.17 |
| VidSTG | Spatial | 2/4 | 0.00 | +1.10 |

Projection mean vIoU: 27.85% HC-STVG2; 22.02% VidSTG. Both legal VidSTG spatial choices selected the Native spatial trajectory while retaining Projection time. A Native fallback includes loss of Projection time; its total outcome must not be attributed solely to teacher ranking. Complete raw-choice and semantically admitted pseudo-reference results, including no-information fallback, are separate in the anonymous rows.

192 scores = 48 requests x 4 candidate GT diagnostic readouts. They are not 192 new fits or 192 complete teacher evaluations. No full-rank correlation, statistical significance or parameter-adaptation benefit is claimed. Official and independent dense metric arithmetic agrees within 1.23e-15. Continuous candidate intervals were retained; official frame scoring uses its original integer frame-bound API.

## Cost and resource outcome

48 requests: preprocessing + generation 276.90 s; generation 275.70 s. Median 3.89 s/request, range 2.49-11.65 s. Input/output tokens 126,912 / 17,124. Maximum allocated GPU memory 17,780,548,608 bytes (16.56 GiB); model load 5.48 s. The completed run had no OOM or resource-guard stop.

This is cached-student-input verifier timing, not end-to-end TTA latency. Cold student forward, input extraction/rendering, adaptation, final decode, and report/custody work are not all included. The two normal calls per query and four additional P0 control calls are fully recorded. No new student source capture, fit, backward, optimizer update, DINO, YOLO or SAM2 call occurred.

## Decision and retained implementation

`stop_current_sparse_ID_protocol_before_P1_P2`. P1/P2/P3 were not executed. Student-selected pseudo-reference must not be described as independent teacher geometry. Spatial Gaussian OPD / matched Direct and lightweight temporal residual interfaces passed synthetic CPU contracts, but their actual task effect remains unmeasured. Query/LN/carry interfaces and deployed CURRENT were retained; no promotion occurred.

Old full native tracking queue was explicitly stopped with all accepted assets/partial logs preserved. A library-introspection guard initially rejected Transformers' scanning of Python sources; it failed before model/output/GT, and a separately pinned v2 allowed library source inspection while blocking actual tracker/DINO assets. Postseal reader revisions separated rendered/body prompt hashes, display-rounded/full scientific coordinates, ID permutation geometry, and discrete metric APIs. Original versions and failures remain private. These are engineering corrections, not new science measurements.

Public exports exclude queries, free-form model notes, RGB/video, annotations/GT coordinates, source .pt caches, model weights, credentials, and personal conversations. Code, actual protocol, full anonymous positive/negative/unknown scalar evidence, opaque byte bindings, PDF and root receipts are included. Private execution requires the authorized local source assets; this export does not redistribute datasets or weights.

Files: `anonymous_responses.json` (48), `ID_swap_controls.json` (16), `wrong_query_categorical_evidence.json` (43 raw supplied evaluations: 41 original postseal evidence rows, 2 unqualified supplemental rows), `summary.json`, `SOURCE_BINDINGS.json`, `OUTPUT_SCHEMA.json`, two-page PDF and rendered PNGs. Raw/private evidence remains bound to the sealed local run.

Two VidSTG spatial wrong-query responses contained raw categorical evaluations but failed the minimal selected-ID contract. Their two remaining candidate entries are retained as unqualified raw supplements (`postseal_reader_evidence_eligible=false`), not converted into valid model choices or verified visual evidence. Original postseal report has41 eligible evidence rows; full raw categorical export has43.
