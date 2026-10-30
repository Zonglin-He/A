# Round5 frozen official-baseline runner contract

Code ready: `common.py` SHA256 e0e50243e076e8491a61e5b6e1c826f22fbf3644d17c818a2e86645f11a66084; `run_frozen.py` SHA256 55fd844b96e99791f59f3af1b99ec9508ed189164dd6259e79e819d4b58c1c39. No GPU job was launched by this preparation agent. The parent owns authorization, immutable run registration and execution. P0 training cancellation remains in force.

## Inputs and population

`inputs/all.json` has 9,706 raw caption/object inputs: Waymo 3,794, Drone 2,984, Quad 2,928. Every `ground_info` object is represented separately, including objects the original class-dependent caption filter would drop. IDs include platform, scene, frame and object index. All ready inputs have captions and point files; no unusable row was silently replaced. Waymo has four split-listed scene directories absent from the released on-disk archive, listed in `PREPARATION_RECEIPT.json`. Thus this is complete available released official-VAL coverage, not recovery of missing author data.

`inputs/smoke48.json` selects the first sixteen salted SHA256 sample IDs in each platform, before model outputs or GT scoring. Preparation opened metadata containers containing labels but extracted only captions and object count/index. `SCORING_INDEX_PRIVATE.json` separately binds metadata path/hash and object index; the GPU manifest has no numeric GT, true class or metadata path.

Model input is one 16,384-point XYZ sample plus either the cleaned full caption or its first fixed-lexicon noun, followed by the original ` . not mentioned` suffix. No RGB is input. Full and noun conditions share identical point bytes and sampled raw indices. The seed is deterministic per expression; this explicitly replaces the author's worker/global RNG order. The geometric preprocessing is unchanged: Waymo front-suffix XY rotation, Drone z+1.8, Quad identity, no additional normalization. B=1 is explicitly registered. Its lack of batch padding can differ from the author's larger dynamic evaluation batch because the original softmax denominator includes actual special/sentinel/padded tokens.

The category-word intervention uses text only. A missing lexicon noun keeps the caption unchanged and is flagged `noun_fallback_identity`; it never uses the target class. It is not guaranteed to equal the target class. Full-VAL actual token lengths across both conditions are 6–106, below the original 256-slot capacity.

## Saved candidate readouts

Each sample writes `{id}_full.npz` and `{id}_noun.npz`. H=7, Q=256, C=6. Head order is `proposal_,0head_,1head_,2head_,3head_,4head_,last_`. Class order is `car,pedestrian,bus,othervehicle,truck,cyclist`.

| Key | Shape and meaning |
|---|---|
| `head_names`, `class_names` | Fixed Unicode arrays identifying axes |
| `boxes_raw` | H×Q×6, untouched center XYZ/full dimensions XYZ |
| `boxes` | H×Q×6, original official evaluation size clamp at 1e-6; no yaw/PCA |
| `bbf_scores_by_class` | H×C×Q, original contrastive token softmax, binary candidate class-mask sum |
| `bbs_scores_by_class` | H×C×Q, original frozen soft-token classifier 256-slot softmax, same mask sum |
| `firstnoun_bbf`, `firstnoun_bbs` | H×Q, actual-input text-only first-noun readout |
| `bbf_native_top10_by_class`, `bbs_native_top10_by_class` | H×C×10, original GPU descending argsort order retained for ties |
| `firstnoun_bbf_native_top10`, `firstnoun_bbs_native_top10` | H×10, native order for text-only readout |
| `class_masks` | C×256, binary map for every predefined candidate class, never selected by actual class |
| `class_synonym_found` | C, original dictionary match existence, the original eligible criterion |
| `class_positive_map_nonzero` | C, separately records nonempty token map; a found-but-empty map remains an original eligible row |
| `class_span_chars` | C×2, chosen span in original leading/trailing-space formatted caption, or -1 |
| `firstnoun_mask256` | Actual text first-noun mask padded false to 256 slots |
| `input_ids`, `attention_mask`, `offset_mapping` | Actual B1 model token IDs/mask/offsets |
| `class_map_input_ids` | Token IDs from the original separately formatted caption used by get_positive_map |
| `point_sample_sha256`, `raw_indices_sha256` | 32-byte arrays proving matched geometry across text conditions |

The source program computes all six fixed candidate class maps from caption text and the official platform synonym dictionary. It does not load the actual target class. Dictionary priority, rather than earliest character occurrence, determines the first synonym match, exactly as released. Leading/trailing spaces and the original `get_positive_map` fallback behavior are preserved even though actual model text uses the sentinel template. The official evaluator binarizes that map; the runner does the same. All mask-score arithmetic uses multiplication then reduction, matching the original evaluator rather than changing the accumulation into matrix multiplication.

Actual GT class selects one candidate score column only after every prediction file is sealed in a separate CPU process. Primary official reporting uses full-caption last/proposal BBF and BBS on the original eligible population. Pure-text full/noun metrics also report all raw objects. The raw noun-input class-assisted map may lack a target-class span; no GT fallback is allowed. Top5/10 and all-candidate maximum IoU are GT-assisted diagnostics, not selectable online output. The author's seven-head mixed mIoU is retained separately from primary last-head mIoU.

## Plan, execution and seal

`RUN_PLAN_TEMPLATE.json` enumerates required fields with GPU permission false. Parent registration supplies true authorization and fixes the manifest, code/dependency/checkpoint hashes, seed, conditions, B1 and expected count. Execute under the existing shared GPU lock:

```
/absolute/workspace/crossplatform/env/bin/python /absolute/workspace/crossplatform/round5/baseline_engineering/run_frozen.py --manifest MANIFEST --output NEW_OUTPUT_DIRECTORY --plan REGISTERED_PLAN --plan-sha256 PLAN_SHA256
```

The output directory must be new. `sealed.json` contains `n`/`n_samples`, conditions, head/class names, manifest/plan/checkpoint hashes, `files={npz_basename:sha256}`, strict state-entry count and frozen parameter/buffer checks. It also binds point provenance, trace, repeat and native readout bridge receipts. All 1,034 checkpoint state entries are strict-loaded and finite. No optimizer is created.

The first sample is repeated once, checking all saved numeric readouts/candidate boxes to <=1e-6 and recording bitexact equality. First-sample full and noun conditions each bridge all seven heads and all six candidate class readouts against separate original scalar mask-sum calculations; equality is asserted. This does not open labels.

`trace.json` records `model_only_seconds` measured with CUDA synchronization around `model(inp)`, and a broader forward-plus-fixed-readout duration. After each complete two-condition sample, its noun trace row additionally records `sample_end_to_end_seconds_including_sampling_saving`, covering sampling, both model/readout calls and NPZ writes. Sum that field once per sample. The total seal duration also includes startup, checkpoint loading, repeated-forward control and output hashing. Memory is peak allocated CUDA memory per condition. These scopes must not be conflated.

## Current CPU audit status

`CPU_READOUT_AUDIT.json` passes: all 9,706×2 input lengths fit; 103 smoke candidate-class spans match an independent offset-based reconstruction of the original leading-space map; all 96 smoke actual-input first-noun masks match the existing text-only implementation. No target class/bbox was indexed by this audit and no model/GPU ran.

Static review of the parent scorer found its 7D GT transform/rounding and official eligibility treatment consistent with the loader. The legacy seven-head mean needs an explicit platform-level aggregate in addition to row values. Unknown metadata classes should fail clearly or remain all-raw text-only with an explicit unsupported-class flag, rather than silently disappearing. No new GT was opened for this review.
