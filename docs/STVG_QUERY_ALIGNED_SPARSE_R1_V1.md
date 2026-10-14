# Query-Aligned Sparse STVG-R1: finite development diagnostic (2026-10-14)

The actual geometry integration and fixed eight-case feedback screen are complete. This is not a qualified teacher-feedback method or a completed adaptation experiment. The current fixed feedback/input recipe stopped before conditional candidate-pool expansion and student adaptation.

## Evidence and implementation

- Frozen official STVG-R1-7B (c0db00d), unquantized BF16 / SDPA; cold cross-domain TA-STVG source captures; historical development exposure disclosed.
- P0-A independently integrates all 32 original spatial candidates, including 13 with out-of-bounds geometry. Original and mapped hashes are distinct. Explicit fixed legal mapping is shared by visual boxes, candidate metrics and the reward/anchor interface. Zero-area boxes fail closed. Gaussian probability stays in original z space. Forced CPU receipts are interface tests, not actual teacher validation or student fits.
- A reuses the original 48 cached P0 replies and original unmodified-geometry scores. B and C are matched to each other on the same mapped candidates and exact visual bytes; A is a historical reference, not a newly rerun identical visual protocol.
- Temporal evidence uses four global and actual boundary frames; spatial evidence uses separate full-scene and candidate-context views. Eight frames are delivered as four short two-frame clips with their correct physical gaps, plus timestamps/inside flags. This differs from native R1 input and is an unqualified capability assumption, not an inherited official performance claim.
- Predictions, raw outputs, source pins and output-only choices are sealed before each label readback. GT never constructs candidates or selects an online update. The eight cases are not a fresh unseen confirmation; a restricted source reviewer searched annotation JSON, was interrupted, and the exact affected split/count remain unknown. Root did not use that output for candidates or selection.

## Actual results and engineering history

Initial screen: 498 fresh calls (192 direct, 144 facts, 162 alignment). B has 192/192 legal short preferences, but ID agreement is 9/12, 0/12, 0/12 and 2/12 for HC2 temporal/spatial and VidSTG temporal/spatial. Base paired accuracy looks better than its control consistency and is not a reliable update signal. No stable improvement choice was admitted.

C has 118/144 complete facts records. All 162 text-alignment replies omit only the version marker; independent raw readback finds Tie or Uncertain in all three dimensions. This is not merely a parser-based negative conclusion. Facts often describe panel position (178/236 order values) or echo renderer prose (six relation values). Lexical variation is not validated localization quality.

A fixed display revision removes that prose, adds the exact legal crop box, and defines facts explicitly. It tests sixteen source-role pairs with 112 fresh calls. A source audit finds a correct-optional-schema normalization path and a non-JSON placeholder in the facts example; these are implementation errors, not scientific NO-GO evidence. All original code and results remain preserved.

The schema completion corrects those two specific errors, reads all 64 existing B categorical replies losslessly (including exact bare labels), and reuses their raw/prompt/marked-pixel hashes. It makes only 48 additional facts calls; no B inference is repeated. No complete facts record passes, so no new alignment calls are made. Among forty complete JSON packets, some individual subject/action/relation/geometry strings remain valid; all event-order strings are numeric sequences, boundary descriptions do not differ, and only two geometry pairs differ. Partial facts are retained separately and never fabricated into rewards.

Total: **658 fresh calls**, 64 exact B reuses and 48 historical A cached replies. Preprocess plus generation costs **755.686 s**, 1,006,769 input / 35,755 output tokens, peak allocated GPU **16.17 GiB**, with no observed OOM or resource-guard stop. These are verifier costs, not end-to-end TTA latency. There is no elapsed deadline; the call rosters and owned-resource limits are finite.

## Decision scope

- Stop only this fixed feedback/input recipe. C's protocol failure does not establish that explicit Query Alignment is worse than B or impossible.
- Current saved TA-STVG outputs contain one spatial trajectory and no verified distinct-instance hypotheses. Decoder layers or random shifts are not automatically other objects. The current pool supports local correction only; a candidate oracle is not an upper bound on parameter OPD.
- P0-C GPU pool expansion and P1 Direct, spatial OPD, temporal residual, Combined and matched old-DINO control are not triggered. **Zero new student fits, backward or optimizer steps**; their effects remain unmeasured.
- No new DINO, YOLO or SAM2, no LN carry / parameter sweep, no CURRENT promotion and no stopped old-queue restart.

All anonymous positive, negative, neutral and unknown case/pair/cost rows, source hashes, code revisions, CPU contracts, independent readbacks and the three-page actual-pixel-reviewed PDF accompany this record. Private queries, raw free-form facts, RGB, annotations, model states and caches stay local.
