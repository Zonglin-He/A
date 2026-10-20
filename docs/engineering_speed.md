# Exact engineering acceleration

The method and all dataset parameters are unchanged. Three implementation changes are applied in order:

1. Exact visual-cache hits skip inactive-model and visual-tensor CPU/CUDA staging. Cache misses retain the original memory protection. A mutation-aware probe shares the verified key with the immediately following call; a changed tensor version or argument cannot reuse the probe.
2. The 448 Strong branch captures the original norm-before multimodal prefix and runs only its existing FP32 temporal replay. Both original offsets, first-pass action gate, fallback rules, second-pass temporal query, all-six-layer head shapes and reduction order remain. Unused spatial decoders and the discarded native FP16 decoder suffix are removed. An AST check rejects an unaudited QueryDecoder version. No new head, weights or training is introduced.
3. The two physical-grid interval mapping indices are memoized in one bounded cache (at most 256 MiB). Posterior weight multiplication, flatten order and bincount reduction are unchanged. MBR and its complete support are unchanged.

Seventy CPU tests passed. Four previously generated queries across both directions were replayed in twelve finite checks, using exact cached teacher evidence and no expert forward or GT. Original sealed Source/E_Ref/Live_raw/Live, physical intervals, complete Native/Strong posteriors, both gate rounds, complete six-layer temporal hidden tensors, RNG states and frozen weights all matched bitwise. See [the regression receipt](../results/engineering_speed.json). These finite regression timings do not establish representative full-dataset latency.

There was one maintenance window at a completed-query boundary. The original source and run were preserved; all 1902 HC and 296 Vid predictions were adopted through an explicit compatibility receipt retaining original hashes and measured execution costs. The archived source verified the original runtime proofs. New outputs have a new complete code fingerprint and strict resume hash. No fingerprint exception or ignore flag was added. HC offline score bytes are unchanged. The old pre-speed queue must not be resumed.

The physical interval construction still uses FP64 on CPU; model inference and FP32 replay precision are unchanged. Increasing cache size, reducing frames, dropping a Strong component or reducing adaptation steps are not part of this acceleration.
