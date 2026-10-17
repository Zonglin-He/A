# Source-trained visual event critic: finite joint-tube pilot

Additional source-supervised preparation; target model/critic frozen outside the per-Query direct1792 adaptation. Two directions each use 36 source-train +12 source-validation parents and16 previously exposed target development parents. No independent confirmation or official full result is claimed. Five HC movie-parent overlaps were excluded before the fixed source selection.

The critic compares complete Source/direct Student trajectories crossed with three Native-only time hypotheses. It sees actual frozen appearance/motion ROI features, full-frame context, full Query tokens and physical positions, including frames outside the proposed interval. A visual input interface is not itself proof of object/event understanding.

Before critic training, a source label identity defect was corrected: seven old processed pointers mapped a repeated identical caption/video to a different event occurrence. Source labels now bind original video, Query, exact used_segment and target ID, and use original VidOR frame coordinates. GPU candidates were unchanged; only their waiting CPU controller was handed off before source labeling. Original receipts and deadline were preserved.

| Target | Frozen Native vIoU % | Direct + Native % | Direct + conditional time % | Joint % | Oracle % | Joint minus Direct pp |
|---|---:|---:|---:|---:|---:|---:|
|hc2|18.831315|18.446397|20.008012|20.955719|25.667591|+2.509322|
|vidstg|7.147781|11.604658|11.958873|7.876929|13.726246|-3.727729|

All signed per-parent metrics, strict-pair denominators, same-trajectory/time and same-time/trajectory factors, source-validation curves, intervals and unknown cases are retained in JSON. Oracle uses post-seal target GT only as a disclosed development diagnostic and is not a usable method. All six candidate outputs sealed before target diagnostic GT; critic training and source checkpoint selection never use target labels.

**Decision:** stop_finite_pilot. The locked gate did not qualify the new feedback in both directions. No OPD extension or online/full stream is automatically run; this fixed pilot does not rule out all visual event critics.

CURRENT remains the restored historical DINO P7-A; its online Gaussian/LN-carry implementation is not silently replaced by this episodic research candidate. Qwen official evaluation remains paused.

Mechanism diagnostic: for the HC-source critic transferred to Vid, same-interval spatial-state ordering drops from32/33 correct source-validation pairs to9/30 target pairs. For the Vid-source critic transferred to HC it is16/24 source-validation pairs and38/45 target pairs. These dependent pairs are descriptive, not independent samples. Direct-conditioned time selection has positive mean deltas in both targets, but paired16-parent95% intervals both cross zero. The evidence supports a transfer failure of this fixed spatial update evaluator, rather than absence of candidate headroom or proof that all conditional critics fail.

Integer membership correction was completed before the first target critic inference. Six of384 float-seconds masks differed by one frame; all384 integer-frame masks now match exactly. The two superseded source models were retained and both source critics were rerun with the same seeds/80epochs/configuration; no Source/DINO/Student GPU computation was repeated.

Corrected fresh-run entrypoint: `scripts/execute_stvg_joint_event_critic_v1.py`. It requires the original official-data/source-manifest/checkpoint adapters, freezes a fresh scope, applies strict source identity binding and integer memberships from the start, and refuses to overwrite an existing scope. The actual001/002/003 handoff history remains available as implementation evidence.

Private fixed8-anchor actual pixel inspection is an input/output engineering check, not independent semantic annotation. The old512 panel remains44 model-assisted/468 pending. Raw videos, annotations, GT coordinates, model weights, tensors and original Query strings are excluded from public files.

Related scope: [E3M](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11011.pdf) already alternates temporal relevance and spatial region estimation; [CAPD](https://arxiv.org/abs/2609.34581) studies masked temporal-group influence in MLLM token distillation. This pilot makes no novelty claim from coupling alone.
