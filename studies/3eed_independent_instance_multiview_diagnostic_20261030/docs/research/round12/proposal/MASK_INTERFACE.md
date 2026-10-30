# R12 independent full-image instance-mask interface

Proposal/specification only. Root selects and acquires the single frozen frontend and controls its first inference release. No model is constructed or executed by this document's author.

## Strict frontend boundary

Input: `frame_id` (opaque input identifier) and literal `rgb_uint8[H,W,3]`. No query, source box, source point crop, caption, target class, positive_map, prompts or target labels are accepted. Infer once over the **entire RGB image** and retain all model classes above the registered score threshold. The original3EED q0 is only a later association hypothesis, never a segmentation initializer. This differs materially from two source-seeded GrabCut masks agreeing with each other.

Root's preferred frontend is official torchvision0.22 `maskrcnn_resnet50_fpn_v2`, `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`, frozen/eval/no_grad; native short side800 and long side cap1333. Initial confidence retention is score>=0.5 and foreground is postprocessed mask probability>0.5. These are predeclared operating points, not calibrated task-correctness probabilities. Exact weights/release/SHA and library code/runtime binding remain root's acquisition receipt. This is an additional frozen2D perception model supplying independent proposals; it is not a trained target-domain model or an EMA/3D grounding teacher. Its extra parameters/cost/information must be disclosed, rather than calling the entire system one network.

## Per-frame returned data

- `boxes_xyxy`: float32[N,4], original full-resolution image pixels, with native model order retained.
- `class_ids`: int64[N], exact native COCO category indices; no contiguous reindexing assumptions.
- `scores`: float32[N].
- `masks_bool`: bool[N,H,W] after official mask paste/postprocess returns to original H/W. An explicitly versioned lossless RLE/packbits serialization is acceptable. Never resize twice or apply resized-image K to original-coordinate masks.
- `instance_indices`: int64[N], original post-head output indices before the registered score filter.
- `original_hw`, `actual_transformed_hw`, `mask_threshold`, `score_threshold`, `detections_per_image_cap`, raw detection count and retained count, image-byte and decoded-RGB hashes, model/checkpoint/category-table/config hashes, timing and memory receipts.
- `status`: valid, empty_predictions, invalid_image or explicit engineering failure. An empty prediction remains in the denominator, not a reason to fetch another frame.

Native detection cap must be recorded (the usual default is100 across all classes); allCOCO means all returned classes, not all objects guaranteed found. Internal model resize is aspect-preserving and original coordinates are restored by torchvision's transform. Calibration K/E must correspond to that original RGB frame. Bound category metadata including background/unused slots must be retained; the number of indices cannot be guessed from80 semantic categories.

No probability maps need to be persisted: they can be large. Retain exact binary masks/RLE, per-mask hashes, declared thresholds, scores and reproducible code. Mask hashes certify bytes only, not segmentation or identity correctness.

## Later association, not frontend filtering

A fixed **input-text-only** mapping from target noun to supported COCO class may restrict associations later. No target GT category fallback is allowed. Unavailable/ambiguous noun mapping, missing class, occlusion, no trustworthy point support, multiple sameclass masks compatible with q0 or inconsistent cross-frame track yield explicit NA. Mask class and temporal stability do not establish literal referring-expression identity; relations/actions/appearance in a full caption may remain untested. Q0 may be consistently wrong even when an object is tracked correctly.

Sources checked: [torchvision0.22 model/weights documentation](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html), [official transform/postprocess source](https://raw.githubusercontent.com/pytorch/vision/v0.22.0/torchvision/models/detection/transform.py), [official detector constructor/source](https://raw.githubusercontent.com/pytorch/vision/v0.22.0/torchvision/models/detection/faster_rcnn.py). No local inference has verified these frontend outputs yet.
