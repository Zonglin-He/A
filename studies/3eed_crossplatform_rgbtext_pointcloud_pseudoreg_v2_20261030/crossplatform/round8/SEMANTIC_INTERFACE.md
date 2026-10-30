# R8 frozen crop/text semantic encoder

`semantic_encoder.py` provides a frozen OpenAI HF CLIP ViT-B/32 image/text encoder.
It emits similarities, never 3D boxes, target labels, pseudo-instance identities,
EMA targets, or a trained target-domain model. Extra public pretraining information
is explicitly additional to the one Waymo-only 3EED grounder.

```python
from semantic_encoder import FrozenSemanticEncoder, crop_rgb
encoder = FrozenSemanticEncoder(device='cpu', batch_size=32)
# CUDA requires a separate parent GPU lease; no automatic CUDA selection.
crops, crop_receipt = crop_rgb(original_RGB, explicit_pixel_xyxy)
result = encoder.match(crops, [literal_full_caption],
                       crop_roles=['tight', 'context'])
# result['cosines']: float32 NumPy array [Ncrop, Ntext]
# result['text_receipt']: context77 token lengths and actual truncation counts
```

For G's ≤16 physical groups, the caller supplies at most32 explicit tight/context
crops in a declared order. The encoder does **not** merge views: the registered G
owns `.5*tight+.5*context`. G also owns 20%-plus-min4pixel context expansion.
`crop_rgb` only floors/ceils bounds and clips to the original image; it rejects
empty crops and never replaces them with a full image.

`encode_crops(crops)` returns CPUfloat32 normalized `[N,512]` tensors;
`encode_texts(texts)` returns the same text features plus truncationreceipt.
All frozen feature vectors are L2-normalized before the unscaled cosine dot
product. No softmax, learned logit scale, prompt templates, noun-only substitutions,
confidence calibration or color-label gates are applied. Text remains literal full
description; right truncation uses pinned official tokenizer end-token behavior.
Token counts include the start/end special tokens (77 totalcontext, 75 content).
Long descriptions may lose the relation clause, and this loss is recorded.

Original uint8 RGB/PIL crops use the pinned HF CLIPImageProcessor: convertRGB,
bicubic shortedge224 resize, center224 crop, rescale1/255 and official mean/std.
A long thin ROI can lose endpoints during this canonical center crop; this is a
real supporting-condition limitation, not an aspect-stretch workaround. Tight and
context values remain separately inspectable. Similarity does not establish
visibility, instance identity, referential correctness, calibrated probability, or
3D localization ability; same-class objects and broad context remain ambiguous.

All runtime assets are local/hash-verified at fixed HF revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. That official repository has only
`pytorch_model.bin` at this revision. The local safetensors derivative uses
`torch.load(weights_only=True, map_location='cpu')`, verifies official LFS SHA,
strictly loads all learned CLIP keys/shapes and verifies a bitexact tensor roundtrip.
Legacy deterministic position-id buffers, if present, are removed only after exact
arange verification because current Transformers registers them nonpersistently;
this is explicitly recorded. No unrestricted pickle or remote executable code is
used. Weights and private media are not public export artifacts.

Primary provenance:

- https://huggingface.co/openai/clip-vit-base-patch32
- https://huggingface.co/openai/clip-vit-base-patch32/blob/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/preprocessor_config.json
- https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6

Original CLIP code is MIT licensed; locally downloaded originalcode references and
license are retained under private assets for provenance. No foreign repository
text is treated as agent instructions.
