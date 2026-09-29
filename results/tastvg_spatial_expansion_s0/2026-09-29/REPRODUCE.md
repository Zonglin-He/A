# Reproduction entrypoints

The public repository provides implementation and scalar results. Licensed videos, existing TA weights/caches, private16-source roster and GT are not redistributed; run the model commands in the established local project environment with those assets. Public scalar audit needs only Python and NumPy.

```bash
python scripts/audit_tastvg_spatial_expansion_public_v1.py results/tastvg_spatial_expansion_s0/2026-09-29
python -m unittest tests.test_tastvg_spatial_expansion_s0_v1
```

Expert setup uses a separate target directory; do not downgrade the TA environment:

```bash
.conda/tubedetr/bin/pip install --target .runtime/sa2va_deps --no-deps transformers==4.44.2 tokenizers==0.19.1 peft==0.12.0 accelerate==0.34.2
```

Base runtime is Python3.11 / torch2.7.0+cu128, with torchvision, timm, einops, huggingface_hub and normal TA dependencies already installed. Run `.conda/tubedetr/bin/python scripts/download_sa2va_s0_weights.py` from the project root. It retrieves and Git-blob-verifies official configuration/tokenizer/code files at pinned revision3fee777d49ee9276eac51ea3e5f9b69e81d09be6, then downloads the four official weight shards by bounded HTTP ranges and verifies published LFS SHA256. Review this pinned remote model code before loading. The script writes OFFICIAL_CODE_RECEIPT.json and DOWNLOAD_RECEIPT.json required by the worker. Complete model provenance and code hashes are in PROVENANCE.json. The model is loaded offline after setup; no additional SAM2 weights or custom hole-filling CUDA extension are used.

Stages, in order:

```bash
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_expansion_s0_v1.py prepare
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_expansion_s0_v1.py capture
PYTHONPATH="$PWD/.runtime/sa2va_deps" bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_expansion_s0_v1.py expert
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_expansion_s0_v1.py ta
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/audit_tastvg_spatial_reinsertion_s0_v1.py
.conda/tubedetr/bin/python -B scripts/score_tastvg_spatial_expansion_s0_v1.py
.conda/tubedetr/bin/python -B scripts/report_tastvg_spatial_expansion_s0_v1.py
.conda/tubedetr/bin/python -B scripts/summarize_tastvg_spatial_s0_components_v1.py
```

The wrapper only selects locally installed matching NVIDIA userspace libraries for the current process. The protocol is fixed before inference. Workers resume receipted cells, verify H/expert/native hashes and seal before GT scoring. Report stages are intentionally write-once: do not run over existing outputs. Temporal reranking is an already established research component; S0 itself does not rerun it or deploy any selector.
