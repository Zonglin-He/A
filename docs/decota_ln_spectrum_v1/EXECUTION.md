# Finite CPU execution

1. Read current ledger, production registries, predecessor FINAL_COMPLETION and
   runtime/input receipts. No preceding queue is resumed.
2. Run `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
   .conda/tubedetr/bin/python -B scripts/run_decota_ln_spectrum_v1.py prepare`.
3. Run the same entry with `extract`; it validates the full saved state chain,
   writes private vector matrices and anonymous Gram/geometry inputs, and seals
   label-blind selections. It never opens GT-derived metric rows.
4. Run `analyze`: geometry first, then join the previously sealed anonymous
   metrics. No model or annotation access, no new GT scoring.
5. Independent root audit reconstructs selected vectors and Gram/projection
   arithmetic. Public audit reconstructs spectra, prefix projections, pair
   cosine and source-node utility statistics from anonymous scalar inputs.
6. Generate/visually inspect figures, update research archive check/snapshot/check,
   publish new code/protocol/config/complete results and verify every remote file.

STATUS and receipt counts distinguish extraction, analysis and verified
publication. Temporary analysis files are private. Parameter vectors, saved
states, candidate boxes, media, annotations and checkpoints are not public.
The public Gram matrices contain derived correction geometry, not model weights.

