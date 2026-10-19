# STVG: frozen ReferDINO-supervised spatial OPD research version

Current research entry: **`stvg_referdino_opd_v1`**, HC primary. [Authoritative research manifest](methods/RESEARCH_METHOD.json), [single frozen configuration](methods/stvg_referdino_opd_v1/config.json), [method and execution guide](methods/stvg_referdino_opd_v1/README.md), [final results and full costs](results/stvg_error_state/2026-10-19/referdino_final/REPORT.md).

Video + text → frozen TA-STVG Source → four GroundingDINO-T observations and E initialization with Source finalMBR → native ReferDINO eight-frame evidence → dual-scale output-box OPD → full student trajectory. Both expert stages are real dependencies and are charged. E retains the original budget of at most four planned-position calls, including the existing skip/rejection behavior; actual counts are reported and all queries remain in evaluation. All pretrained weights stay frozen. Each query independently resets **8×256=2048 temporary content states** and Adam; no online LayerNorm carry.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_referdino_opd_v1.run \
  --input request.json --output prediction.pt --seed 41
```

Input metadata specifies video path, caption, original width/height and physical frame IDs. The [guide](methods/stvg_referdino_opd_v1/README.md) documents input validation, optional cache acceleration and separate sealed-prediction scoring. Runtime weights, local model packages and exact dependency hashes are listed in the [provenance record](results/stvg_error_state/2026-10-19/referdino_final/PROVENANCE_PUBLIC.json). Private weights, media, labels, captions and raw caches are excluded from this repository.

The final round compares only temperatures .05/.10/.20 on exposed HC128 with seeds41/42/43; the selected recipe is evaluated on32 prelocked, disjoint parent videos. Earlier Source/old-method exposure remains disclosed. Configuration freeze is not a claim of a stable2–3pp OPD gain. See the report for signed results, confidence intervals, tails, Direct/fresh Fixed/Live comparisons and total spatial versus incremental OPD effects. Three seeds are experimental repeats; default inference is one fixed seed, not an ensemble.

The frozen RF32 comparison did not reproduce the development gain: Live−E full vIoU **−0.8672pp** (95%CI[−3.2597,+0.7790]); Live−Direct **−0.2719pp**, Live−freshFixed **−0.4222pp**, with intervals crossing zero. The configuration stays frozen; this is not a verified replacement for E or a stable2–3pp OPD gain. Severe cases remain in the published mean.

## Version identities

- `methods/RESEARCH_METHOD.json` is the unique current research definition; MAIN/PAPER/WORKING registries reference it.
- [CURRENT_METHOD.json](methods/CURRENT_METHOD.json) preserves the historical deployed P7-A algorithm and is explicitly marked historical deployment. It is not the current paper recipe.
- [Historical P7-A](methods/stvg_opd_p7a_v1/README.md) uses1792 query/LN parameters and online fractional LN carry; the new method does not.
- Historical T2+B, P7-A, protocols, saved predictions and negative results retain their original paths. Old paused queues are not resumed.
