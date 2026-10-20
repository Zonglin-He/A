# Offline score serialization recovery

HC1902 predictions were completely sealed before labels were read. The original scorer completed its numerical calculations but failed at the final JSON write because seventy count values had NumPy int64 types. All original failure logs and the failed controller state remain private and preserved.

An independent CPU adapter invoked the unchanged scorer and replaced only that module’s JSON interface, converting NumPy scalar values to equal native JSON scalars. No metric, label, prediction, inference source, frozen plan or runtime fingerprint changed. All 1902 original prediction receipts were verified before and after recovery, 9510 official-versus-dense metric checks passed, and JSON roundtrip values exactly equalled the original scorer output. Sixty existing CPU engineering tests passed. The verified score was adopted into controller state with the original error and recovery receipt retained; the unchanged controller resumed with its original validation and started VidSTG.

`results/formal/scoring_recovery.json` records the exact adapter used for HC and all bindings. `scripts/score_json_adapter.py` reproduces the same serialization operation while requiring scratch data to stay under ignored `outputs/`. It refuses existing scores, changed frozen plans, invalid seals or unsupported JSON objects. Its SHA differs from the original HC adapter only because the reproducible entry uses an explicit private work directory.

Run only after a fully sealed job has failed on this exact JSON scalar issue, from the project root:

```bash
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/score_json_adapter.py --dataset vidstg
```

This is offline CPU scoring, not model inference or parameter selection. Adopt a score only after its completed receipt, metric digest, frozen labels and original worker validation pass. Do not relax any resume fingerprint. VidSTG predictions continue under the unchanged plan; if the same final-write issue occurs there, use the same adapter after the job is sealed.
