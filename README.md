# BoxFixed + TargetSemantic K16

The selected TA-STVG method combines sparse-expert complete-box propagation with a
frozen one-round target-region semantic prototype readout. One ReferDINO, eight
observations, no network-weight training, no additional high-resolution branch.

1. Original frozen Source224, two offsets, original native process.
2. Same eight expert observations, initialization and complete-box propagation.
3. Six-layer Global6 interval prior; auxiliary full-time target trajectory.
4. Original Source word-semantic head on target context1.5; opposite-offset positive/
   negative prototypes with fixed common-mean shrinkage **kappa16**, exactly **one** update.
5. Original physical MBR and unchanged spatial score/DP.

This is the sole main profile; there is no temporal-parameter selection CLI.
The user explicitly selected it despite a small negative HC short-event recheck.
All previous failures remain reported. Full evaluation is newly launched under an
independent fingerprint, not renamed old results or an unseen-test claim.

See [method](docs/method.md), [protocol](results/k16_full_v1/protocol.json),
[compatibility](results/k16_full_v1/REGRESSION.json), [history](docs/history.md),
[evaluation](docs/evaluation.md) and [private assets](docs/assets.md).

## Run

Use the dependencies in environment.yml, requirements.txt and requirements-teacher.txt;
place frozen assets at documented private paths. No data or weights are bundled.

```bash
python tools/setup.py --prepare --tokenizer /path/to/existing/bert-tokenizer
python tools/setup.py --build
python tools/check.py
python -m stvg predict --request /path/to/request.json --output outputs/example.npz
python -m stvg evaluate --manifest /path/to/label-free-manifest.jsonl --output outputs/evaluation
python -m stvg score --manifest /path/to/label-free-manifest.jsonl --labels /path/to/offline-labels.jsonl --predictions outputs/evaluation --output outputs/metrics.json
```

Official frozen full controller: `python -B tools/full.py run`. It requires a prepared
content-bound plan and compatibility proof; never resumes an older fingerprint.
HC1902/227 parents then Vid10303/732 parents. Source/SharedInit/Global6 must match old
sealed official outputs; K16 Full/new-time spatial ablations are newly generated.
Each complete domain seals before independent CPU official-label scoring. All signed
outcomes and severe declines, parent-cluster intervals, time/boundary/space metrics
and real segmented cache costs are retained. Timing with cached evidence is not a
claim of fresh deployment speedup.

## Evidence and archive

[Original Global6 full](results/boxfixed_full_v1/REPORT.md),
[K16 development/confirmation](results/target_semantic128_shrink_v1/ANALYSIS.md),
[last failed temporal repair](results/target_semantic_final_v1/ANALYSIS.md).
These reports describe their own protocols and exposures, not the new full outcome.
The entire prior code/evidence tree, including all experiments and negative controls,
is preserved in [archive/research-pre-k16-20261023](https://github.com/Zonglin-He/A/tree/archive/research-pre-k16-20261023)
and Git ancestry. Main/research retain the single selected deployment entry and historical
results; old research entry points are archived. No raw videos, queries, GT coordinates,
weights, masks or features are published.
