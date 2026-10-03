# CPU execution

From the repository root with the existing Python environment:

1. `.conda/tubedetr/bin/python -B -m unittest scripts.test_tastvg_anchor_quality_v1`
2. `scripts/run_tastvg_anchor_quality_v1.py prepare`
3. `scripts/run_tastvg_anchor_quality_v1.py cases`
4. `scripts/run_tastvg_anchor_quality_v1.py fit`
5. `scripts/run_tastvg_anchor_quality_v1.py seal`
6. `scripts/run_tastvg_anchor_quality_v1.py evaluate`
7. Independent audit, figures and report, public audit/remote byte verification, archive check/snapshot/check.

Prefix script invocations with `.conda/tubedetr/bin/python -B`. Every stage is separate and finite; do not repeat immutable completed stages. STATUS distinguishes preparation, fitted models, sealed decisions and actual evaluated completion. Runtime pins freeze the protocol, source labels, original scores/state hashes and code. Baseline GT exposure is explicit; fit/seal prohibit target labelled and diagnostic reads. Preserve failure originals and pin a repair revision instead of overwriting. Existing GPU queues and completed experiments remain unchanged. The public audit uses anonymous scalar caches, not weights, raw annotations or media.
