# CPU execution

Run from the repository root with the existing `.conda/tubedetr/bin/python -B` environment. No GPU job is launched.

1. `-m unittest scripts.test_tastvg_pairwise_certification_v1`
2. `scripts/run_tastvg_pairwise_certification_v1.py prepare`
3. `scripts/run_tastvg_pairwise_certification_v1.py calibrate`
4. `scripts/run_tastvg_pairwise_certification_v1.py seal`
5. `scripts/run_tastvg_pairwise_certification_v1.py diagnose`
6. Independent audit, figures and report; public-export audit and remote byte/hash verification; archive check/snapshot/check.

The immutable configuration records the scientific protocol. STATUS is the live execution state; plans and source fits are not target results. The runner pins its imports, code, protocol and original scalar inputs. Failed runs must retain evidence and pin a repair revision; no silent overwrite. Target annotated metrics cannot be parsed in the calibration or decision phases. Public release omits private weights, hidden tensors, media, raw annotations and personal conversation records.
