# First fixed persistent spatial OPD reproduction

Requires the existing TA-STVG environment/checkpoint, S0 exact H/expert caches, native parameter-support cache, and completed S0.6 reward barrier. Same16 historical GT entries are only used after prediction seal. The model environment and raw private evidence are not distributed with public scalar files.

```bash
.conda/tubedetr/bin/python -B -m unittest discover -s tests -p 'test_tastvg_spatial_online_opd*' -v
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_online_opd_s1_v1.py prepare
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_online_opd_s1_v1.py run
.conda/tubedetr/bin/python -B scripts/score_tastvg_spatial_online_opd_s1_v1.py
.conda/tubedetr/bin/python -B scripts/audit_tastvg_spatial_s06_s1_public_v1.py online artifacts/tastvg_spatial_online_opd_s1_v1
.conda/tubedetr/bin/python -B scripts/report_tastvg_spatial_s06_s1_v1.py
```

Use fresh write-once output directories for a rerun. The report script expects both S0.6 and S1 public-audit files; shell redirection can save audit JSON before reporting. It does not launch a search or change CURRENT.

Public scalar reconstruction (Python+NumPy):

```bash
python scripts/audit_tastvg_spatial_s06_s1_public_v1.py online results/tastvg_spatial_online_opd_s1/2026-09-29
```

This checks saved task/CI arithmetic. Raw state-chain, current-policy generation, full native reinsertion and SGD checks require the private run artifacts and are recorded separately in AUDIT/CURRENT_POLICY_AUDIT/REINSERTION_AUDIT.
