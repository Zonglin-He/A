# Single ReferDINO spatial initialization and OPD

The formal design uses one expert: native ReferDINO with `ryt_mevis_swinb.pth`.
Its same eight selected-track boxes initialize E_Ref and supervise current-state
candidate distributions. TA-STVG and ReferDINO weights are frozen. Eight content
states are reset per query. Strong time is computed only from TA-STVG224/448 and
the six temporal layers, followed by the original physical MBR.

There is no independent GroundingDINO detector, four-frame detector stage,
admission/`what` skip rule, or old detector cache requirement. ReferDINO retains
its native internal architecture; its complete checkpoint is loaded without the
redundant external GroundingDINO-B checkpoint preload. All learned state is
covered and the runnable teacher denies access to detector checkpoints.

`config.json` contains both transfer directions. HC uses40 steps; Vid uses30,
selected from the completed old-initialization development grid. Other optimizer
settings remain lr0.01/tau0.2/scales(0.05,0.25)/M32/seed41. The interrupted
learning-rate/temperature search produced no winner. No new tuning is implied.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_single_referdino_opd_v1.cli \
  --request /absolute/path/request.json --output /absolute/path/prediction.npz
```

Request JSON contains `dataset` (`hc2` or `vidstg`), optional deterministic
sampling `key`, and `metadata`: original caption/index/video path, physical
frame_ids and geometry as used by the existing Source decoder. Metadata has no
GT boxes or temporal labels. The CLI decodes RGB, executes Source224/448 and one
ReferDINO8 query, computes E_Ref and runs actual OPD. It saves deterministic
student trajectories and separate cost metadata, without scoring or GT access.

E_Ref deliberately retains the old robust four-dimensional correction geometry
and inherited Source-domain scale/truncation. It uses eight singleton native
ReferDINO boxes; relative within-frame score costs are zero. This is not a newly
calibrated reliability probability, and a native selected track can still be
wrong. New one-expert results must be evaluated separately from historical
GroundingDINO+ReferDINO results. The package's existence is not a claim of
confirmed task gains or an automatic CURRENT deployment promotion.
