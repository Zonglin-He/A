# Historical cached readout experiments

`core.py`/`run.py` test detrended whole-interval state separation. It failed the development gates.

`action_*` and `rank_*` are **retired historical negative experiments** using an architecture-specific internal auxiliary head. The user explicitly excludes this signal from the requested generic algorithm. Do not use their predictions as method inputs, gates or new-trial feedback. Keep them only to reproduce the historical signed results; they were completed before the user correction.

The requested model-agnostic interface is implemented separately in `scripts/generic_tube_readout`. It does not depend on these files or outputs. It also failed measured adoption criteria; no new production method was registered.
