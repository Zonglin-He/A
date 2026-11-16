# Read-only audit of previously tested normalization baselines

Actual local code inspected: crossplatform/round18/algorithms.py, supplement_algorithms.py and crossplatform/round46/external46.py. No baseline rerun or protected-input access in this audit.

- Old TENT was not LayerNorm-only: configure_norm selects all BatchNorm1d/2d outside the frozen text encoder, uses current batch statistics (track_running_stats=False), and updates affine parameters. Its first-pre readout explicitly isolates statistics from gradient updating.
- SAR is the LayerNorm/GroupNorm-only variant in supplement_algorithms.py. It must not be conflated with the TENT scope.
- Old MEMO replaced all BN forwards with source/target moment mixing using source prior16/17 plus current-input1/17. Its BN-prior-only readout exists separately from augmentation and parameter updating. A generic claim that backbone BN statistics were never tested is false.
- These older ports use subject-token probability mass as the matching readout; current native log-odds and source-prepared architecture have changed. This is a protocol difference, not evidence that a normalization rerun will improve grounding. It would require a distinct locked comparison, not an unbounded BN/temperature grid.
- R110 changed the two CE-head BN buffers only; it was not a PointNet backbone normalization experiment. Its negative result also cannot be generalized to every backbone normalization procedure.

The audit rules out an unsupported next-step rationale; it does not constitute a new TTA result. Do not rerun old TENT/MEMO or label a repeated statistics-only experiment unexplored merely because it has another name.
