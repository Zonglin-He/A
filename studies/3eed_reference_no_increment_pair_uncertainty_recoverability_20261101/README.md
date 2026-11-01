# R22 — reference veto removal, pair uncertainty and recoverability

Waymo-only source100→Drone/Quad; source weights and256boxes frozen. Full-phrase vs category increment is removed from new G decisions. All208 observations are historically exposed development; all154 original pairs actually processed. Reference proposal acceptance increases223→275, including7additional person proposals. New pair uncertainty avoids two old Quad harms but loses useful corrections; matched relation increment0 both domains.1194new frozenVLMforwards,0format rejects,0TTA steps, no adoption.

Three evaluations retained: cached veto ablation208×7, real-pair/geometry208×13, and21×3 label-enriched full-scene diagnostic.4223scalar rows reuse208inputs. The21-case diagnostic was selected using known labels to include18recoverable-with-relation cases and3regressions; not source-free test performance or independent confirmation. Both selection and limitations are explicit.

Read REPORT.md and protocol before results. Refer to prior R21 study at commit2e6b017ebf6e0cbbd44284f6f02e3cdce18df3f4 for dependencies. Reproduction requires licensed official data, private manifests and frozen source/model assets. Raw captions/RGB/point clouds/calibration/boxes/opaque IDs/model features/weights are excluded; this scalar export does not pretend to be a self-contained data release.

Execution: recoverability22.py (offline only); cache22.py then score22.py; pair22.py, assemble22.py then score22.py; scene22.py, assemble_scene22.py and score_scene22.py only under label-enriched diagnostic specification. All runtime readouts exclude task GT and modifier-increment decisions. No method novelty, relation contribution, or TTA effect claimed.
