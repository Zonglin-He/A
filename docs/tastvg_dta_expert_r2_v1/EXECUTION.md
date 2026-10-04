# R2 execution

CPU runtime: `.conda/tubedetr/bin/python -B scripts/run_tastvg_dta_expert_r2_v1.py`.
Private `artifacts/tastvg_dta_expert_r2_v1`; public anonymous results
`results/tastvg_dta_expert_r2/2026-10-04`.

Finite separate processes: prepare GT-free support and lock → E-Deploy under
GT-read guard → deploy barrier → E-Oracle with explicit GT center selection →
global prediction barrier → score both arms and direct-teacher/Gaussian-MAP
controls with fixed A boxes → independent gradient/selection/dense root audit →
public scalar/CI audit → figures/report → archive → GitHub exact verification.
The controller never reads labels or metrics and launches each stage once.

Head optimizer is imported unchanged from R1. R1 outputs/source choices are
immutable inputs. Current Native output must match R1 and original capture
indices exactly. Historical internal GT_KL field from that shared Gaussian
routine is exported as teacher_KL; Deploy uses no target GT.

Completion requires real 576 adaptations/1152 readouts, audit and remote
verification receipts. No R2b mixture, R3 persistence, additional expert, new
cohort, decoder/LN or old queue automatically starts. Preserve any engineering
failures and pin revisions without scientific retuning.

