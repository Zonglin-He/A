# Policy implementation review

The policy is in corrected student output coordinates, not raw native sigmoid logits. Centers are perturbed in target width/height units; log widths/heights are additive Gaussian components. A single global mixture component applies jointly to four coordinates for each candidate.16 candidates per component,8 independent normal vectors and their negatives; fixed .5 mixture weights. No unreported per-coordinate mixing or clipping to image bounds.

Actual density is logsumexp of two properly normalized4D diagonal Gaussian components, with detached diagonal. NLL optimizes only the current mean; the old Gaussian analytic KL/control-variate is absent. Teacher weights are detached softmax of the same geometric pseudo-box cost over32 candidates. Current state is re-decoded after every update.

The new fresh-Fixed comparison retains its initial whole size-dependent proposal (mean and D0). Live refreshes both mean and Dk. Each has40 independent batches from an identical per-parent/seed/round stream; no more batches for Live. Three scale configurations share noise; all three seeds retained. First-step boxes/weights/state are asserted exact. All original state resets, physical offset scatter, E-retaining mapping and frozen-network checks remain.

CPU synthetic tests compare mixture density with torch distributions and gradients with finite differences. First actual parent independently reconstructs720 NLL values across all18 arms/configurations/seeds, without GT. This checks implementation, not teacher correctness or downstream benefit. Independent complete saved-trace/metric audit follows global prediction seal.
