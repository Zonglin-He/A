# Numerical repair and inference boundary

Ceres Solver's official nonlinear least-squares documentation describes gain-ratio trust-region acceptance: compare actual objective reduction to the linearized prediction, then enlarge or contract the allowed region. https://ceres-solver.readthedocs.io/latest/nnls_solving.html

R95 implements its own damped Gauss–Newton system for the existing normalized matching projection. It borrows standard Levenberg–Marquardt acceptance, not the Ceres implementation, and does not claim this numerical method as novelty. The task hypothesis is that R94's 5%-norm clipping prevented its already-fixed self-supervision from being executed. Removing that clipping does not certify the target's identity correctness or grounding utility.

The primary compares the repaired parameter update to equally informed direct distributions, including four-observation direct aggregation. The third synthetic occlusion is not used in optimization, stopping, or gain ratios; it is another sampling/masking of the same physical observation, not a new camera viewpoint or an independent scene.
