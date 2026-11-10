# R60: internal-state detected channel patches

Claim and constraints: frozen Waymo-only epoch100, exposed DEV336 and original A256 final boxes. Existing R57 A/B/C correspondence and UNKNOWN remain fixed. Test-time execution must be deterministic detector -> actual column assignment -> native replay -> source restoration, with no autograd, least-squares, pseudo-target fitting, external model or inputwise patch search. FINAL384, source retraining and full target evaluation are outside this run.

Prior evidence: R59 direct internal representation fusion 22.611116%, ridge 22.337610%, frozen ABC 22.302460%. R59 exact fitting did not add value; that does not test a nuisance-component detector. Its source prefix activations can be reused because all editing was strictly terminal and source restoration/audits passed.

Minimal unresolved question: can an internal, label-free diagnostic select a small set of columns whose attenuation improves grounding beyond three-sampling average and equal-cost fixed/shuffled/random controls?

Two candidates only, zero revisions initially. Both share alpha=.25, maximum8 columns and no score search:

1. Noise: normalized additive contribution to an entity-versus-other-token direction. Align B/C to A using existing soft geometry. Attenuate at most8 columns where correspondence-weighted within-view variation exceeds between-candidate variation, weighted by excess. This is not raw activation-variance deletion. The contribution diagnostic is approximate; all final decisions replay the actual normalized logsumexp log-odds.
2. Common response: columns with large normalized output energy, candidate-common activation fraction>.5, and small squared entity-direction alignment. Candidate-common components can compress angular differences after normalization; they can also encode useful shared category semantics. Select at most8 by that fixed score. This is a distinct hypothesis, not a proven fault.

Deployment: source-reset per input; detector reads frozen activations/tokens/correspondence, changes last-linear columns, scores all candidates and restores. No labels, sample/scene special cases, external inference or iterative patch optimization. Research was designed by the Codex assistant; do not assert that a separately invoked Astra model ran. There are zero new external model calls during inference.

Controls: source, frozenABC, R56one, both R58online readouts and R59directAll inherited with exact lineage. Per mechanism: dynamic; shared fixed patch derived from mean unlabeled dynamic gates across all336 (target-transductive calibration, not source-only); within-platform donor permutation retaining the same patch multiset; random columns with identical count and each edit magnitude/total norm; activation gate using same attenuation. Fixed alpha/count/formula selected before task evaluation, no hyperparameter grid. Negative tail and all cases remain in denominators.

Causal boundary: a stability improvement with no task improvement stops the mechanism. A task improvement without superiority to fixed/shuffled/random does not establish input-state-specific editing. Equal gates/weight scaling are algebraically equivalent, not novel merely because source parameter storage is changed.

Status: detector pass completed336; both candidates trigger336 and median8 columns. All diagnostics and gates sealed without task scoring. Patch replay and independent source/formula/control audit follow before one saved-IoU evaluation. No scientific revision yet.
