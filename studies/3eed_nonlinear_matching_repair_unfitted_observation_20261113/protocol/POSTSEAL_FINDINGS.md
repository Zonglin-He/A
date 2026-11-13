# Postseal interpretation

R95 fixes an execution limitation but does not repair the entire grounding pipeline. All following interpretations are descriptive on historically exposed development data.

- Same factual-view target: R94 one-shot GN executed 11/60 requested Top1 changes; R95 executes 39/60, and agrees with the target on 305/336 inputs. Mean remaining centered-logit squared error is 24.4175%. All ten iterations were used, so complete convergence is not established.
- The same dense correction changes factual-view mIoU by +0.128552 pp but changes the unfitted C-view result by -0.479186 pp. This intervention shows that reusing the correction is not safe across masking conditions. It does not by itself identify the precise image-instance identity error.
- Parameter transfer on C is -0.350605 pp. The main FC update is -0.262340 pp versus its own zero and -0.006161 pp versus same-information direct use. It fails the locked task criterion.
- The source-mixture secondary yields 23.148086% versus 23.068600% for its direct counterpart. This is a disclosed secondary positive comparison, not a replacement for the failed primary; it also stays below factual-only R94 direct 23.160598%.
- Same-input fitting is not enough: wrong target choices, candidate-coupling through the shared head, residual fitting error, and observation-dependent corrections all remain possible. None is resolved by proving that parameters changed.
- Stop this fixed R95 recipe. Retain the R91 model and the R94 direct evidence as development references. No robust new TTA adoption, source retraining, architecture replacement, or FINAL evaluation is claimed. A further iteration would need to establish observation-conditioned, task-useful evidence rather than merely push the same target harder.
