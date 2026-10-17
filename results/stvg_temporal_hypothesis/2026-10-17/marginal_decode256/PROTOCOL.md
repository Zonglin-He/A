# Original256: change only final B2 decoding to merged-interval probability MAP

Human authorized after fixed-evidence B2 marginal-training result a3981e71.
Same original HC128/Vid128 exposed development parents and cross-domain TA-STVG
Source (target HC2 uses Vid source; target Vid uses HC2 source), frozen R1
revision c0db00d019bf746f54b67770b82261b2161fed59. No new Teacher, caption,
optimizer invocation, Source capture, pixel decoding, Spatial or Full run.
Reuse exact B2 I1/I2/rewards/prior/target/config/512 theta and accepted trajectory.

One factor changes: final physical interval is the global MAP after summing
probabilities over all two-offset legal state pairs merging to that interval:

    p_theta^o(s,e) = softmax_{s<e}(actual_FP32_logit_start(s)+logit_end(e))
    P_theta(I) = sum_{merge(z0,z1)=I} p_theta^0(z0) p_theta^1(z1)
    I_hat = argmax_I P_theta(I)

FP32 actual integrated endpoint outputs from B2 were independently ROOT-checked
bit-for-bit against the saved full-grid logits for all256. Bind original Source,
fit, full prediction SHA and actual-head ROOT before reusing those exact outputs.
Probability normalization/summation uses float64, identically before and after
adaptation. This is a new CPU decoding measurement from actual full outputs,
not a new GPU forward, fitted parameter set, or newly acquired Teacher evidence.
Retain original Source boxes. Global search over every legal physical interval,
not just I1/I2; deterministic sorted physical row-major ties, no clamp/top-k or
GT routing. The production implementation sums explicit independent outer
products; independently audit all bins with CDF/inclusion-exclusion and test
Python pair enumeration, overlapping endpoints, deterministic ties and no-clamp.

Same decoder applied to theta=0 is the already measured marginal no-update
control. Reproduce that control exactly for all256. All17 original no-information
zero-parameter queries must equal this marginal control, not a forced old Native
MAP fallback. No information still means zero adaptation; decoder-only changes
can affect final output and must not be attributed to learning.

Seven matched arms: Frozen, T2, B0 original, B1 I2-only ranking, B2 marginal
training/original Native decoding, Source marginal-MAP no-update, B3 B2 exact
parameters plus marginal-MAP decoding. Primary adaptation contrast B3-minus-
no-update; readout contrast B3-minus-B2. Other comparisons secondary. Seal every
new decoded prediction and independent probability/Source/no-update ROOT before
GT. All previously exposed GT only used for postseal evaluation. No update,
selection, reward, parameter, hyperparameter or decoding rule may use GT.

Report vIoU/tIoU/dense and common-Native-time sIoU/recalls, all source signed
positive/negative/neutral/unknown, paired source t95 by domain (not multiplicity
adjusted), old six-arm scalar metrics exact. Prior negative results remain.
No independent confirmation or backbone-generalization claim follows. CURRENT/
PAPER unchanged. Numerical R1 rewards do not establish referent/event semantics;
512 semantic panel remains partial44 model-assisted/468 pending.

One isolated finite CPU unit, 4h cap, MemoryHigh16GiB/Max24GiB/Swap1GiB;
host6GiB/disk30GiB floors and no new GPU claim. No historical queue touched.
Publish code/protocol/all anonymous results and material limitations, excluding
text/media/GT coordinates/tensors/weights/credentials; same fixed actual input
pixel review and chart inspection. Verify HTTP bytes/SHA/Gitblob/fetched Git
mode/tree/two refs fresh CAS, archive check/snapshot/check and close finite group.
No automatic follow-up or promotion.
