# Decomposition oracle: what is being tested

This is a source-GT diagnostic of correction geometry. It does not train a
reader or constitute an unlabeled adaptation method. The original sixteen
source-training parents are all retained. No outcome-selected replacement is
permitted.

```text
same observed pixels -> frozen PTD merger F [T,H,W,2560]
                                     |
              fixed caption features q + frozen B1 stem/readers
                          /                         \
             F + R_T(F,q)                         F + R_S(F,q)
                 |                                     |
        native endpoint CE                native full-vocabulary coordinate CE
                 |                         on baseline student reference/time
                g_T                                    g_S
                 \________________ same F ______________/
```

The derivative includes the identity residual and the frozen reader Jacobian;
caption features remain fixed. The source GT labels supervise the native
endpoint/coordinate actions, not a GT-generated text prefix. This is distinct
from the historical post-adapter merger-delta diagnostic.

One analytic correction is constructed. Temporal and spatial directions are
projected into their frozen output-projection column spaces. Their radii are
the previously locked, source-exposed `.087687 * ||F||` and `.170316 * ||F||`.
The joint direction uses the union of these spans and equally normalized
initial branch gradients. There is no line search, optimizer, or best step.

| Native condition | Event pass F | Spatial pass F |
|---|---|---|
| Base | F | F |
| T-only | F + delta_T | F |
| S-only | F | F + delta_S |
| Decomposed | F + delta_T | F + delta_S |
| Joint | F + delta_J | F + delta_J |
| Joint-pass-matched | F + delta_J / sqrt(2) | F + delta_J / sqrt(2) |

Joint satisfies `||delta_J||² = ||delta_T||² + ||delta_S||²` as requested.
Because it is inserted twice, its total pass exposure is twice that of
Decomposed. The final row also matches actual two-pass squared exposure. Both
comparisons must be reported. The union span gives Joint greater channel
freedom; this test compares specified correction rules, not globally optimal
joint and decomposed adaptation.

All finite objective readbacks additionally hold the baseline student's
reference, interval, anchors, and cached probe schedule fixed. Final native
generation is free to change these conditions; its complete tube and failures
are evaluated separately. An S-only intervention preserving endpoint decisions
is structural, not evidence that temporal-spatial inference is independent.

Primary outcomes are parent macro vIoU for Decomposed versus both Joint
conditions, alongside Base/T-only/S-only tIoU, sIoU, cross harm, negative tails,
and native-good retention. Local gradient cosines and descent dots are
diagnostics, not substitutes for these finite outcomes. The source panel has
been used for development; unadjusted paired bootstrap intervals do not provide
an untouched generalization estimate.

Implementation: `vg_tta/desta3d_v3_decomposition.py` and
`scripts/desta3d_v3_decomposition_oracle.py`. Numerical and geometry audit:
`scripts/score_desta3d_v3_decomposition.py`; independent summary reductions:
`scripts/crosscheck_desta3d_v3_decomposition.py`. Exact locked scope and decisions:
`protocols/desta3d_v3_decomposition_oracle_v1.md`.

Any directional privileged policy, external specialist, or OPD experiment
requires its own subsequent protocol; it is not started by this oracle.
