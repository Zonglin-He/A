# Fixed P7-A OPD paper method

Entrypoint: `methods.stvg_opd_p7a_v1.predictor.P7AOPDPredictor`.

P7-A selects the final WHEN using the source-frozen actionness projection and
the original two-offset physical envelope. Native I0 remains the sparse DINO
observation support. The existing joint Query Residual and decoder-LN OPD
objective, target-specific fixed parameters, per-query QR/Adam reset and
fractional LN carry are retained exactly. No temporal head training is added.

The complete new paper suite is tracked in
`artifacts/stvg_opd_p7a_paper_revalidation_v1` with its fixed protocol. Existing
sealed spatial fits are reused only through exact source/input/state-history
manifests; missing source temporal outputs are actually captured and all final
readouts are sealed before offline labels. Reused computation is labelled as
reuse. True deployment cost requires fresh-expert measurement.

The research entrypoint does not modify `methods/CURRENT_METHOD.json`.
