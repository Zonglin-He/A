# Post-seal event compatibility and conditioning audit

This separate diagnostic ran AFTER all64 predictions were sealed and independently
scored. It changes no inference, threshold, acceptance or frozen runtime code.
[Complete anonymous phase audit](phase_evidence_audit.json); reproducible code is
`research/native_event_all_phase_v1/phase_audit.py`. Its private receipt binds the
64-query prediction seal, scored report, exact labels and audit-code SHA256.

30 real middle texts in21queries: mean posterior probability of any temporal overlap
with the FULL-description labeled event is0.909503;3phases place most mass outside.
Mean Source-box overlap versus Full is0.773075;2phases below0.5. These are compatibility
and conditioning proxies, not atomic event annotations or identity truth. High any-
overlap can come from broad predictions, and a true subevent need not occupy the
entire full event; never use this diagnostic as online event supervision.

RC056 loses35.7441pp time against endpoint-only. One middle Source-box proxy=.03567,
another=.46047, while any-overlap masses remain high. Real valid-order probability
is1.0265e-5 versusFull-as-middle3.7836e-4: both conditioning drift and incompatible
marginal event assignments are plausible. It does not prove a correct online gate.
RC041 loses23.5496pp even though both middle Source-box proxies are.8931/.8798 and
both any-overlap masses>.997. Stable boxes and some event overlap do not guarantee
reliable ordered endpoints. RC048 has a middle with only.2074 event-overlap mass and
real order mass4.70e-6. There are multiple failure modes.

The all-phase assignment assumes AND follows monotone starts AND ends, while real
concurrent/nested actions need not obey both. Independent positive text predictions
can also select separate occurrences. Do not repair this by rejection using order
mass, or declare extra event information intrinsically harmful. Stop this bounded
order-composition version; next feedback must change how occurrences are conditioned
or provide genuinely different event evidence, and still beat fixed fair controls.
