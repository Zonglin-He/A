# Source native actuation v2: support/format distinction

The scientific protocol is [v1](desta3d_v3_free_actuation_v1.md): identical cases,
30 fixed steps, scopes, AdamW, loss class support and final success criteria.
The free001 failure after event30/spatial24 is preserved. Native24 kept its event
text, interval, reference and time logits exactly, and still performed the same
11-anchor box probe, but box grammar failed and parsed positions became empty.
The old assertion conflated parsed successful boxes with actual action support.

V2 saves the complete trace before validation and derives spatial supervised
positions from the native box-probe query IDs. It verifies those anchors and event
policy remain unchanged. It neither constrains/repairs generated tokens nor
changes evaluation; format failures remain zero-scored. The coordinate-only CE
still does not supervise grammar positions. This is a substantive diagnostic risk,
not something repaired by silently adding a structural loss.

Resume free001 in a separate free002 directory. Hash-verified hardlinks reuse the
completed temporal episode and spatial steps0..23 plus gradient updates1..24.
Reconstruct the one-tensor optimizer using saved raw gradients and identical CUDA
clipping/AdamW, verifying every before/after parameter hash, every actual delta
and all counters. Reconstructed24 arithmetic updates are reported separately and
their time is charged. If any comparison fails, stop without claiming restoration.
Only then replay failed native24 exactly and perform the remaining6 model-gradient
updates. Preserve all malformed outputs. No sample/seed/loss/dtype changes.

Span registration is conditional on independently audited fixed endpoints in
free002. The same v2 support distinction applies. Every allocation retains the
v1 resource/accounting constraints; old files and pins stay unchanged.
