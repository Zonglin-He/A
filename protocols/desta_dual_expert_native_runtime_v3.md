# DESTA runtime v3: engineering amendment

The v1/v2 workers failed the host-capacity guard, including a fresh-process v2
attempt before its first backward. Preserve both failures and the v1 partial
coefficient field. Isolated v3 uses a 14GiB lossless activation offload ceiling
and the unchanged 6GiB host reserve, leaving additional saved tensors on GPU.
The original v7 module and historical pins remain unchanged. Before continuing,
v3 must match original C0 losses/native logits/full-gradient hashes/projected
gradients and norms exactly. This passed on the fixed first query; it is an
engineering equivalence check, not a native-utility result.

The first failed configuration resumes the exact saved post-step1 coefficient;
no optimizer or model state exists to reset. Completed configurations are reused
by hash. A CPU-only process-handle waiter hands off to the registered controller
only after a successful measured wrapper exit; failures require root review.
The first Python pidfd wrapper was unavailable; glibc pidfd_open was independently
tested on child exit in the isolated waiter repair. No scientific settings change.
