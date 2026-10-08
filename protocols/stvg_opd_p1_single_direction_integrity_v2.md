# P1 single-direction opaque integrity readback

This is a CPU integrity check during the existing P1 run. It does not change
the experiment, read prediction arrays, or authorize any partial GT scoring.
The global P1 deployment barrier still requires both directions and all 41,355
adapted arrivals before the original CPU scoring and mathematical audits.

The HC2 direction consists of 3,482 validation clip/query inputs from 237
parent movies and three complete orders, totaling 10,446 adapted arrivals.
The VidSTG direction consists of 10,303 official queries from 732 videos and
three orders, totaling 30,909 adapted arrivals. The direction seals are distinct
from the global barrier and from a scientific or paper completion receipt.

## Check and evidence

`scripts/audit_stvg_opd_single_direction_bytes_v2.py` imports only the standard
library. It verifies the existing design, selected configuration file and
original runtime code pins; checks exact order permutations, prediction and
input path coverage; and streams every sealed `.pt` file as opaque bytes for
SHA256 verification. It does not import a model framework or deserialize tensor
data. Each JSON receipt must match its payload digest and original runtime,
declare `GT_read=false`, and precede the direction seal. Prediction byte counts
and receipt schemas are also verified. Source checkpoint immutability is a
sealed metadata assertion here, not an independently measured weight comparison.

The initial HC2 readback and its exact auditor source are preserved as
`P1_HC2_ROOT_BYTE_READBACK.json` and
`integrity_readback_versions/single_direction_auditor_initial.py`. The initial
receipt claims payload, receipt, coverage and time verification. It does not
claim a before/after barrier snapshot check.

The current auditor additionally hashes the exact barrier bytes before parsing
and verifies the same digest after the opaque readback. It also rechecks the
design, runtime and auditor source digests before writing a new immutable
receipt. It requires actual Boolean `false` for GT access declarations. The
supplementary HC2 receipt is
`P1_HC2_ROOT_BYTE_READBACK_SNAPSHOT_VERIFIED.json`; no original receipt is edited.
Both receipts must agree on counts, byte totals and the ordered payload/receipt
manifest digests. The auditor uses exclusive creation and refuses to replace
an existing receipt.

`scripts/check_stvg_opd_opaque_readback_contracts_v2.py` uses only synthetic
metadata and arbitrary temporary opaque bytes. Four valid contracts and 25
rejected invalid cases cover missing/extra/duplicate coverage, wrong design
counts, runtime/config/GT declarations, digest/byte mismatches, unexpected
receipt fields and receipt-after-seal timing. These tests are CPU contracts,
not new GPU qualification or actual mathematical/state-chain validation.

## Continuation boundary

The original HC2 worker exits normally after sealing its direction; the same
finite controller proceeds to VidSTG through the current dynamic `STAGE.json`.
No controller, GPU worker, fitting configuration, input order, checkpoint,
baseline or original receipt is restarted or changed by this readback. EATA
and its preparation remain paused by the user.

GT, dense geometry, gradients, Adam/state chains, effectiveness, bootstrap
statistics, costs, cases and report figures remain the separately required
post-global-seal audit. This integrity evidence alone establishes none of
those results. Code and safe integrity metadata will be included in the
existing P1 public closing export; private media, annotations, weights and
raw prediction/state payloads remain excluded.
