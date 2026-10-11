# Original orders and every reward round

This additive CPU readback publishes 95470 actual saved reward-round observations (32 original actions each), including mean, variance, maximum, minimum, zero fraction and constant-reward flags. It verifies 1792 original Full fits against their sealed hashes and all 3553 quality observations against the original first/all-round fractions. It makes no model calls and reads no additional GT.

PER_ORDER_DIAGNOSTICS.json.gz reports each original P2 stream order for all historical sources and the development exclusion: four quality/actionability cells plus their margins, real decoder/interpolation/outside-Native groups and identical evaluable-parent intersections. Each parent is weighted equally after frame/arrival averaging; 10000 descriptive parent bootstrap draws use seed 20261011. The root independently reproduced 309886 scalar comparisons. Original source ordering is real; bootstrap is uncertainty, not an order rerun.

Reconstruct REWARD_ROUND_ROWS.jsonl.gz by concatenating REWARD_SCALAR_MANIFEST.json chunks in listed order, verify bytes/SHA256, then gzip-decode. The original main SUMMARY/REPORT/PDF and every old scientific runtime remain unchanged. This does not close the pending LN-only, expanded-state, calibrated-baseline or deployment-latency studies.
