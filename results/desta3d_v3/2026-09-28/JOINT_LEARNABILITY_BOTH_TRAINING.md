# Joint mixer: both locked training seeds complete

Both seeds completed one full epoch on the same 618 source queries / 95 Vid parents.
Each has 155 actual Adam calls, no empty windows, eight restored integer-key
optimizer states at counter 155, deterministic full coverage, and an unchanged
union basis. PTD4B/B1/union256 are frozen; only the 103,424-parameter mixer is trained.
These are training audits, not a measured privileged teacher advantage.

| Training audit | Seed 20260928 | Seed 20260929 |
|---|---:|---:|
| Queries / parents | 618 / 95 | 618 / 95 |
| Windows / actual Adam calls | 155 / 155 | 155 / 155 |
| Clipped windows | 58 | 52 |
| Fixed schedule zero-LR final window | 1 | 1 |
| Event missing: no observed event frame | 6 | 6 |
| Event missing: absent native support | 1 | 0 |
| Spatial missing: no annotated box at native anchors | 7 | 7 |
| Spatial missing: absent native support | 1 | 1 |
| Training native-format failures retained | 1 | 1 |
| Early event failure with a valid endpoint objective retained | 0 | 1 |
| Maximum correction / F norm | 0.1354551763 | 0.1354470627 |
| Worker + non-overlapping wrapper seconds | 8322.032340 | 10194.156771 |

The fixed correction bound is 0.1354558043. Missing objectives do not remove
queries or change the accumulation divisor. Both runs retain a format-failed
query whose spatial pass was not started. The second seed's failed generation
still exposed valid time logits: two 32-class endpoint actions, exact cached
replay, finite CE. Format validity and availability of an action objective are
separate contracts. No teacher-forced GT prefix is supplied.

The initial CPU auditor rejected every early event failure; v2 correctly
accepted the first seed's failure with both supports absent. Applying v2 to the
second seed uncovered its remaining overly strict assumption that an early event
failure always removes the time logits. Its original code and failure evidence
are preserved. Isolated v3 checks the original decoder's early return, explicit
absent spatial support, and any available endpoint objective/replay. Seed two
passes v3; seed one's valid v2 report remains. No training, prediction, label,
metric, scientific pin or GPU run was changed or repeated for these CPU fixes.

Final checkpoint SHA256:

- Seed 20260928: `0126916f03ed7405d82a12846a577453373195d90ff5cf35d1de6948bf7ab01c`
- Seed 20260929: `daf1bb770b792af5d3620f63b56d66b4aca39a03b7863bcc24a3a580f13f8e51`

Settled historical GPU-allocation accounting after both seeds is
61649.367593102135 seconds, cap=null, including every prior recorded failure,
load, replay and measured non-overlapping wrapper overhead. Confirmation
inference is a separate running allocation and is not charged from an estimate.

Confirmation uses 447 queries from 31 newly locked source parents, with B1 and
both fixed final seed checkpoints. All 1,341 predictions must be sealed before
metric scoring. Source GT is explicitly privileged evidence at inference;
this is neither label-free TTA nor a claim of globally untouched or
pretraining-unseen data. Native utility, parent CIs, negative tails and native-good
retention will determine qualification. No expert, OPD or target run follows
from training completion alone.

CPU audit limitations: no inspection of the departed worker's memory, no new
full-backbone tensor copy, and no full per-query vocabulary matrices or raw
training gradients saved. Frozen scope uses locked worker assertions and saved
hashes; this is distinguished from direct live-memory observation.
