# Single-factor identity student view at fixed calibration alignment

Question: does the mild brightness1.05/contrast.95 student view help or harm
the existing three-step calibration-alignment update? The time-only output
anchor did not recover noise gains. Four fixed source parents show that the
view changes total calibration gradients by 19%-67%, including multimodal
query tokens. This is enough to distinguish one input factor, not task harm.

Same B1 and frozen official PTD4B; old eight historical development parents,
one query each, clean/noise_medium/defocus_extreme =24episodes. For corrupted
episodes, teacher and student both receive that observed corrupted input,
never its clean counterpart. Replace only the student mild view with observed
input; keep latent/ref/event/mean parameter-anchor weights1, alignment.01,
joint0, FiLM/LN66816 and frozen gates/heads/out/textpool/backbone. No output
time or coordinate anchor. The old no-output calibration-alignment arm is
the primary comparator; temporal/both-output and all earlier controls remain.

Fresh AdamW1e-5/wd0/clip1, fixed3steps per episode, exact B1 reset. Identity
consistency starts at zero but can regularize drift after alignment updates;
do not drop those terms or call it alignment-only. No loss/GT state selection.
Source statistics remain the fixed full618/95 query-conditioned statistics.

Reuse all192 previous predictions by hash only after current observed physical
pixels/preprocessing/query/grid/time and complete sourcefit output/logits
replay exactly. Add24 predictions and72steps, total216 across9arms. Preserve
all invalid zero boxes and malformed outputs under the existing evaluator.
All prediction/support/update files sealed before reading the already exposed
LABELS_SCORER_ONLY for this new arm. Independent scalar/P3 geometry, paired
parent mean/CI and second summary audit must agree. Report v/s/t, all conditions,
native-good retention, >5pp tails, full cases/logit drift; no loss-as-efficacy.

Primary readout new minus old no-output calibration-alignment for clean and
corruption (noise/blur equally within parent), with noise/blur and B1/Frozen
contrasts fully retained. If removing view suppresses useful noise gains, keep
that negative result and reduce augmentation-bias priority. If it helps, only
an exposed development result, not confirmed generalization. No automatic64,
coefficient grid, target-selected steps or production promotion.

Finite600s engineering allocation,8GiB disk floor, cumulative GPU cap=null.
Include all imports/failures/replays/wrapper overhead in existing receipts.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
