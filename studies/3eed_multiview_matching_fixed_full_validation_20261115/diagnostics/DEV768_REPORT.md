# Fixed R99 endpoint mechanism diagnostic

This is a post-seal analysis of all768R99 development inputs, not a new method, target selector or tuning run. It uses source, one-view and three-view last-linear states with identical cached native features. CPU FP32 head replay differs from stored GPU probabilities by at most1.2128151900547834e-6, with noTop1 differences across the768inputs. The initial over-strict1e-6 CPU parity assertion failed and its log/code/output directory are retained; the measured approximation is reported rather than called bit-exact. All reported task changes and pair probability margins come from the already-sealed GPU outputs, never from CPU replacement predictions.

Equal-six mean centered score fit errors:

| state | F | A | B | unfitted C |
|---|---:|---:|---:|---:|
| source | .005371020 | .005371020 | .005371020 | .005371020 |
| one-view | .001249637 | .001637358 | .001640881 | .001769373 |
| three-view | .001313668 | .001314453 | .001309930 | .001498437 |

Three-view fitting reduces error on the additional fit views and C while increasing F error slightly. This supports a reduction in single-view fitting specialization, not proof of correct pseudo-supervision. Every one of7IoU improvements AND3IoU regressions versus oldFC has lower C fit error. Thus this diagnostic does not support an auxiliary-error-decrease gate for accepting beneficial grounding changes.

22FC query changes occur;18ofthese occur without changing either F or C individualTop1. Only10of22changeIoU (7improvements/3regressions). Hence average-readout probability margins are a dominant observed path; unchanged individualTop1 does not imply unchanged final decision. ZeroIoU ties/duplicate candidates remain recorded, not reinterpreted as certified instance identity. The fullR100 recipe was locked before this diagnostic and remains unchanged.

No target labels are used in fitting here (there is no fitting), no newGT IoU is calculated, and no expert or original source parameter changes. Auxiliary and task signals are explicitly separate. The unresolved alternative is weak task information inG: improving its approximation can still help or hurt grounding. Full fixed-configuration expansion is required before deciding whether this candidate is worth retaining.
