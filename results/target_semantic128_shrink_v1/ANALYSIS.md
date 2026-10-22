# Expanded target-semantic prototype study: completed, not qualified

The four-family experiment is complete. **One-round common-mean shrinkage reduces
serious harms but does not establish improved event discrimination or stable HC
increment.** No new full evaluation was started. The user-cancelled07fallback remains
disabled and originalpaperHC1902/Vid1373 remains paused.

## Locked selection and one-time confirmation

Input SHA selected128 official parents/direction, onequery/parent, from the full
227/732parent pools. Each direction has64development and64confirmation parents.
These are prospective candidate partitions, not pristine tests: old full results
were already exposed. New labels did not choose the cohort. All256 acquisitions and
all18arm predictions sealed before new scoring. Only development labels selected
kappa and the separate beta control, threevalues each; there was no stacked shrink+
beta, increased capacity, ROI/layer/step scan or perquery GT choice.

The fixed rule selected **kappa16** and **beta.75**. No configuration passed every
development admission condition, so the predeclared fallback selection froze the
maximum balanced complete-v utility **as an unqualified diagnostic**, not for full
admission. The configuration file preceded the one-time confirmation evaluation.
No confirmation result selected another configuration. Unselected configurations
were not GT scored on confirmation.

## Four-family results

All cells are ΔtIoU / Δcomplete vIoU percentage points relative to unchanged Global6.
Each cell uses64 parents/queries, with queryweights identical to parentweights here.
The final official full protocol remains queryweighted across all queries.

| Frozen family | HC development | HC confirmation | Vid development | Vid confirmation |
|---|---:|---:|---:|---:|
| Original2round |−0.1849 /−0.0486|−0.5237 /−0.4156|−0.5884 /−0.3431|+2.1439 /+1.8491|
| Original1round |−0.1113 /+0.0164|−0.1105 /−0.1976|−0.4795 /−0.1157|+1.8825 /+1.4841|
| **1round+shrink kappa16** |**+0.4274 /+0.3209**|**+0.0914 /−0.0268**|**−0.4953 /−0.1748**|**+1.1331 /+0.6891**|
| 1round+weak beta.75 |+0.2656 /+0.1555|+0.0895 /−0.0649|−0.4132 /−0.0995|+1.2910 /+0.8848|

HC confirmation shrink−Global6: timeCI[-0.9398,+1.0692], complete-vCI[-0.6166,+0.5277].
Vid confirmation: timeCI[-0.1711,+2.6100], complete-vCI[-0.0029,+1.5292]. These
intervals do not establish positive population gains. HC shrink time improves23,
harms17,ties24; Vid improves21,harms16,ties27. Relative to Source, confirmation
shrink time changes are HCshort−0.1923/long+0.6107, Vidshort+1.3354/long+4.5283pp.
HC complete-v relative to Source still increases+3.3926pp, Vid+4.8505pp, but these
include the preserved spatial framework and must not be presented as time-module
increment. Failed HC short Source and Global6 complete-v conditions remain failures.

Seriousdecline means morethan5pp relative to Global6. Confirmation time losses fall
from20 to11 queries; complete-v losses from8 to3. Their sum28→14 counts two metrics,
not distinct queries. The union also falls20→11 here because the serious-v losses
are contained in the serious-time query sets. This is real harm reduction but does
not by itself establish generalization or event-specific correction.

Shrink−weak confirmation time is HC+0.0019pp CI[-0.5573,+0.5600], Vid−0.1579pp
CI[-1.0661,+0.5094]; complete-v is HC+0.0381 and Vid−0.1957pp. There is no reliable
prototype-estimation advantage over ordinary evidence weakening.

## Why scalar mean shrinkage mostly weakens the update

The independent source-only algebra audit verifies a structural property. Let P and
N be positive/negative softweight sums in one fitted temporal parity, and let mu0
be the unconditional mean. Complementary weights imply both centered prototypes
lie on the same line. Applying the same scalar kappa gives

    delta_shrink = gamma * delta_original
    gamma = P*N/(P+N) * [1/(P+kappa) + 1/(N+kappa)].

Across512 actual fitted folds, the maximum absolute vector discrepancy is1.39e-16
and relative discrepancy5.07e-15. Thus this shrinkage preserves the prototype-contrast
direction. Nonlinear Bernoulli curvature can change exact scores, but to first order
it primarily changes evidence strength. It does not separately repair unreliable
semantic coordinates. This explains why it can reduce harm while resembling beta.
It does not prove every regularizer or prototype method ineffective.

## Candidate ranking and implementation checks

Post-confirmation candidate audit uses top64 original-prior candidates, matching
samegridspan and priorratio<=2. GT only evaluates their quality; no oracle interval
enters inference. Samegridspan is not exact physical-duration equality; gap counts
are saved. Queryweighted pair accuracy on HC confirmation is originalone47.15%,
shrink46.47%, weak47.15%; nativeposterior51.95% on the matched set. Only62/64 queries
have candidate quality disagreement; all remain in final task metrics. Shrink does
not repair the raw evidence ranking in this test. On Vid confirmation, raw evidence
is57.88%→57.97%, a tiny uncertain change. Weak scaling cannot add raw ranking
information, although combining with a native prior can change posterior ranking.

A separate provider-vs-maintained-path regression recovers **all128 prior Global and
Target semantic probabilities AND pre-head vectors bitwise**. This excludes this
particular implementation-path discrepancy as an explanation for the older HC64
optimism. It does not make old/new panels comparable samples or establish no other
bug exists. OriginalSource/E/Full/interval/eight-position checks also pass all256
new acquisitions.

A source-only contribution audit finds median HC clipconfidence below.1 on97.06%
of coordinates, accounting for91.44% of absolute current logevidence; corresponding
Vid figures95.00% and57.76%. Clipconfidence here is reconstructed from clippedp and
is an approximation, not true relevance. Lowconfidence coordinates may still carry
useful negative information: these statistics identify a class-allocation hypothesis,
not permission to drop dimensions, amplify scores or claim a root cause. Original
training uses wordhead frame-mean logits and BCE wordtags, not frame-event labels.
No class weighting candidate is scored or adopted in this phase.

## Evidence, cost and state

[Development report](develop_report.json), [confirmation](confirm_report.json),
[all dev signed rows](develop_signed.tsv), [all confirm signed rows](confirm_signed.tsv),
[dev severe tails](develop_severe.tsv), [confirm severe tails](confirm_severe.tsv),
[frozen configuration](FROZEN_CONFIG.json), [failed qualification](QUALIFICATION.json),
[candidate audit](CANDIDATE_RANK_AUDIT.json), [direction audit](DIRECTION_AUDIT.json),
[class audit](CLASS_CONTRIBUTION_AUDIT.json), [implementation bridge](BRIDGE_AUDIT.json)
and [verification](VERIFICATION.json) preserve both positive and negative outcomes.

Actual source acquisitionv3 takes1023s, including249 Source queries/2490 additional
semantic ASA reads;7 prior complete records are imported byte-for-byte. Two earlier
Source passes failed before accepting their member and remain overhead in preserved
v1/v2 logs, not fresh successful inference or a method result. Earlier7 costs are not
added as new v3 time. RGB/Source/semantic work is research acquisition, not deployment
speed. Additional engineering bridge uses512 ASA reads/15.21s, no Source visual/
fusion/time forward. CPU all-arm prediction44.70s, no model/expert/observation. No
expert call, new expert model, added observation, training,448 or320 occurs.

Five CPU core tests pass. Independent official/dense metric checks total7168 with
maximum error below2e-10. Source48/provider434 remain unchanged. This phase changes
no maintained default. Starting a full despite the failed admission would require
an explicit user decision accepting the recorded tradeoff, not relabelling the
candidate as having passed.
