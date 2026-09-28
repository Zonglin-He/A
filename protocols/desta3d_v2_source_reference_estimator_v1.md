# Source-only reference-estimator sensitivity, no update

The previous source audit found a population/episodic statistic distinction;
global variance includes between-query means (~10% of total trace), while
target episodic std is within one query. This is not yet an efficacy mechanism.
Anchor and view ablations retained clean/noise tradeoffs; do not reopen a
target sweep on this observation alone.

Construct one alternative from all sealed618 train queries/95Vidparents:
mean = existing global mean exactly; std = equal-parent/equal-query expectation
of sqrt(max(second_q-mean_q^2,1e-12)). Within-query THW moments already use
equal cells. Do not weight queries by grid size. E[std], sqrt(E[var]) and
sqrt(E[second]-E[mean]^2) are different; store each diagnostic explicitly.
Preserve the original MOMENTS.json and all historical pins. Independent NumPy
aggregation must match before GPU; unequal-parent/query/grid synthetic controls.

Same fixed B1/officialPTD4B and four source train parents/queries
10016/10107/10132/10190 selected previously, no reselection. Same observed
teacher and original brightness1.05/contrast.95 student. Only reference std
changes; source mean, all loss weights(.01alignment), FiLM/LN66816, frozen
gates/backbone, physical query/grid/time and state are held constant.
No optimizer, no decoding, no source/target labels or target inputs. Actual
fields and original global-reference gradients must reproduce previous source
records before comparing the new reference. Save both raw66816 component and
total gradients, scalar terms, channel moments, parameter hash and frozen flags.

Readout: exact source-statistic difference; old/new weighted alignment and
total gradient norms, cosines and relative L2 change; unaffected components;
raw summation versus direct backward; NumPy scalar and gradient audit. Report
all four parents separately. These are not task gradients or TTA benefits.

Predeclared resource screen: if every total-gradient cosine>=.995 and relative
L2 change<=.15, treat initial signal as insufficiently distinct and do not run
another target trial. This is an engineering sensitivity rule, not a statistical
test or proof of route impossibility. Otherwise review direction versus pure
scale and competing explanations before separately registering any target
experiment; a larger gradient difference does not automatically authorize a
claim or target sweep. No coefficient/step tuning from exposed target scores.

One300s serial GPU allocation,8GiB disk reserve, cumulative cap=null. All actual
loading/failure/replay/wrapper overhead is added to prior receipts. Eight raw
cases,12 sealed files plus immutable source-reference artifact; exact unchanged
adapter, no optimizer state. Failures retain partials and receive new-version
repair rather than overwriting registrations.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
