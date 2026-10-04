# Critic-DeCoTA P1: matched LN1/16 inheritance

The user explicitly authorized entering LN inheritance after the P0
readback. P0's Vid uncertainty and severe harms remain recorded; this
continuation is not an automatic promotion or a claim that P0 passed both
datasets.

Reuse exactly P0's two same-domain EMA checkpoints, 48 sources per dataset
(32 development/16 historically exposed confirmation), one query/source,
both original orders, clean plus five5% corruptions. Reuse the unchanged
four native-I0 DINO observations and cached proposal support, parser,
pixels and Frozen H. No new DINO/backbone forward, temporal adaptation,
support/temperature change, preservation, gate or sweep is introduced.

The sole new mechanism is spatial LN persistence. At each arrival, reset
query256 to zero and construct fresh Adam; inherit only the1536 last-block
norm1/norm3/norm4 affine coordinates. Run the exact P0 energy critic,
lr.03, ten updates, temperatures1/1, own-objective earliest minimum0..10.
Return its current corrected tube with the fixed source native interval.
Then commit `LN_next=LN_arrival+(LN_selected-LN_arrival)/16`; query remains
zero. Empty evidence commits an exact unchanged arrival state. Chains
reset at dataset/split/corruption/order boundaries. All1152 arrivals are
actually evaluated; order results cannot be reused for persistent states.

Readouts on the same input/native interval:

- Frozen: source checkpoint result.
- EpisodicCritic: P0's hash-bound fit, reused without re-inference.
- InheritedBefore: inherited LN with zero query, sealed before current
  evidence is used in the fit.
- OnlineAfter: current critic fit starting from that inherited state.

Every query still has the original four-observation expert budget. Thus
InheritedBefore measures next-query utility **before current-query expert
correction**, not an executed stream with25% expert arrivals. Do not call
it a formal nonexpert evaluation or conceal this budget difference.

Pre-GT cached smoke checks source-start parity against P0, a second-arrival
inherited state and query reset, selected/committed parameters and native
time invariance. CPU lifecycle contracts use the original consolidation
operators. Do not add new model forwards for this already validated cache
interface. Globally seal all1152 online payloads before new GT scoring.

Independent CPU checks reconstruct every predecessor chain, exact1/16
coordinate arithmetic, query reset, own-objective selection, NumPy energy
and Adam updates, immutable native time, official/vectorized dense metric
agreement. All signs and failure records remain. GT never chooses a state,
retention rate, cohort or configuration.

Primary per-dataset confirmation corruption comparisons are
InheritedBefore−Frozen, OnlineAfter−EpisodicCritic and OnlineAfter−Frozen.
Average the two orders/five conditions within source; paired10000-source
bootstrap seed20261004. Also development, clean, each condition/order,
first versus later arrival readouts, gross positive/negative contributions,
severe tails, .3/.5 correctness transitions and source concentration.
Do not confuse current after-correction gain with inherited before gain.

This is a bounded small-panel research experiment, not full data or fresh
test. No automatic CURRENT promotion or cross-domain work follows. Record
all actual replay/backward counts and worker wall time (not pure kernels),
publicly export code/protocol/all anonymous metrics, verify GitHub bytes,
and update RESEARCH_HISTORY check/snapshot/check and FINAL_COMPLETION.
