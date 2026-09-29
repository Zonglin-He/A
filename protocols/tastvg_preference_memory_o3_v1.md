# O3: Temporal Critic Preference Memory

Last temporal slow trial authorized by attachment bda2202b. Stop global768 linear
ranker and its LR/alpha/normalization search; keep Fast specialist rerank. New
mechanism stores local signed pair experiences rather than fitting one weight.
Original O1 heterogeneous32-source roster/order/conditions,25% expert positions
1,5,...29, same raw768D features/candidates/native envelope score/UniversalVTG.
No model/backbone/teacher recomputation, no optimizer, GPU0. Original inputs and
all historical outcomes remain unchanged. This is exposed development, not fresh.

Initialize empty memory once. At expert arrival, emit identical expert-best Fast
output first, then store each unordered strict teacher pair i<j as normalized
phi_i-phi_j with y=sign(r_i-r_j), provenance(position,i,j). Same inherited1e-12
teacher tie rule; zero feature difference has no defined direction and is omitted.
These are algebraic tie/undefined-vector rules, not reliability thresholds.
At nonexpert arrival, use past memory only; no write or current teacher read.
No eviction, normalization fitting, gate or parameter tuning. Fixed m=3.

Pair orientation has no semantics. Each stored(d,y) represents both directed
views(d,y) and(-d,-y). Retrieve the nearest of these views per historical pair,
then top3 distinct historical pairs by cosine: equivalently descending absolute
cosine of canonical d. Its contribution is original signed cosine*y. This avoids
counting a pair twice. Exact retrieval ties preserve insertion order (arrival,i,j).
Compute each current i<j vote once, set reverse to its negative. Positive vote
adds one Borda win to i, negative to j, zero to neither. Zero current difference
or empty memory yields no vote. Candidate maximizes wins, then original native
score, then earliest index. No alpha/native-gap mixing. Save all neighbor IDs,
cosines, signs, votes, wins, memory size/hash before/after every arrival.

Arms only Budgeted Rerank, Fast Rerank + Preference Memory, Full Rerank reference.
Primary24 nonexpert arrivals: paired delta tIoU/vIoU against Budgeted. Secondary
all32, exact Full agreement, positive/negative cases and >5pp harms. Fixed paired
10000 bootstrap seed20260929 is conditional on this shared trajectory, not new
online streams. Expert positions identical between all arms. GT never influences
memory, retrieval, tie breaks or parameter choice. Online outputs/state seal
before accessing old24 Full teacher caches; all3 outputs seal before consuming
old O1 dual-verified cached GT-derived candidate metrics. No raw GT reload.

Independent local scalar/NumPy reconstruction audits all states/retrieval/votes;
public scalar audit recomputes wins/selections/metrics/CI. New code tests empty
memory, direction reversal and top3 uniqueness. CPU only; no follow-on temporal
rescue regardless of outcome. Clear gain may retain memory as candidate; weak or
negative leaves sparse Fast rerank as temporal evidence module. Neither criterion
promotes production or establishes universal scientific impossibility. Next S0
spatial expansion is separately authorized, with expert/cohort fixed before run.
