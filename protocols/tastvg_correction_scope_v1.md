# Episodic Rank-RKL and correction-scope functional audit

Authorized by attachment 000d5a5b on 2026-10-04. This tests the old Uniform
Sa2VA Rank-RKL A, not C1/Scale06 DeCoTA. No deployment promotion is authorized.

## Frozen science

VidSTG and HC-STVG-v2 retain their existing 32 development + 16 confirmation
sources, one query per source, two orders, clean and five 5% corruptions, and
25% expert arrival positions: 1,152 arrivals, 288 expert writes. All sources
have historical exposure. Original Paper48 sampling/pixels and same-domain EMA
checkpoint stay fixed. Parameters: query residual 256 + final decoder norm1,
norm3, norm4 weight/bias, 1,792 total. Vid lr=.033761698432507946,
teacher T=.34902548789596055, K=1; HC lr=.006097133675874025, T=1, K=8.
Student T=1, rho=.05, four fixed directions/nine probes, rank reward,
unprojected SGD and Uniform five-frame cached evidence are unchanged.

## Lifecycle experiment

Episodic resets ALL seven spatial tensors to the source checkpoint before each
query. Scheduled arrivals refresh nine probes at every unchanged inner step,
write using original Rank-RKL, and read the post-update current spatial tube.
Nonexpert arrivals have source-state output. No parameter update persists.
Order duplicates can reuse an identical source-input/condition/evidence result;
receipts preserve all logical arrivals and unchanged expert schedules.

Persistent A_before is the sealed pre-update stream; A_after reads its sealed
post-update spatial tube on that same current query. Episodic_before/after and
Frozen are reported. Primary spatial attribution fixes the source-native
interval for ALL five arms; native free intervals and original Fast output are
secondary. Fast uses the before-update support/selection; post-update boxes do
not trigger another temporal expert call. Compare A_after with E_after as a
matched current-output lifecycle test, not only A_before with E_after. The
latter changes both state lifecycle and prediction timing. No reset baseline
is silently called the deployed online method.

## Functional matrix

Use each sealed original full K1/K8 A write at its saved common pre-state.
Replace only query, norm1, norm3, norm4, or all tensors with their exact saved
post-state values. Compare target predictions under this common pre-state and
the intervened state, holding the target's common-pre native interval fixed.
Free native decode and target source-native interval are secondary checks.
These finite block interventions are NONADDITIVE and are not one-step gradient
decompositions or newly accumulated streams.

Scopes: self; another actual query on the identical media SHA; ALL eligible
future nonexpert different-video queries in the same original split/order and
corruption; frozen text-nearest/farthest among those same targets; one fixed
hash target from that future set, evaluated both in the donor corruption and
the next corruption in the locked five-corruption cycle. Clean maps to the
first corruption and is a separate control. Thus cross-corruption is paired
with the identical target query, not an unmatched mean. Near/far may coincide
with other roles; unique target inference is reused, aliases are reported.
Semantic selection uses frozen RoBERTa contextual token mean, excluding BOS,
EOS, padding, with L2 normalization and earliest arrival ties; no GT.

For Vid, choose one different-caption query per expert source by SHA256 of its
official input key, from the sanitized old full PLAN; retain its OWN original
input frame grid. Same media SHA is required. Query-conditioned frozen H must
be newly captured: swapping a caption into old H is invalid. This tests
same-video transfer across query views, not an identical-pixel text intervention.
HC has no other caption on the identical media for these 48 sources, so this
scope is N/A there. No videos/samples are added to the original online streams.
Maximum additional image captures is the locked Vid expert-source count × six
conditions. No new spatial/temporal expert calls or expert adaptation.

## Evidence, audit and decision

Lock code/checkpoint/input/candidate-target rules before prediction. Deny GT
and outcomes in GPU workers. Seal ALL episode, matrix and alternative-capture
receipts before reading any diagnostic GT. Dense official vIoU/tIoU/sIoU;
paired 10,000 source bootstrap with seed20261004, source macro after averaging
targets within donor/scope then condition/order. Development/confirmation,
clean/corrupt, expert/nonexpert, individual orders, gross gain/loss, >5/>20pp
harm and .3/.5 correct→wrong/repair are retained. For transfer, primary cluster
is donor source; target-source sensitivity and donor concentration are reported.
Only donor-source-corrupt paired changes identify relations. Shared corruption
is a controlled distribution, not proof of natural episodic continuity.

Verify original donor arithmetic/state hashes, exact block mask, all full
source resets, K-step refresh and single expert fetch; smoke checks original
A/self prediction parity and source-replay parity. Preserve failures and pin
engineering revisions. No lambda, LR, K, loss, optimizer, routing, memory or
dataset-dependent promotion. Outcomes inform a later scope proposal; they do
not authorize one. Archive check/snapshot/check, code and anonymous scalar
results/negative findings to Zonglin-He/A, verify every remote byte, then close.
