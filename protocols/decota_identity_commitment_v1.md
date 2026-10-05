# Hard latent path commitment and duration-bias diagnostic

User authorization: 2026-10-05 following public 03b8830. Research candidate;
production CURRENT is unchanged. No previous queue is resumed.

## S-next: fixed prestate

Use the same 48 sources per dataset (32 search,16 historically exposed confirm),
one fixed query/source, two original orders, clean and five existing 5% pixel
conditions: 1152 matched arrivals. Use P1 pre-arrival state and cached hidden
inputs, four actual old DINO positions and unchanged Native temporal readout.
All spatial arms: joint1792, Adam .03 beta(.9,.999),eps1e-8,10 steps, reset Adam,
choose first minimum of their own no-GT loss among step0..10.

1. Frame-Top1: EXACT existing admitted-frame top1 energy; not raw ungated top1.
2. Track-Marginal: EXACT R3 sum-frame IoU path energy with fixed path posterior.
3. Track-MAP: fixed E step score = sum raw DINO score + prestate IoU + adjacent
   physical-time IoU (all coefficients1); enumerate all Cartesian valid paths,
   at most81. First argmax. Fixed through all steps. M step minimizes
   logsumexp(mean-frame GIoU /tau)-selected-path mean-frame GIoU/tau, tau=1.
   No posterior averaging/authority. Duplicates and missing frames retained;
   singleton path gives zero loss and a no-op, empty evidence a no-op.

Existing Top1/Marginal fits are receipt-verified reused, with first two clean
inputs per dataset recomputed and compared bitwise before the MAP run. New
predictions are sealed before fresh scoring. This compares complete schemes:
MAP changes loss as well as path treatment; Top1 also retains original admission.
IoU continuity is not an identity guarantee; relative GIoU is not invariant to
expert box displacement. E and M have different objectives, so this is not a
standard EM likelihood-monotonicity claim. Cartesian contrastive normalizer
factorizes over frames; it does not learn a new temporal tracking model.

Primary: corrupt source-macro vIoU gain versus same Before; paired10000-source
bootstrap; clean/orders, gross gain/loss, >5pp/>20pp tails and observation
MAP/best/expected overlap reported. Select a SINGLE shared interface using search
only: MAP must improve mean versus Top1 on BOTH datasets, have no >20pp harms,
and paired lower95 versus Top1 >= -0.005 on both. Otherwise Top1 is the frozen
fallback. Confirmation never chooses a different interface. Marginal is a control.

## Independent online and budget robustness

The frozen search choice is run on actual own LN trajectories, source reset per
dataset/split/condition/order/stream. Query residual/Adam reset each query; LN
writeback1/16. Native temporal parameters/readout frozen. Episodic100 (no writes),
online100, and five independent deterministic schedules each with nested25/50%.
Ranks sha256(`decota_identity_budget_v1:seed:dataset:split:row-key`), seeds0..4.
Roster fixed across conditions/orders, no GT; do not force any source in/out.
12 streams ×1152=13824 logical predictions. Episodic identical two-order inputs
may be receipt-reused; independent online streams are genuinely executed.
Report schedule means/ranges, each schedule's source bootstrap, paired source
bootstrap after averaging schedules (conditional on these five schedules),
expert/nonexpert, clean, two-order and severe negatives. No claim that sparsity
is intrinsically safer. No full-query/full-dataset promotion is implied.

## T-next: independent CPU diagnostic

Use ALL existing valid raw UVTG durations in matching receipts (duplicates kept,
no confidence/topK/threshold); no missing cache is generated. Units physical
half-open frame intervals. `rE=median(log(duration_m))-log(native duration)`;
evaluate both same-domain and already-sealed opposite-source Native predictions.
Deduplicate the two stream orders: each source-condition input is counted once.
Seal all unlabeled rE/native inputs before GT. GT then supplies
`rGT=log(GT duration)-log(native duration)` for post-hoc source-balanced analysis.
Report cache coverage/missing sources, raw and within-source centered correlation
(common subtraction of native can induce spurious correlation), log expert vs
GT duration correlation, per-source condition-averaged median aggregate signs,
10000-source bootstrap intervals, clean separately. Limited expert-covered subset
is not the full target stream. No labels choose median formula or a calibration.
Only a positive cross-domain confirmation diagnostic can motivate a separately
locked later slow temporal state; no EMA/clip parameter is invented or executed.

All results including negative cases go to Zonglin-He/A; preserve raw private
payloads, publish only anonymous metrics/config/code and exact-byte verify.
