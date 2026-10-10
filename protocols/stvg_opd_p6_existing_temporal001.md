# Original fixed P6: matched existing temporal signal and head diagnostic

This implements section 9 of the original attachment 029fff2e and the already
locked P6_hc2/P6_vidstg entries in LATER_DESIGN_LOCK.json. No cohort, source,
spatial OPD configuration, expert, native decoder or main-method registration is
selected using this diagnostic. P5 must actually close root/view/public/archive
before qualification or deployment on the same serial GPU lease.

Each direction uses its original 32 hash-selected historical parents, one query
per parent, clean RGB, original full frame IDs and both original temporal offsets.
All four deployment arms share exactly the original P0 source checkpoint, input
pixels, frame IDs, native boxes/interval and full packed frozen-DINO input binding.

1. Native: frozen source boxes and original native temporal prediction.
2. Actionness projection: the existing final-method structured actionness target
   from the unchanged median/MAD, physical-time-cell and native-prior recipe;
   frozen source boxes, no optimizer and no additional specialist calls.
3. Existing temporal head: unchanged 66306-parameter two-layer temp_embed,
   original NLL plus margin .2 loss, source reset and fresh AdamW per query,
   exactly five steps, last state, eta .25 shrink and original full-model
   reinsertion. Temporal lr .001/center_fraction1 for Vid-source to HC2 uses
   the original vid_to_hc1 representative recipe without HC2 tuning; HC-source
   to Vid uses original lr .1/center_fraction .5. Betas .9/.999, eps1e-4,
   weight_decay0, prior_weight .1, epsilon1e-6 remain unchanged.
4. Spatial OPD: exact complete preserved P0 on_policy order1 payload for the
   same query and input, including every original predecessor and inherited LN
   state. This is a matched diagnostic subset of the original 128-query history,
   not a fresh 32-query online stream. Native WHEN and dataset-specific spatial
   configuration remain unchanged. Its own Before output can be described
   separately; it does not remove the different temporal/spatial state histories.

Qualification uses the first two fixed queries per direction, actual original
unrecorded versus separately recorded complete head fits and full reinsertion.
Every original scientific output/state/path/loss remains bitwise identical.
Qualification is excluded from formal outputs; formal first-two fits must match
qualification before acceptance. Source process hash is checked before/after;
no DINO instance is created. No GT, score rows or result summaries are read by
the qualification/deployment workers. Each query has an immutable native input
capture, complete optimizer trajectory, original input/receipt/hash bindings and
four complete outputs. All 64 queries and 256 deployment outputs must globally
seal before any new P6 GT scoring or GT-head oracle.

The numerical recorder uses observation-only forward/gradient and original
optimizer step hooks, without changing tensors, reductions or optimizer code.
Independent CPU64 algebra checks the actual saved projection, NLL/hinge and
native-head-output derivative, the two-layer head VJP, all AdamW moments/states,
last-state/shrink/source reset and native physical readout. Float32 arithmetic
comparison uses a declared 2e-5 floor plus operation-count gamma(n) bounds on
absolute product sums (loss/projection CPU64 comparisons use 1e-9 relative
scale). This is an independent audit of the frozen-prefix temporal head; it is
not an independently implemented full decoder Jacobian, CUDA transcendental
proof or future numerical/OOM guarantee. Original spatial receipts retain their
original dispatch and thresholds.

Only after the full deployment seal, reproduce the original offline GT-head
capacity diagnostic: each offset target is the first legal start<end pair with
maximum physical GT tIoU, then the same five-step AdamW and eta .25 readout from
the saved source head/frozen prefix. This CPU fit is separately labeled offline,
supervised and unavailable at deployment. Its CPU source-logit precision and
native interval agreement are actually checked, not called GPU bitwise. The
oracle is neither a fair deployment baseline nor a selected checkpoint/step.
The oracle uses exactly the original official truth span returned by the existing
scorer, without extending its end frame or changing metric conventions. All
confidence intervals use 10000 paired original-parent bootstrap draws within
each target, seed 20261006, and have no multiplicity adjustment. The cohort is
historically exposed and bounded; this is not a fresh generalization estimate.

After the seal, independent official and dense fixed-GT-frame metrics, query and
parent macro, all 10000 paired parent bootstrap intervals, every negative tail,
actual capture/fit/oracle cost, positive/negative signal and failure cases, actual
PNG/PDF and RGB view, anonymous all-result remote bytes/SHA/Gitblob/tree checks
and archive check/snapshot/check are required. RGB/query/caption/GT geometry,
weights, native logits/boxes/actions and fit/gradient/Adam inputs remain private.
EATA and every historical paused queue stay paused. No new branch, algorithm,
expert, training sweep, scorer, gate or memory. Preparation/CPU contracts/
qualification/finite worker completion alone do not close P6 or the paper.
