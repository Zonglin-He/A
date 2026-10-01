# Separate native temporal-head transfer test

Authorized as the second batch in the user's 2026-10-02 proposal. It starts only
after the spatial four-arm comparison and per-dataset spatial selection seal.
The same original development cohort, pixels, offsets, orders, expert schedule,
Fast critic and selected spatial arm are retained. No full-query job is resumed.

Update only the original TA-STVG temp_embed.layers[-1] weight and bias, 514
parameters; no extra temporal ranker, prototype or model. All other temporal
parameters remain frozen. The updated head is after the actionness routing
decision, so it does not change native routing or boxes by itself. Verify this
against the selected spatial stream, including every spatial state hash/box.

The same retained Fast candidates retain their generating legal start/end span
pair for the two offsets. For each offset the full native legal span distribution
is proportional to P_start(s)P_end(e) for s<e. Its pair probability is the product
of the two offsets, then renormalized on the retained (up to eight) pairs. This
is explicitly a restricted candidate likelihood; it is not the full tube policy.
Layer-origin spans and final top-pair spans preserve native offset indices and
the original physical envelope, checked exactly before learning.

Use the same UniversalVTG max(confidence * physical interval IoU) reward scores
as Fast, rank direction and per-dataset sealed teacher temperature. Build one
proximal target using the incoming head policy as detached p_ref. lambda is
reward spread/(spread+s_ref_T); s_ref_T is the positive-spread median of baseline
A's sealed first-arrival critic scores, all original conditions/orders, without
GT. If all spreads are zero, s_ref_T=1 and every feedback update is an exact no-op.

One SGD step per scheduled arrival, learning rate equal to the dataset's sealed
learning rate, with a 0.005*source-head-L2 displacement cap. This is a declared
development starting budget, not an expert-probed parameter neighborhood or a
literature guarantee. No tuning of this new head budget in this batch.

Current output (including Fast) is sealed first. Spatial updates occur as before;
temporal loss uses detached CURRENT hidden states, then writes the head for
future arrivals only. Expert providers are fetched only once, reused for both
branches. Reset head and spatial parameters per original condition/order.

No-GT fixed first-two scheduled clean input smoke checks zero-head baseline
prediction parity, 514-only parameter update, original candidate mapping and
finite loss/gradient/cap. Both datasets' 768 new arrivals seal in a separate
temporal prediction barrier before CPU scoring. Reuse the spatial selected
baseline, report paired source-bootstrap 10000 future-corruption tIoU/vIoU,
clean/current-expert impact and spatial spillover; retain reverse/negative results.
No improvement means this head learning is not connected to the method; it does
not forbid all temporal adaptation. No promotion or extra branch is automatic.
