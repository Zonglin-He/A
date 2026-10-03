# Continuous endpoint-balance addendum (post-hoc)

After reading the first locked analysis, strict centre-dominance was found to
include cases with one large endpoint movement and a tiny same-sign movement
of the other endpoint. This addendum is explicitly post-hoc, not a predeclared
test on unseen outcomes. It introduces only continuous descriptive quantities,
no acceptance thresholds, source selection, model, candidate or scorer change.
The original tied-oracle choice, local radii, input metrics and state remain fixed.

For a=|ds|, b=|de|, d=a+b:

* start fraction a/d, end fraction b/d (sum 1 for nonzero movement);
* endpoint balance 2min(a,b)/d (0 single endpoint, 1 equal movement);
* centre component 2|c|/d, extent component |l|/d (max 1, not additive).

For zero movement use zero; such a row has zero added capacity and contributes
no mass to the gain-weighted quantities. Weight each quantity by O32−O8 within
the original source/order/condition aggregation, divide by aggregate added
capacity, and bootstrap numerator/denominator together. Keep all eight corrupt
and clean dataset/panel groups, 10,000 draws and the original denominator-zero
disclosure. Verify geometry from signed endpoints with an independent
aggregation/bootstrap implementation. No resulting value is an online rule.

The purpose is to prevent claiming that a strict co-directional classification
proves wrong event identity, or that near-one-boundary errors must be small.
