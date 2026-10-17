# B score: agreement regularization and exact robust interpretation

The implemented score is s=.75 min(rQ,rC)+.25 |rQ-rC| on separately ranked, comparably scaled text relevance. With rbar=(rQ+rC)/2 and Delta=|rQ-rC|,

    s = .75 rbar - .125 Delta
      = .75 min_{alpha in [1/3,2/3]} [alpha rQ +(1-alpha) rC].

The identity follows from min(a,b)=(a+b-|a-b|)/2. The interval minimizer is an endpoint. For a fixed mean relevance, increasing disagreement lowers the score. Equivalently s=.5 min+.25 max: the score is monotone in either relevance, giving twice the marginal weight to the lower one. The example (1,0) versus (.2,.2) gives .25 versus .15 because the means differ (.5 versus .2); it does not demonstrate an overall preference for conflict. Earlier wording that inferred an overall disagreement reward from the positive absolute-difference term is corrected.

The robust set [1/3,2/3] remains a fixed modeling choice. Ranked values are not calibrated probabilities, and Query and Caption are not independent experts. The interpretation concerns a scalar criterion under uncertain relative weighting; it is not proof of optimal global Token allocation, semantic correctness or lower downstream risk.

The implementation also uses s inside MMR=.5s-.5max_similarity. A positive rescaling preserves a standalone ranking, but dividing only s by .75 changes MMR's relevance/diversity tradeoff. A completely rescaled equivalent MMR is .5(s/.75)-(2/3)max_similarity. Accordingly the scientific selector code and its original scale remain byte-for-byte unchanged. The paper presents s=.75rbar-.125Delta, and describes the robust form as s/.75; it does not substitute a newly scaled score.

The CPU audit checks10005 pairs including endpoints and ties, the exact robust identity to floating numerical tolerance, and an MMR counterexample. This is an algebra/interpretation correction with zero model or GT access. It changes no existing predictions, frozen T2+B implementation or previously failed confirmation conclusion. Coefficients, quotas, ID/context protection and candidate coverage remain fixed and openly specified.
