# Saved-gradient addendum to the identity/update experiment

CPU readback used the exact previously sealed eight-source tensors and produced 75 anonymous step records. It did not execute new autograd, model inference or GT scoring. Ordinary/reward-weighted OPD gradients were saved; Direct gradients were not saved and remain explicitly unknown rather than reconstructed as measured gradients.

HC01 ordinary IoU-OPD:

| Proposal (1-based) | Saved gradient L2 | Actual parameter displacement L2 | Gradient dot displacement |
|---|---:|---:|---:|
| 1 | 16.386926 | 0.423319 | -3.063747 |
| 2 | 4.042729 | 0.359915 | -0.957174 |
| 3 | 3.399374 | 0.344879 | -0.770714 |
| 4 | 7.555475 | 0.323902 | -1.164286 |
| 5 | 2.257460 | 0.272835 | -0.191184 |

The fourth proposal is a descent-direction step for the saved stochastic loss to first order, and its parameter displacement norm is smaller than the preceding steps. Nevertheless, teacher-anchor IoU collapses at its actual decoded endpoint. The evidence therefore does not support calling this a gradient-sign bug or simply the largest parameter step. The stochastic distribution objective, shared decoder nonlinearities, Adam state and mean-box reference agreement are different quantities. The measured endpoint rejection rule prevents this particular collapse without claiming to identify every underlying cause.

All steps retain reward ranges, importance-weight effective sample sizes, saved-gradient norms, actual displacement norms and gradient/displacement inner products in the anonymous addendum. The extra readback does not change the fixed method or select a new hyperparameter after GT evaluation.
