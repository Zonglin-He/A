# Frozen source Native-path diagnosis

Question: separate initialization damage, one-step denoiser behavior, and recursive
sampling after the failed two-model source qualification. No new training,
checkpoint selection, target inference/GT, observations, Teacher/Student calls,
new backbone or proposed main-method changes.

Freeze repaired HC epoch30 and Vid epoch20, existing source-val12 per domain,
cached conditions/labels, raw16-level sigma2..03 and four original CUDA noises
(seed20261018+1000+within-domain index). Source validation was held out only for
the auxiliary models; it is NOT an untouched Student evaluation set.

Before replay lock:
1. Reconstruct original Native+2epsilon path exactly: preserve initial x, each
   step's noisy x, predicted clean D, proposal and all four final paths.
2. Paired GT+2epsilon path with EXACT SAME epsilon: source-GT-assisted positive
   control only; not deployable output.
3. No-initial-noise counterfactual with the SAME sigma schedule (starts Native
   at sigma2): isolates initialization, but is deliberately off the training
   corruption law. Do not call it calibrated diffusion or select a new method.
4. At original levels15/8/0, one-step GT+sigma*epsilon and Native+sigma*epsilon
   with the same epsilon. The GT noise here is the original CUDA warm-start
   noise, different from the prior separate CPU-probe seed. Old low-noise
   qualification numbers stay unchanged. Report input and output separately.

Primary decomposition: Native codec -> Native+2epsilon -> first predicted clean
D(x,2) -> final16-step output, per query and sample then equal-query means.
These deltas telescope but are NOT a unique causal attribution: later effects
depend on the initial condition. Report final minus noisy input and final minus
one-step separately. No-initial-noise comparison isolates only this inference
toggle; no noise search or training change. Compute v/t/s, duration fraction,
equal-block raw GT state loss, and normalized state error. Preserve raw states
privately, never average coordinates or choose paths using GT.

Pure sampler contract: identity denoiser leaves input unchanged; original
proposal equals Euler x+(sigma_next-sigma)*(x-D)/sigma; no wrong-sign claim.
Independent official/DenseTube metric checks; original terminal-state/scalar
reproduction, protected models/configs/old predictions hashes checked.

Budget: one GPU lease <=300s, added to previous33.94118248508312s (total remains
under original14400s cap); no reset or additional training authorization. Host
6GiB/disk30GiB floor and16/20GiB memory,1GiB swap guard. Pure CPU scalar audit,
plots and publication separately measured. Old results and failed gate remain.
Publish anonymized scalars/code/negative diagnosis through existing GitHub
workflow; exclude Query/Caption, GT or predicted coordinate arrays, media,
weights and tensor payloads. No deployment change or subsequent automatic run.
