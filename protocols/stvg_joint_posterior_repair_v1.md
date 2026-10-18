# Bounded source-only joint denoiser qualification

User-authorized reopening of the failed joint posterior, not a new main method.
Old pilot and gap diagnosis are immutable. No SAM2/Qwen/new DINO, no target
inference/scoring, no Student updates, no association training, no TopK/dense OPD.

Use exactly the cached old 36 source-train and 12 source-validation parents in
each domain. Source validation is held out only for the new auxiliary model;
its unseen status to the pretrained Student is NOT established (official train).
Existing source checkpoints/features/labels and Native/direct controls stay fixed.

Train at most one joint model per domain, one seed, 60 epochs x 6 batches,
AdamW lr .0003, weight decay .0001, gradient clip1, original architecture.
Training remains raw GT + sigma*Gaussian, log-uniform sigma .03..2. Do not train
on Native+noise or claim that Native warm starts are exact prior samples.

Diagonal EDM coefficients use ONLY source-train empirical mean and population
standard deviation: each temporal logit separately, each spatial coordinate
shared across annotated event knots; standard-deviation floor .05 is numerical.
In raw coordinates D=mean+c_skip(x-mean)+c_out F(c_in(x-mean),log(sigma)/4,c),
c_skip=d^2/(d^2+sigma^2), c_out=sigma*d/sqrt(d^2+sigma^2),
c_in=1/sqrt(d^2+sigma^2). Loss weight is 1/c_out^2; temporal/spatial blocks
are equally weighted per query. Dynamic ROI always reads RAW physical state.
Raw perturbation units do not change; original diagnosis epsilon and sigma.03
inputs are reproduced exactly and compared to saved probe metrics.

Validation checkpoints fixed before training: epochs10,20,30,40,50,60. At each:
all12 low/mid/high paired GT perturbations (original levels15/8/0); 4 separate
Native+2epsilon warm-start paths, 16 steps, no observation guidance. No averaging
latent coordinates. Primary deployable path is prelocked sample0; average of
per-path metrics is diagnostic and also gated. Seeds identical across epochs.
Select highest sample0 Native mean vIoU among checkpoints satisfying the full
within-domain gate; tie choose earlier epoch. If none, select minimum EDM-weighted
multi-noise validation loss FOR REPORTING ONLY, record failed qualification.
No retraining or epoch extensions based on results.

Within-domain engineering gate (both domains must pass):
- Original raw sigma.03 near-GT mean vIoU loss <= .01 (1pp), all cases reported.
- Medium original level8 mean vIoU strictly improves over unchanged noisy input
  by >1e-6, preventing an identity-only pass.
- Native rollout sample0 AND mean per-path vIoU each exceed raw Source mean by
  >1e-6. Native tIoU sample0 and mean per-path drop no more than .01; mean decoded
  duration/context for both >= .5 of Native. Report individual collapse/tails.

If either domain fails STOP before observation isolation or target work.
If both pass use SAME checkpoints, original source-val24 only: no guidance,
mixture p=.5 fixed (outlier component retained), frozen OLD association weights.
Same seeds/variance/expert observations; clean-estimate likelihood with full
chain-rule gradient to noisy state, no detached denoiser. No identity-time-axis
redesign here; old exterior hard mask limitation stays disclosed. Synthetic
finite difference validates this gradient before using source observations.

Cumulative GPU-reservation time <=14400 seconds, including model validation,
qualification and conditional observation isolation, never reset on resume.
One serial worker, existing shared GPU lease, memory.high16GiB/max20GiB,
swap1GiB, host6GiB/disk30GiB floors. CPU preparation/publication excluded from
GPU-reservation time but separately measured. No first-use end-to-end latency
claim: inputs were cached; prior upstream capture cost reported separately.
Budget failure preserves partials and does not trigger another configuration.

Independent aggregation/codec/source-noise regression/protected seals must pass.
Publish code/protocol/anonymized scalar results and all negative gates to
Zonglin-He/A, verify HTTP bytes/SHA/blob/mode/tree/main+research fresh CAS;
exclude media, raw states, GT coordinates, query text, weights and raw tensors.

References checked: https://github.com/NVlabs/edm/blob/main/training/networks.py
and training/loss.py (coordinatewise preconditioning extension); clean-estimate
guidance principle https://github.com/francois-rozet/sda/blob/master/sda/score.py .
This is not a full reproduction of EDM image training or SDA VP sampling.
