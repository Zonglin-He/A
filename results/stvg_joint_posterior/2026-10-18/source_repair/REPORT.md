# Bounded source-only diffusion denoiser repair

Two repaired joint priors, existing source36train/12val per domain. Source validation comes from official train and is held out only for these auxiliary models, not established unseen to pretrained Student. No target inference, no association training, no Teacher/Student calls. Original negative pilot/diagnosis and production registries preserved.

## Prelocked qualification

Values in percent; low and medium probes are source-GT-assisted interface diagnostics, not deployable TTA results. Native uses the prelocked first path; all four separate paths are also reported, never latent-averaged.

| Source | Epoch | Low input/output vIoU | Medium input/output vIoU | Native Source/rollout vIoU | Source/rollout tIoU | Domain gate |
|---|---:|---:|---:|---:|---:|---|
| hc2 | 30 | 90.987791 / 90.773746 | 61.112768 / 65.342609 | 43.280488 / 21.212924 | 67.739714 / 48.321120 | False |
| vidstg | 20 | 79.979969 / 80.735990 | 43.030994 / 45.818500 | 18.195709 / 12.479204 | 35.521063 / 39.216143 | False |

Decision: STOP: source qualification failed. Observation isolation NOT triggered; no target evaluation, identity-axis redesign, TopK or dense OPD.

All six checkpoints and all12 cases/domain remain in results. If none passes, selected minimum weighted validation loss is a negative-report checkpoint, not a qualified deployment model. Source validation selection creates development exposure, not an independent confirmation.

## What changed and what did not

Diagonal source-train-only EDM preconditioning in RAW coordinates, matched inverse-c_out-squared loss, and actual Native-rollout validation. Original raw GT+sigma Gaussian law, .03..2 range and original .03 regression inputs retained. ROI reads physical noisy state; preconditioned inputs are not passed as physical boxes. Original fixed60 epochs,360 optimizer steps/domain and original small joint architecture retained. No extra epochs, seed search or target-dependent selection.

Sampler remains a finite Native-warm-start VE approximation, not an exact draw from a generative prior. The16-step sampler and event-relative trajectory representation remain; physical-axis identity modeling is NOT implemented. Each of four paths is preserved; a mean of per-path scores is descriptive, not a coordinate-mean prediction.

## Cost and audit

Cumulative GPU reservation (including validation, selected-output replay and CPU scoring waits while holding the lease): 33.941182 seconds, against14400s cap. CPU preparation/publication excluded. This is cached-feature component cost, NOT first-use end-to-end deployment latency; no new upstream capture was executed. Original feature generation and DINO costs cannot be erased by caching.

Independent scalar/codec/selection/protected-seal checks: 1772. Original raw-noise input metrics match prior CPU diagnosis; selected GPU replay matches checkpoint metrics. All private per-path states sealed locally, omitted from public export. Source visual checks are engineering view only, not semantic identity annotation.

References: [EDM coefficients](https://github.com/NVlabs/edm/blob/main/training/networks.py), [matched EDM loss](https://github.com/NVlabs/edm/blob/main/training/loss.py), [SDA clean-estimate observation guidance](https://github.com/francois-rozet/sda/blob/master/sda/score.py). Diagonal coordinate scaling is this prototype extension, not a full EDM/SDA reproduction.
