# Why source and target differ: frozen-artifact diagnosis

No new training, checkpoints, Teacher/Student calls, GPU execution or benchmark arms. Frozen association CPU replay and source-only GT-assisted one-step denoiser diagnostics are labeled separately. Original pilot cf213404 is preserved.

## Correct directional comparison

Each row tracks the SAME prior from its12 held-out source queries to16 target development queries. Both groups were exposed previously. Numbers are vIoU percent.

| Prior direction | Source Direct | Target Direct | Source joint | Target joint |
|---|---:|---:|---:|---:|
| hc2 → vidstg | 46.093364 | 11.604658 | 18.551165 | 1.636205 |
| vidstg → hc2 | 17.919784 | 18.446397 | 14.948784 | 4.527597 |

For HC→Vid, Direct falls34.488707pp while joint falls16.914960pp. For Vid→HC, Direct rises .526613pp while joint falls10.421187pp; joint-minus-Direct worsens10.947800pp. This is an arithmetic baseline accounting, NOT a randomized causal domain-effect decomposition. Comparing source HC with target HC would compare different learned priors. Low target baselines and the zero floor also limit absolute-pp comparisons. The12 source-validation parents are held out for the NEW prior/head, but originate from the original source-domain official TRAIN split. They are not established as unseen to the pretrained Student checkpoint. This source/target comparison is not a fully matched held-out domain experiment.

## Verified problem1: low-noise denoising fails on SOURCE validation

At the original final noise level sigma=.03, start from source GT state plus a paired small perturbation. Both inputs and outputs are decoded with the same representation/scorer. These GT-assisted probes are diagnostic only, not a deployable method.

| Source domain | Near-GT input vIoU | One-step output vIoU | Input tIoU | Output tIoU |
|---|---:|---:|---:|---:|
| hc2 | 90.987791 | 36.476147 | 98.683676 | 71.335256 |
| vidstg | 79.979969 | 24.014100 | 93.441440 | 51.541656 |

All12 cases in EACH source domain worsen in vIoU. Weighted latent loss HC .000725716→.223236255, Vid .000840155→.713459689. A useful low-noise denoiser has not been established even before transfer. The architecture directly predicts absolute clean state without an identity skip/low-noise preconditioning limit. This is not a mathematical proof that x0 prediction is invalid; it is a failed training/supporting-condition test for this finite implementation. At the final reverse step the proposal becomes the raw clean estimate, while the observation-gradient multiplier is .02×.03²=.000018 before clipping. Bad late clean predictions can therefore replace a much better state. No target parameter adaptation was executed.

## Verified problem2: validation and application kernels differ

Training/selection for diffusion uses x_GT+sigma epsilon. Application starts from x_Native+2epsilon and repeatedly visits states produced by the network. The denoising validation computes one fixed noise level for each of12 queries, NOT a full Native-initialized posterior rollout. Both are valid distinct diagnostics; a low value of the former does not establish the latter.

With identical source-validation noise, at sigma2 the HC GT-centered one-step loss is .721547 versus Native-centered1.379901; Vid2.470085 versus15.652176. Source-GT temporal/spatial scales are unstandardized logit coordinates. In the HC-trained direction, mean Native-to-GT temporal state RMS changes .7129→3.6154 and spatial .3439→.8462 from source HC to target Vid; the conditional Gaussian-center KL at sigma2 rises2.4050→18.3471nats. This quantifies a starting-distribution mismatch, not an estimate of the true generative prior KL. For Vid→HC the same center diagnostic19.2782→17.3953 does NOT worsen, so this mismatch alone cannot explain the additional HC target loss.

## Verified problem3: association transfer can invert geometry ranking

Source-HC head: source validation AUROC .86, target-Vid .202797. Mean probability for geometrically compatible/noncompatible references changes .933612/.249254→.502593/.869123. Among14 target references assigned probability>=.9,11 are geometric noninliers. Source-Vid head: source validation .790909→target-HC .721739; compatible-reference mean probability .913015→.568560. These labels are precise-GT IoU>=.5 compatibility, NOT independent visual identity labels. Anchor observations within a query are correlated.39 event-exterior target references remain unknown.

The mixture likelihood can still suppress correct but distant references even though the learned association head has no explicit Source-distance input. In an analytic unnoised-Source-state diagnostic,4/20 compatible HC references and4/11 compatible Vid references have mixture inlier responsibility<.01. H016 has four geometrically compatible references but learned prior probabilities .002602,.054144,.016829,.004152; their Source-state responsibilities are approximately3e-16,1e-25,2e-23,3e-15. These are NOT the actual noisy iteration0 responsibilities. Joint exploration would need to overcome this; the present source-trained prior has not shown that ability.

## Verified problem4: temporal displacement can switch observation feedback off

Vid source validation retains27/41 admitted observations in the inferred event;2/11 nonempty source queries have no remaining event-interior observation. Target HC retains14/56;8/16 nonempty queries have none. Target HC mean inferred duration is .1240 of context while GT duration is .3390. The shared observation function applies a detached hard interval-membership mask. A pure-function contract confirms that if all observations lie outside the event, likelihood and its complete state gradient are exactly0, even with inlier prior .99. This is an observation-feedback blind region, not proof those exterior observations are wrong. Conversely target Vid has no such empty-event query among14 nonempty cases, so this cannot explain all directions.

## Supporting factors and unresolved attribution

Source HC validation context/query length19.86seconds/23.42tokens changes to target Vid25.59seconds/12.94tokens. Source Vid31.39seconds/13.58tokens changes to target HC19.96seconds/23.56tokens. These describe changed conditioning populations, not proof that query length causes failure. Each prior saw only36 training parents. HC joint validation loss best .326865 at epoch16 rises to.461634 by60 while train loss falls .287543→.176974. Vid validation1.238325 at45 rises1.442199 at60 while train loss falls .460615→.248755. More epochs alone is not supported; weights already use source-only best checkpoints.

Original checkpoint routing, all128 frozen Source readouts and integer Native interval codec had passed prior independent checks; no new evidence of a source/target checkpoint swap or coordinate protocol fault was found. Representation-only32-knot Direct changes remain far smaller than the measured loss. Pooled8x8 visual detail, lack of explicit event-exterior identity modeling, posterior sample averaging and the noisy-state rather than clean-state observation approximation remain material but not causally isolated limitations.

## Decision and next discriminating work

Current diagnosis: the prototype has not met basic source denoising/application conditions; genuine association transfer failures and an observation-support blind region add direction-specific problems. This does NOT refute data assimilation or diffusion as a route. Before any new target measurement, prioritize source-only low-noise identity/physical-output retention and Native-initialized rollout validation, and separate source prior competence from association calibration. Noise normalization, network preconditioning and clean-estimate observation guidance are implementation candidates requiring a separately locked test, not changes made in this diagnosis. Do not simply extend epochs or tune target OPD parameters. No repair, new benchmark or method promotion was started.

Primary implementation references: [official SDA score/sampler and clean-estimate DPS guidance](https://github.com/francois-rozet/sda/blob/master/sda/score.py), [EDM design-space paper](https://arxiv.org/abs/2206.00364). Our finite warm-start VE sampler is not a faithful reproduction of their full training/sampling protocols.
