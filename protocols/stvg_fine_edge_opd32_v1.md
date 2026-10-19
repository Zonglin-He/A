# Fine-ROI four-edge external distribution OPD32

One new finite user-authorized experiment after Fine-ROI scope32. Other exploration/heartbeat remain paused. Same repeatedly exposed NH16/NV16, original E/finalMBR, layer3 raw maps/valid, retained four DINO anchors, ordered3x3 ROI score, equal anchor score mean and exp(reward/.1). No new model/head training, source feature extraction, backbone/DINO/expert/weight update/GPU. Only geometry state and proposal allocation change; no probability-mixture anchor aggregation or MPO change.

D-FINE uses trained fine-grained localization distributions and localization self-distillation. Only per-edge supervision idea is borrowed; this is not D-FINE reproduction, its trained head or33bins are not imported. Primary source read https://arxiv.org/html/2410.13842v1 .

## Fixed legal geometry and sampling

Each actual node maintains four latent unit-variance Gaussian means mu[L,T,R,B], reset0. Physical edges g(a)=Eedge+.5*Espan*tanh(.5*a), with span Ewidth for L/R and Eheight T/B. Local derivative at0 is original.25*Espan; global each-edge correction strictly bounded by half original span. Ranges for L/R and T/B meet only at Ecenter, so finite nonsaturated actions form legal boxes. Nonfinite latent/physical proposals cause a preserved engineering stop before ROI; finite floating-saturated nonpositive-size samples are excluded by legality, never GT; numerically invalid composed mean is an engineering failure, not a score-based fallback. No image clipping added beyond existing official evaluator. g(0) explicitly returns E bitexact. Final readout g(mu) is not asserted equal to mean of transformed physical distribution.

This smooth bounded parameterization is an explicit added implementation condition, not learned D-FINE uncertainty or unchanged physical exploration: same.25 local scale does not imply global candidate-size/center equivalence to Center. No range search.

Each of2 rounds has17 unique proposals per node: current zero plus4 single-edge perturbations for each edge. Two independent open-uniform draws in first two Gaussian-CDF quartiles, mirror to other two quartiles, inverse standard normal: stratified and antithetic. Seed SHA256 fine-edge32-v1|alias; same draws across three Edge arms. Four edges share zero but each has its own5-state categorical support. This different sample allocation is disclosed when comparing to Center's17 2D candidates.

Each edge teacher q_e=softmax(-eps_e^2/2+ROIscore_e/.1), using original equal-anchor ROI score; no anchor posterior mixing. Student p_mu,e=softmax(-(a_e-mu_e)^2/2) on the same finite latent candidates. Detached q, mean update mu_e -=.5 sum_k(p-q)*a_e, same original per-block step; no temperature or optimizer search. Scope/4planned fixed as previous pure-support experiment. All four means update simultaneously; g(newmu) yields a full box, never separately select four argmax edges. Live second shared-zero candidate scores this composed frame box before any new edge updates. Fixed proposals and visual feedback remain around initialE both rounds by design. Same first step exact. Shuffle original target maps+valid permutation with fixed anchor templates. Final2 only. No-reference/constant/scopeless no-op and all failures retained.

Saved full path permits postseal ROI scoring of both composed boxes:2 extra teacher-score diagnostics per arm/node, not used for acceptance/update or candidate generation, reported separately from34 training candidate evaluations. No reward-based gate or GT rollback.

## Outputs and evaluation

Only new Edge Fixed/Live/Shuffle96episodes192slots; E and prior Fine-Center Live reused exact. All32x5 outputs and actual candidates/edge distributions/gradients/state sealed before NEW GT. Hard selection never gates OPD execution. Full-v, ALL known unobserved actual-s original denominator, tIoU, tail counts and same-parent10000 fixedseed bootstrap. Scoped s only auxiliary. No production promotion/automatic new parent evaluation.

Offline only: exact fixed-E-size free-center geometry ceiling (unclipped analytic intersection; HC official lower-coordinate clipping handled separately), size residual and known-frame coverage; not a deployable method or full-tube oracle. New17candidate per-edge5 supports both rounds evaluated against GT only after seal, good/bad reward/posterior, mass on E-better candidates and candidate best headroom. Moved second-round quality explicitly scored here. Unknown GT remains unknown. Full composed ROI reward diagnostics expose edge interaction but do not guarantee GT quality.

Decision: retain edge route only if Edge_live exceeds E and Center in both domains full-v and actual-s; on-policy claim additionally needs Live above Fixed and Shuffle with uncertainty/tails retained. If Edge harms Center, keep Center and attribute only scoped implementation evidence; oracle headroom without ranking calls for future feedback work, not optimizer strengthening. No simultaneous anchor mixture branch or sweep.

CPU4threads, no CUDA, prediction wallcap900s, MemoryMax20G Swap1G, host/disk guard. Separate finite chain measure→all-outputseal→score; root independent numerical/metric audit, same GPT-6.1 Sol review, verified GitHub dual-ref CAS and full anonymous signed/unknown public export, archive. No new automation or old queues.
