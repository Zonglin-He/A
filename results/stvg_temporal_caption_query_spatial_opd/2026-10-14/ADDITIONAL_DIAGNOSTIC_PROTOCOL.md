# Additional diagnostic protocol

Diagnostics are separate from the main five-arm comparison. The teacher cost ledger records 64 method requests. The auxiliary stage is 96 teacher requests: 64 paired-condition visual forced-likelihood requests, 8 reverse/replacement likelihood checks, 12 counterfactual caption calls, and 12 alignment calls. Candidate quality is reported from post-seal metric readouts. Visual-information gain is G=(log P(caption|video) − log P(the same caption|no video))/token count.

The 32 candidate rows pair categorical reward probabilities with candidate tIoU/vIoU and visual-information gain per token. The within-query-centered correlations use 16 candidate rows from four independent sources per dataset. They are descriptive and are not significance tests. Pooled-candidate correlations must not be read as independent-source estimates. In particular, visual-information gain is not evidence that tIoU is higher.

The 12 counterfactual rows compare the original description with a no-video, reversed, or replacement control. Only hashes and scalar reward differences are published; no description text is included.

The eight gradient diagnostics compare the QA gradient and the separate SSL-replay gradient with a same-case endpoint-supervised NLL gradient. This is an offline 512-parameter comparison on frozen hidden states, not the derivative of discrete vIoU. GT gradients were not used for updates. Separately, eight CPU reward-shuffle controls permute rewards and replay the temporal readout; they made no teacher requests or full-model forwards and are not held-out tests.

All trajectories retain per-case, per-step temporal/spatial loss and metric scalars. `null` spatial-loss values are preserved as unavailable measurements. The fixed temporal KL+L2 objective ended above its initial value in seven cases; spatial loss is re-sampled on-policy each step and is not a fixed-target convergence trace.
