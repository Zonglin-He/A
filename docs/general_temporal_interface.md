# Class-free temporal feedback interface

This research replaces an additional closed-vocabulary word-class projection with
target-conditioned content comparison. The common algorithm consumes a temporal
prior, physical timestamps and a sequence of within-model representations. It never
compares coordinates between different backbones or treats hidden channels as word
probabilities. The deployed K16 profile remains frozen and paused pending a separate
decision; research does not silently replace it.

For each sampled time, pool the original full-description multimodal feature grid
on the existing full-time BoxFixed trajectory and fixed1.5 context. Fit centering on
the other sampling offset, normalize vectors, and form a nonnegative cosine kernel.
Original temporal soft membership defines positive/negative weighted matching means;
both shrink to the same fitting-fold unconditional matching mean. One compatibility
log-ratio update corrects the original interval prior. Original physical-time MBR
and spatial propagation/path selection remain. This contains no additional trained
head, class vocabulary, external semantic encoder or added expert observation.

The TA adapter reads the actual native multimodal memory's visual suffix before its
time decoder. The Source itself, including its existing learned modules, is unchanged.
Other grounding backbones require verified frame/token/position/padding adapters and
their own native temporal prior. A suitable MLLM requires accessible visual tokens
and a readout actually conditioned on the full query; visual positions before later
text in a causal sequence are not automatically query-conditioned. A black-box model
returning only an interval cannot supply this feature interface. No CG-STVG,
TubeDETR or MLLM inference is performed or validated in this study.

The measured distinction is between a general implementation interface and improved
event feedback. Correct target support must beat whole-grid and matched-location
controls before claiming spatial-to-temporal causal value. Time alignment, native
posterior sharpening and exact-strength old feedback are separately controlled.
Representation consistency, stronger prose and positive development means alone
do not establish correct event boundaries or cross-backbone transfer.

See [fixed development results](../results/general_region_v1/ANALYSIS.md). All
signed outcomes, harm tails, complete-v/time differences and failed gates are retained.
