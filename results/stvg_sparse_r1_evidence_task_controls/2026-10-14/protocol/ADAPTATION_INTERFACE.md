# Adaptation interface record

The finite report records zero eligible conditional-fit cases and zero student fits. The adapter was not executed. This means the adaptation mechanism was not measured; it is not evidence that a trained student failed.

For temporal adaptation, choosing the Native prediction as its own target is a no-op because Native is the original MAP readout. A non-no-op Projection target is required to update Native temporal logits. The locked interface allows at most two eligible cases per dataset/branch. For spatial adaptation, an alternative target is gated by the same-case coarse control. No case passed into an actual fit in this run, so neither interface received an adaptation test.
