# Integer frame membership representation correction

Independent pre-target-inference audit of all384 source/target hypotheses found
six float32-seconds interval membership masks differed by one boundary frame
from the actual integer frame IDs. All visual/Query features, source/target
trajectories, temporal hypotheses and source labels remain unchanged.

Restore original frame IDs by rounding saved physical timestamp * original FPS;
all256 feature sequences (128 Queries *2 trajectories) round-trip exactly to
their separately stored original frame grid. Compute candidate membership using
integer half-open comparisons. Continuous time-position features still use
seconds. No model/output/GT criterion decides whether to apply this correction.

Original two source critics finish and are retained under recovery, without any
target critic inference/selection/evaluation. Rerun both source critics with the
same fixed seeds/80epochs/data/labels/architecture/optimizer/source validation
selection, then perform first target inference with the corrected representation.
This is not a seed/architecture/target-performance search. No new GPU Source,
DINO or Student fit; no deadline reset. Original runtime stays immutable and
runtime003 explicitly records its single helper and regression-test overrides.
