# Source location diagnostic with matched merger magnitude

Question: does moving identical correct/wrong support before the local reader
produce more directionally useful native effects than the old late mask?
This follows completed free and frozen-output-span source controls, whose very
large equivalent latent changes establish reachability only, not easy learning.

Same original PANEL16 Vid training parents, B1 and official frozen PTD4B, original
observed pixels/native decoding. All parameters/gates frozen, optimizer0, no
new target or external expert. SourceGT explicitly builds the already sealed
correct box-occupancy/time masks and original matched-wrong masks; missing box
support stays neutral. No sample replacement, GT-prefix or best output selection.

Nine arms: original plus event/spatial × early/late × correct/wrong. Early is
post-FiLM/pre-local-reader, downstream event temporal reader/LN/SiLU retained;
late is old post-reader/LN/SiLU/pre-out_proj. This is NOT an experiment on earlier
query pooling or shared stem. Same alpha.25. Per source/branch, compute original
late-correct FP32 merger difference once and lock its L2 norm. Scale each of the
four raw signed differences to that norm using FP64 norm and FP32 arithmetic;
add only to that branch's original merger tokens, retain the real BF16 cast.
No norm/gate search. Target0 means explicit neutral intervention; target>0 with
zero raw delta is an interface failure, not a skipped sample. Realized FP32 norm
must agree within relative1e-3 (absolute1e-10 floor). BF16 norms/counts need not
match and must be reported. Save full raw/scaled deltas plus all sparse postcast
changes, original hashes and all native distributions/outputs before validation.
Late-correct must exactly reproduce old native predictions and baseline all16.

Primary contrasts: event tIoU and spatial sIoU correct-minus-wrong at each
location, then their paired difference across locations. Retain all16 and also
old outcome-blind eligible temporal subset13. Report full t/s/v, endpoint and
coordinate GT-class margins, native time/conditional-coordinate KL, reference/
interval/support changes, native-good retention and >5pp tails. Coordinate KL
is conditional on the1001 classes and must NOT be called full-vocabulary KL.
Spatial-only must preserve base event outputs. Raw norms/normalization/cast
statistics and scalar/tensor geometry receive independent CPU readback. Parent
bootstrap10000/seed20260927 is descriptive, uncorrected, source development.

CPU preflight: real hidden128 identity/branch isolation/exception cleanup,
normalization sign/nonfinite/zero support, early-before-LN vs late-after-LN and
LN common-channel scaling behavior. These do not establish GPU usefulness.
Registration/pins before1800s serial GPU allocation, maximum16GiB new artifacts,
8GiB free disk floor, all actual load/failed/overhead seconds cumulative cap=null.
Original scientific pins/outputs remain untouched. No OPD/target/64 automatically.
A null result at this old small norm cannot prove all earlier evidence useless.
Any later context support test needs the current outcome and a separate protocol.
