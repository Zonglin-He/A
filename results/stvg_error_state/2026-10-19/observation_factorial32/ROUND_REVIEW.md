# Independent sealed-result review: observation factorial32

Existing GPT-6.1 Sol/max reviewer, read-only code and sealed JSON review; no
GPU, media, GT access/scoring or file edits by reviewer. Root independently
implemented, ran and audited the experiment. This note records the review and
root's resolution, not an independent rerun of all new GT metrics.

Scientific chain passed static review: real RGB crop re-detection with original
text/span, actual crop-coordinate inverse, Source-relative E reconstruction,
dense physical full-span IDs/original positional limit, independent-offset
envelope distribution/finalMBR, all32 seal before GT, factorial S/T/ST invariants.

Main results: S full-v HC−.2194 / Vid−1.6290pp, T−9.2619/−.6673pp,
ST−9.3439/−2.7544pp. HC T and ST intervals exclude0 negatively; HC T13/16
negative and leave-one means remain negative. Vid S is dominated by NV016
−26.1278pp; deleting it yields only a posthoc saved sensitivity, not a new main
result. Vid T8positive/5negative/3neutral is heterogeneous, not universally bad.
GridOnly all32 unchanged isolates decision support only; dense input length,
position indexing, motion physical scale and new RGB remain bundled.

NV016 has two of three known crop-top boxes better than E, yet propagated S
degrades at all three. Do not equate raw top, admitted crop, E-selected candidate
and propagated output. Reference separation is a postseal conditional diagnostic,
not additional deployable method evidence or a new main denominator. The root
subsequently added the full raw-top/actual-reference/propagated diagnostic with
unknowns retained; no prediction was changed.

Required engineering fixes were resolved: owned880s/5s systemd cap within total
900s; pre-execution001 clarified per-cell/total CDF tolerances and GridOnly
attribution. Helper/processor/tokenizer files were supplementally bound AFTER
worker start, not retroactively labelled complete pre-pins; historical matches,
unchanged-before-plan mtimes and actual declared DINO weights were verified.

Both postseal original-input engineering checks now pass bit-exact logits and
Source boxes. They are2 fixed engineering queries after initial scoring, not
all32 pre-run qualification. Review initially saw primary265.218245s GPU only
and required inclusion of22.400693s engineering. Root updated final cost to
287.618938s, retaining original cost privately:32dense queries/64offset forwards,
plus2original engineering queries/4offset forwards and108DINO image calls.

CPU root audit checks128crop geometries,300NMS candidates,32exhaustive E outputs,
160CDF rectangles,256direct MBR utility checks,160old baseline metrics and70
paired endpoints. Do not inflate that scope into independent all32 GPU replay
or a wholly separate GT evaluator for every new metric.

Decision: valid capability screening; no net advantage established and no
2–3pp complete-v gain. Preserve positive cases and severe tails. No OPD fit,
Fixed/Live or joint conditional adaptation was run; ST is factorial composition.
Publish after cost/disclosure closure. Do not auto-tune crop sizes/density,
restart historic queues, promote a method, or claim all observation-conditioned
distillation is impossible.
