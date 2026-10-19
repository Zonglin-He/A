# R20 native boundary-conditioned temporal SA: HC128 capability

Autonomous temporal/joint goal, spatial branch fixed, no new expert/checkpoint or
model training. R19 Raw mask teacher transfer was net harmful (Live−Native
−2.587845pp) though Live−freshFixed+1.144081. Close Raw temperature/prior/step
rescue. This new factor changes actual native TimeDecoder temporal SA readable
keys, not a static table slice, ROI prior, query repooling or RGB crop.

Reuse EXACT eight existing event_query128 sampled intervals: four from S0=p0,
four from old real R7 Live41 S1 after40 updates, paired saved uniforms. This
state/sample choice predates R19 and is not GT-picked. No new student update,
no claim S1 was produced by this feedback. All exposed RC128 parents retained;
TC109 never used for current selection. Same Vid Source weights, original33/32
offsets/65 physical times, native initial query/time/CA memory/CA padding/head.

For queried start|end j, every one of six TimeDecoder SA layers only reads keys
whose physical time <= original65 physical end node j. For queried end|start i,
only keys at physical time >= start node i. Original query_mask allfalse verified;
apply this new bool mask in original key_padding_mask interface, no key/value
editing or loss. Given positive p0 support, each queried side must leave at least
one key in both offsets; if not abort/preserve rather than invent a key. Wrong
control retains SAME number of keys but opposite side (suffix instead of
prefix, prefix instead of suffix). All keys allowed ->native output bitexact.
No mask width/layer/strength/search. Do not change native decoder source.

Re-run actual frozen TimeDecoder and endpoint head separately for each of the
16 side queries/parent ×Correct/Wrong ×2 offsets. Exact repeated (direction,
cut) reuse permitted and recorded. 8192 conditioned offset decodes max +256
native exact. Both new start/end logits pass ORIGINAL probabilities/mass_table
(two-offset envelope) before extracting queried conditional on ORIGINAL p0
positive row/column support; normalize once FP64. These constructed labels
are not trained/calibrated native conditional probabilities and may not form
a compatible global joint teacher. No additional prior/temperature/mix.

Native conditional is identical q0 row/column on SAME candidate support.
For each sampled(i,j), start labels compare candidates(i',j), end labels(i,j').
Keep actual candidate coords and all original fullv quality variations, not just
local boundary distance. Fixed counterpart held, deterministic tie convention.
No global new interval/MBR output or OPD method score in this capability round.
Conditional expected GTfullv and t are computed AFTER all128 new logits/labels
are sealed. Average both directions and4samples within each state, 3 frozen
spatial seeds within parent, then128parents. Primary S0 Correct−Native and
Correct−Wrong. S1 matched differences secondary, not a refresher-win claim.
Hard conditional-MBR optional diagnostics are NOT a gate or full-system gain.

Original global query and fused memories already contain full-video context;
this is NOT strict causal visual exclusion. First-layer SA values are repeated
query, differences mainly enter after perframe CA and later SA. Real-frame
masking is out of training distribution; wrong student boundaries can suppress
needed context, relations can need outside events. No advance guarantee.

BeforeGT audit masks by physical scalar predicates/counts/allside nonempty,
input/weights/spatial exact, native full-mask logits exact, blocked keys have
zero attention in all six layers (forward-hook audit without changing return),
conditional labels normalize and target support matches q0. Synthetic mask
identity checks. Qualification independent NumPy posteriorCDF on each unique
condition and direct scalar expectedquality checks after seal.

GPU900s incl load/IO/3s cleanupreserve, prelock each parent/view unique nonempty mask count; allfalse native outputs reused.
First2 noGT max per-unique-decode elapsed (including case IO) times maximum
prelocked query decode count times128 +model loading+60 engineering estimate,
shared nonblocking gpu.lock, GPU free4GiB/host6GiB/disk30GiB/allocator.8 and
5GiB reserve at progress checks, no signals to other services. 0 new Source
encoder/RGB/DINO/Refer/training/spatial/student updates; redecoding cost counted
separately from cached encoders. Persist partials/errors and exact actual costs.

Advance only if primary S0 Correct−Native expected fullv >=2pp, pairedCI>0,
and Correct−Wrong CI>0. This conditional gain is only feedback capacity,
not a final2–3pp system gain. If no such advantage, close this mask configuration
without mask/layer/strength scan. If passes, separately preregister actual
Direct/freshFixed/Live transfer. Parent disjoint confirmation still needed for
method gains. Public all signed/null results and failures, exclude raw tensor/
media/text/IDs/annotations/weights. CURRENT/MAIN/PAPER/spatial unchanged.
