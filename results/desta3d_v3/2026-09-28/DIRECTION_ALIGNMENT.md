# Direction alignment: mixed branch diagnosis, no automatic prototype run

Status: completed / independently reduced on CPU. Two previously outcome-selected Vid training parents; B1 frozen PTD4B. No new forward, backward, native prediction, optimizer, label pool or target input.

The initial full-merger gradient was available in the preserved free-control STEP_01, before clipping. It was not inferred from the 128-D span gradient. The gradient-coordinate relation g_span = sqrt(2560/128) g_F Q has relative L2 errors 2.4963e-7 and 3.0254e-7. Initial native reference, endpoint/coordinate distributions, geometry, pixels, preprocessing, query, grid, physical time and objective targets/support agree across the five source runs. Temporal uses 2 endpoint actions over 32 observed times; spatial uses 44 coordinate actions over the complete 152775 vocabulary.

Temporal P07: the fixed optimized span direction has cos(-g0,delta)=+.0615735 and descent dot +101.568719. Large correct mask is -.00217663 / -3.590462; context correct is -.00207962 / -3.430436. Their cosines to the successful span endpoint are only .00352658/.00287864. This supports an adverse *initial local CE* direction for these two late temporal interventions on this source. Early correct is slightly favorable (+.00035848), so the result does not say every scalar intervention has an adverse direction.

Spatial P03: span is +.0850428 / +3.547041. Large correct and context correct are both favorable locally: +.00676063 / +.281978 and +.00764537 / +.318880. Cosines to span are .01649335/.01450208. They are weakly aligned, not sign-reversed. Their saved native sIoU gains are +4.839735pp and +4.446244pp, versus span +14.179430pp. Context wrong gains +5.144507pp despite a smaller favorable gradient dot. Keep this counterevidence: local descent alone does not guarantee correct-over-wrong evidence utility.

Decision: stop further scalar-mask tuning as already agreed, but do NOT conclude a common proven 'wrong direction' root cause for both branches. The conditional strong sign diagnosis is met for temporal only, not spatial; the predeclared both-branch criterion is not met. No prototype field, MLP, reader training, external/pixel qualification or OPD GPU stage was registered or launched. The prototype remains a candidate whose feasibility would need its own qualified protocol; it cannot be labelled a validated root-cause repair. Keep shared THW/dual-reader and preserve the distinction between reachability, evidence direction, reader learnability and distillability.

A 30-step endpoint is not an initial steepest-descent vector: state/support changes and nonlinear/discrete BF16 computation remain relevant. First-order dot magnitudes are not finite CE predictions and must not be compared as calibrated utility across the two objectives. Two selected successful controls do not establish population alignment or native latent sufficiency. The tested masks' failures do not eliminate every possible support/operator combination; scalar STOP is the resource decision.

Implementation: four CPU synthetic controls passed. Audit v1 failed before any direction statistics because whole-dictionary comparison rejected three later metadata fields absent in early records; actual endpoint logits and other fields were exact. v1 script/lock/failure retained. Isolated v2 compares exact shared values and validates extra metadata separately. 54 consumed sealed files /87 pinned files; 176 independent NumPy/PyTorch reductions passed atol1e-9+rtol1e-10 (maximum absolute error1.013e-8 on norm-scale quantities, not an all-values absolute1e-9 bound). v2 measured CPU section4.726919s, prior failed section1.157492s; new GPU0, cumulative42493.15652965409s cap=null. No large raw copy, deletion, checkpoint or production change.

# Saved direction alignment audit

CPU-only readback of two preselected source cases. No new model execution or predictions.

|Case|Direction|cos(delta, span30)|cos(-g0, delta)|-g0 dot delta|matched-span-norm dot|
|---|---|---:|---:|---:|---:|
|P07 event|span_step30|1.00000000|0.06157352|101.568719|101.568719|
|P07 event|free_step30|0.18941880|0.15577937|229.778399|256.96618|
|P07 event|event_late_correct|0.00352658|-0.00217663|-0.0046994784|-3.59046247|
|P07 event|event_late_wrong|0.00055502|-0.00202603|-0.00437431648|-3.34203447|
|P07 event|event_early_correct|0.01190645|0.00035848|0.000773973094|0.591325499|
|P07 event|event_early_wrong|0.00817967|-0.00135202|-0.00291909303|-2.23022497|
|P07 event|event_large_correct|0.00352658|-0.00217663|-3.59046234|-3.59046249|
|P07 event|event_large_wrong|0.00055502|-0.00202603|-3.34203427|-3.34203449|
|P07 event|event_context_correct|0.00287864|-0.00207962|-3.43043618|-3.4304364|
|P07 event|event_context_wrong|-0.00012299|-0.00201129|-3.31772922|-3.31772938|
|P03 spatial|span_step30|1.00000000|0.08504283|3.54704097|3.54704097|
|P03 spatial|free_step30|0.18214141|0.23233506|6.84114022|9.69043437|
|P03 spatial|spatial_late_correct|0.01649335|0.00676063|0.000499978049|0.281978324|
|P03 spatial|spatial_late_wrong|0.01813368|0.00533872|0.000394821775|0.222672139|
|P03 spatial|spatial_early_correct|0.00273066|-0.00002432|-1.79874799e-06|-0.00101446044|
|P03 spatial|spatial_early_wrong|0.00608063|0.00243717|0.000180238947|0.101651421|
|P03 spatial|spatial_large_correct|0.01649335|0.00676063|0.281978405|0.281978325|
|P03 spatial|spatial_large_wrong|0.01813368|0.00533872|0.222672203|0.222672139|
|P03 spatial|spatial_context_correct|0.01450208|0.00764537|0.318879908|0.318879811|
|P03 spatial|spatial_context_wrong|0.01622099|0.00568963|0.237308186|0.237308128|

Positive last two columns predict local CE descent; they are not measured CE/native improvements.

Both-branch strong sign gate: False.

Two previously outcome-selected, source-training exposed cases; no population inference.
g is the initial B1 native CE gradient, not a gradient at the optimized endpoint.
30-step endpoint was reached along changing native states; initial Taylor dot is not its finite utility.
BF16 autograd is not an exact derivative of the discrete numerical program.
Low endpoint cosine alone does not establish adverse gradient direction or a root cause.
No claim of actual-reader learnability, teacher qualification or OPD success.
