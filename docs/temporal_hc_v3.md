# HC2 temporal readout v3: additional mechanism-led parameters

User explicitly authorized continuing temporal tuning after the v2 fixed validation
failed. That failure is preserved; no candidate is chosen using the v2 validation.
The new official input-only SHA panel excludes every v2 parent and fixes64 tuning
and32 validation parents before any v3 labels. All historical official full outcomes
had already been exposed, so this is still disclosed target development rather than
an independent held-out dataset.

## Parameter map and why these are tested

| Parameter | Actual code role | Failure it can test | Cost |
|---|---|---|---|
| Start temperature | Start logits before native interval normalization | Start uncertainty can differ from end uncertainty | Saved logits only |
| End temperature | End logits before native interval normalization | Late endings can differ from early starting errors | Saved logits only |
| Hull weight | Mix exact original two-offset hull with equal native-offset posterior mixture on the same physical grid | A remote mode from either offset can stretch the hull | Saved logits only |
| Last-layer count | Equal mean over last1/3/6 supervised temporal layers | Earlier depth diversity may help or dilute the final prediction | Saved logits only |
| MBR utility power | Exact finite expected physical IoU raised to a fixed power | A multi-mode posterior can favor a compromise interval under linear utility | No model; exact blocked CPU sum |

Hull and MBR-utility changes are new output-only variants, explicitly different from
the preserved default. The baseline parameters reproduce the old posterior and
interval bitwise on all96 sealed v2 captures. No native logits are selected by GT.
The same spatial trajectory control isolates interval effects from final DP effects.

The initial11 profiles are locked before labels, one factor at a time. Root inspects
all signed changes, duration and boundary errors before at most4 justified refinements
and one combination, within18 total. Tune/validation protection rules are identical
to v2. One profile is frozen before the new32 validation. A passing profile gets a new
HC1902 immutable full plan and cannot resume or overwrite old predictions.

## Fixed boundaries

The original source forward, height224/two offsets, sampling, precision, eight expert
observations, ERef and complete-box spatial propagation/weights remain fixed. Backbone
sampling/query thresholds consume internal action signals and are not tuned here;
training-loss weights are not inference knobs. Changing RGB sampling would also
change expert observations and is outside this output-only study. No448, replay,
training or extra teacher is added. Expert evidence is reused with its original exact
binding; new predictions have a new protocol/code identity.

Old full and v2 evidence stay intact. Public inference defaults do not change until a
new selected profile passes validation. See [locked protocol](../results/temporal_hc_v3/protocol.json)
and [coarse schedule](../results/temporal_hc_v3/coarse_schedule.json).
