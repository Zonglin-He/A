# BoxFixed + class-free RegionKernel temporal feedback

One maintained TA-STVG method: eight observations from one ReferDINO, shared geometric
initialization, complete-box propagation/path selection, Global6 prior and one-round
regional representation prototype feedback. Spatial parameters stay unchanged.

The extra temporal feedback pools original full-caption fused motion memory within
the full-time BoxFixed target context1.5; opposite-offset cosine-kernel prototypes
with common-mean shrinkage16 revise the original posterior once. It requires no
word-classifier output, new expert, new training or RGB re-encoding. The original
Source architecture itself retains its native modules.

Use `python -m stvg predict --request REQUEST.json --output PREDICTION.npz`, or
`python -m stvg evaluate --manifest MANIFEST.jsonl --output OUTPUT`, followed by
independent offline `python -m stvg score`. The only profile is region_kernel.
Private checkpoints/data/runtime are not distributed. See [method](docs/method.md),
[protocol](results/region_kernel_full_v2/protocol.json),
[320-query compatibility](results/region_kernel_full_v2/REGRESSION.json),
[history](docs/history.md) and [archive](docs/archive.md).

The frozen new official full covers HC1902/227parents and Vid10303/732parents,
query-equal. Its full results are pending. Existing Source/E/Global6 are checked
bitwise perquery; newFull and Full-time ablations use the selected new time readout.
Complete domain seals precede independentCPU labels. All signs/severe losses,
parentcluster intervals, spatial mechanisms and costs will be retained.

[Expanded development](results/general_region_v1/ANALYSIS.md) and
[fixed additional-parent recheck](results/general_region_recheck_v3/ANALYSIS.md)
retain all positive/negative outcomes and failed qualifications. User selection
explicitly accepts HCvsK16 +.1639t/+.0676vpp and Vid -1.0688t/-.5542vpp,8timewins/
17harms/4severe vsK161. This is an engineering choice, not a passed validation.
No validated targetlocation-specific time advantage or otherbackbone/MLLM result
is claimed. No further tuning is part of the full evaluation.

Old K16HC320/Vid0, paperHC1902/Vid1373 and original completedfull remain frozen in
separate local archives. The complete pre-general public tree is preserved at
[archive/research-pre-general-20261023](https://github.com/Zonglin-He/A/tree/archive/research-pre-general-20261023),
commitfa2805cecef8fa2a929aea1f5755d3102b33d9b0. Previous pre-K16 and pre-mainline
archive ancestors remain intact. No private video/query/GTcoordinates/weights/raw
mask/features are public. Check source/evidence with `python tools/archive.py check`;
intentional updates use snapshot then check. CPU checks: `python tools/check.py`.
