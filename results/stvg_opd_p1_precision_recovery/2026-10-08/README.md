# Fixed P1: softmax audit precision supplement 002

This is a completed engineering recovery and an actual running-P1 launch record,
not a completed P1 experiment or evidence about grounding efficacy. Both-direction
P1 deployment predictions remain unsealed; no P1 GT was read during this recovery.

The independent audit failed after 3,630 receipted HC2 predictions (4,524,043,386
opaque bytes). Their prediction and receipt SHA256 hashes were verified and
preserved. Original code, scientific locks, and the original failure remain intact.
The scalar CUDA division implementation in the installed PyTorch 2.7.0 git revision
uses a float32 reciprocal and multiplication. A direct float64 reference has different
intermediate rounding. See the linked primary source in TORCH_SOURCE_REFERENCE.json
and protocols/stvg_opd_p1_softmax_precision_revision002.md.

The original float64/GPU32 probability discrepancy was 3.477426682718665e-7 and
failed the original 3e-7 absolute check. The supplement retains that failed result,
the unchanged 3e-7 matched-operation threshold, and the original gradient/Adam/state
checks. Matched-operation CPU32 and independent float64 normalization of rounded
CUDA logits differed by 2.9802322387695312e-8 and 1.968405555219377e-8, respectively.
An explicit gamma3 logit-rounding bound and componentwise softmax sensitivity bound
certify the cross-precision discrepancy. No fitting or scientific parameter changed.

Validation includes 16 synthetic CPU regimes, 16 deliberately wrong normalized
probability distributions rejected, all 40 rounds of the reproduced failed fit,
12 already-closed GPU qualification fits audited on CPU (not 12 new GPU fits),
and 9 historical prediction receipts whose saved audit dictionaries were reproduced
exactly. One actual GPU qualification replay matched the serialized first reproduction:
948 tensors, 270,844 coordinates, and 1,186 scalars. The first formal missing arrival
was repeated and matched again before acceptance. The old worker did not serialize
its original failing fit; bitwise equality is explicitly between deterministic actual
replays from the saved full prefix, not against lost dead-process memory.

HC2 remains lr=.01, sigma=.025, tau=.05, steps=40, LN writeback=1/16, M=32;
VidSTG remains .03/.1/.25/10/1/8/M32. The full inherited LN prefix is restored by
the unchanged runner; query residual and Adam reset per query. The finite controller
continues HC2 then VidSTG and only permits postseal GT after all 41,355 P1 arrivals.
The CPU bridge dispatches by the saved original/revision001/revision002 audit metadata
and does not rewrite older predictions, locks, or receipts. P1 postseal root audits,
views, all anonymous positive/negative results, and P2-P6 execution remain pending.
EATA and its media/Fisher preparation remain explicitly user paused.

CPU-only public contracts (PyTorch and NumPy required):

```bash
python -B scripts/test_stvg_opd_softmax_precision_revision002.py
```

The private failing fit, actions, weights, gradients, optimizer states, predictions,
source checkpoints, captions, GT, and media are excluded. Private replay requires
the separately obtained original research inputs. The requalification fit measured
1.508919641s of GPU fit wall time; the first reproduction was not separately timed,
so this does not claim total recovery cost or normal P1 throughput.
