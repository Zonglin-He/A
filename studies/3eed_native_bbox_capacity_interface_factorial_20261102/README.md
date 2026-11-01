# R45 — fixed native bbox capacity/interface factorial

STOP fixed 7B bbox-center primary, no TTA. All A208 development rows, 21 arms,
4368 task IoUs and all 7770 ordered negative comparisons retained. The latter
are repeated comparisons, not independent samples. Primary balanced mIoU
20.551200 vs legacy G 24.727373; secondary rectangle-IoU20.465032 retained.
Primary capacity × interface interaction -.087377pp, secondary -.477953pp.
115 invalid boxes, 20 nulls and 17 bad JSON outputs all remain unchanged;
strict parser uses declared resized-input range, no clipping/reordering/unitswitch.
Coverage center46/rectangle53; old-subset21/24. Immutable24identity18clear6ambig,
center4consistent2contradict12unknown, rectangle5/2/11; no accepted-only accuracy.
No new annotation or GT feedback selection. All 17 cached control arms exact.
208 single greedy calls, preflight2 reused, atomic raw/token persistence;
GPUwall148.796s includesload/CPU/persistence, verification13.429s separate.
The pre-REV1 lock/handoff and the preexecution status wording remain history;
REV1 corrected receipt counters only, before any call. No runtime retries.
The R42 publication retains original download STOP/amendment and 16 file
metadata, linked in SOURCES; no model/tokenizer/config body redistributed here.
Own code/protocol/scalar/flags only. No raw answer/token IDs, masks/coordinates,
private IDs/captions/scene paths, source vectors/features/weights/media or
thirdparty snapshots. Reproduction needs licensed/private assets. A208 exposed
few-scene development only; no FINAL/U/T access, independent confirmation or CI.
This diagnoses a fixed interface pipeline, not all7B models or native grounding.
