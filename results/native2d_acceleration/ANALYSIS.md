# Exact appearance-prefix reuse for full re-evaluation

Native2D requires the original pre-fusion appearance grid. When same-input/source/evidence-bound native all6logits and complete geometry already exist, the full re-evaluation reuses those immutable inputs and calls only the original vis_encoder+input_proj on EXACT original224 two-offset batches/FP16autocast. New geometry still requires full originalSource capture. No model/precision/parameter/frame/expert changes.

Four fixed TA queries across both directions compare actual fullSource observer grids/masks/dtypes and pooledfeatures to the prefix, including cold and visualcache-hit calls: all bitwise. PREFIX_PROBE.json records everytiming. Example resident Source part2.9–3.0s versus prefixcold.5–.8s; decode/modelload excluded. These are matched subpath timings, NOT end-to-end deployment speedup or four-query universalbenchmark.

Native2D fresh deployment still needs Source for its prior/spatial evidence; its appearance is captured during that ONE originalSource forward. This acceleration saves repeated computation when reevaluating old captures, not an assertion that nativeSource can be removed on unseenqueries.

Original fullv1 safelypausedHC69, every prediction/capture migrated byteforbyte to independent v2 with fresh receipts and explicitsameprofilecompatibility. Original69rawSource/query cost includedONCE as a previoussame-method segment; newmigration/currentcachecost not double-counted. Fused401 remains paused. SOURCE code and geometry/math unchanged.

Existing256-cache posterior/time/box regression is explicitly REUSED through input-byte and unchanged-math SHA compatibility; no secondregression execution is falselyclaimed. New acquisitionprefix checkedby actualGPUprobe. Originalfullv1 source snapshot/plan/logs remain privatefrozen, original256mathproof andno-opbridge public.

Currentfull: native2d_full_v2 fp8f56292b25008519f4255de0b9465740036c06af274effadc172b417d4bc58e6. Targeteachdirection tIoU>=Source+.9pp; unknownuntilfullscores, no post-fulltuning.
