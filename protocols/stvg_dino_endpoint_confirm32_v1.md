# Frozen teacher-endpoint candidate: new-parent center versus24-state retest

Freeze9862bd4a's local endpoint target before constructing a new HC16/Vid16
cohort. Exclude parent videos in original OPD tuning32/domain, the original
rollback/error-state32, query-only64, external20-64, soft-correspondence32,
spatial-readout32, controls256 and E Source-expanded256. Their plan metadata,
not GT/scores/DINO quality, determine exclusions. From remaining official parent
IDs, select16/domain by ascending SHA256 of a fixed salt and seed20261018.
Choose one query per selected parent by an independent fixed hash rule.
Seal membership before cache inventory. Never replace a missing, invalid,
unsupported or failed source based on quality. These official videos have
historical full-result exposure; this is not a globally unseen test.

Use the same frozen TA-STVG cross-domain checkpoints and original cached DINO
four-frame inputs. Restore only missing Source temporal logits by frozen
forward; require bit-exact Source boxes/native interval and original input
binding. No historical online replay, QR/LN updates or training. E uses the same
source calibration, beta1/16, candidates, admission and global correction.
Time is the same Source final-physical-distribution MBR in every arm.

On the original four nodes use the frozen aligned conditional-DINO interface:
content is the first maximum native target-word score slot; attention exclusion
is that same content slot. Original900 queries unchanged, added queries isolated,
threshold0.35. Sample24 shared4 states once around E using the existing source
scales and unit Gaussian; any illegal coordinate at any original anchor makes
that complete sample invalid. No clamping or replacement sampling. The exact E
reference is the center query, governed by the same whole-state legality rule.

Four outputs, all other decoder nodes exact E:

* E_MBR: no new update.
* Center_full_local: use T(E) at accepted center nodes, otherwise E.
* Samples24_endpoint: z(E)+sum24 m*(z(T)-z(E))/24, unchanged frozen target.
* Center_p_local: z(E)+p_i*(z(T(E))-z(E)) if center accepted, otherwise E,
  p_i=n_i/24 from the very same24-state admission cache.

The last control uses sampling information and is not a cheap deployment
baseline. Equal p is not equal realized displacement norm. Save admission-only
common-node mask (accepted center and positive24-state count) before GT; report
quality there separately. Missing GT there is unknown. No GT affects admission,
sample generation, support, tuning or output selection.

One native image/text encoding per active anchor is shared across groups;
center and sampled decoder passes are timed separately. Report actual encoding,
decoder time, logical reference queries, unique queries, original cached inputs,
Source forwards and full new wall/GPU budget. Shared/cached timing is not a
first-use end-to-end claim. Original supported/no-op queries remain in all means.

Budget: at most32 new frozen Source captures and128 original-anchor encodings;
center1 plus24 sampled references per active anchor, one round only. Native
replay/clone qualifications on the first fixed active query/domain,8 anchors.
Total new GPU allocation cap900s (Source plus conditional stages); wall cap2400s
including lease waits. Shared lease, MemoryHigh16GiB/Max20GiB/SwapMax1GiB,
disk floor30GiB, host available6GiB, GPU free4GiB. No deadline reset.

Seal all32 complete four-arm outputs before any new GT scoring. Primary complete
vIoU; additionally observed/unobserved actual decoder sIoU, common-admitted-node
sIoU, benchmark interpolation and exact additive vIoU contributions, negative
tails and displacement. All tIoUs must be identical. Bootstrap10000 times at
parent/query level, seed20261018; no independent-sample claim for candidate pairs.
Report Samples24-E, Samples24-Center_full, Samples24-Center_p and Center_p-E,
including both domains separately and all signed/no-op cases.

If24-state endpoints exceed E and p-scaled center, that is evidence to consider
a later dynamic test, not authorization to auto-start it. If only full center
is worse, distinguish conservative p scaling from endpoint information. Similar
center performance weakens the case for extra samples. Failed replication stays
a development finding. No formula variant, threshold/step-size search, new
backbone, temporal change or dynamic OPD in this run. Current registration and
all historical predictions remain unchanged. Publish measured positive and
negative findings and verify remote bytes before closing.
