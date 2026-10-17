# R1 semantic preference, four anchors on fixed exposed32

Latest human instruction: semantic preference can use4 anchors, align Direct.
Question: can the previous one-anchor shortfall be explained by sparse observation,
and does semantic weighting improve over geometric weighting on the SAME support?
Prior one-anchor PREF vIoU HC27.242495/Vid12.507167%; revised Direct four-anchor
28.580257/14.572580%. These end-to-end settings were not a feedback-only contrast.
No claim of semantic improvement is presupposed; no tuning/promotion follows.

HC16/Vid16 exact original exposed32 parents; original T2/Source, all4 prelocked
anchor positions, original image/context4/query are checked EXACT against revised
Direct input lock. TA-STVG same cross-domain checkpoints, frozen R1 nativeBF16,
existing four E1 candidate boxes per anchor, same original preference Prompt and
forward/reverse numbering. Old slot0 outputs32/64 forwards reused byte-for-byte,
new slots1/2/3:96 anchors/192 real forwards, total256 vs Direct128 generations.
No new Temporal/Caption/DINO, free generation, candidate geometry, Prompt or
hyperparameter search. Remaining Direct differences are explicitly reported:
continuous direct boxes vs candidate pool, on-policy32 actions vs fixed support,
and two likelihood orders vs one direct coordinate generation.

Per-anchor GT-free admission: semantic argmax0 abstains; invalid open-logit
candidate or Source anchor rejects that ANCHOR without clipping; all other legal
anchors remain. Query with no admitted anchor remains Source/no-fit. This is the
natural multi-anchor extension of the old single-anchor rule and matches Direct's
partial-reference handling, but its chart restriction is a separate limitation.
All32 stay in denominators; same anchors/ref/admission for GEOM/PREF/D2. Positions
are sorted once and audited at rollout/gradient reconstruction, not averaged into
one frame. GEOM=(1-p0)*softmax(IoU(candidate,common_argmax_reference)/tau), PREF=p_i.
Each anchor retains mass1-p0; loss averages over actual admitted anchors exactly
as original fitter. Stop gradient on actions and Teacher weights. No pseudo-label
for discarded/unknown anchors. Same-information D2 uses same sparse hard refs.

Original1792 episodic Query+LN, original domain lr/sigma/tau/steps, Adam/full-grid
response, one fullstep/ref-IoU finite acceptance, full state rollback/reject-stop,
complete frame decoder readout and exact Source/time invariants. No weakening of
acceptance, precision or scientific audit. Independent multi-anchor gradient and
FD test; one-anchor policy must match old implementation exactly. Root retains
validated gamma8 nativefloat32 moment roundoff audit and strict parameter delta.
At most64 new fits and192 new R1 forwards,2h scientific serial ceiling; live
16/24GiB RAM,Swap1GiB,host6GiB/GPU4GiB/disk30GiB floor. Pause only verified
waiting-download Qwen CPU controller during R1 work; never signal downloader or
another model worker; finally/ExecStopPost restores exact CPU identity.

Both GEOM/PREF32 ROOT and complete Student/Global barriers and same-refD2/root
must seal BEFORE GT evaluation. Reuse only the previously exposed target labels
for postseal reporting. Report all32, per-domain vIoU/dense same-time sIoU/tIoU,
R@.3/.5, actual accepted/rejected/no-fit/anchors/cost, all paired signed/unknown
sources and uncertainty. Primary mechanism contrast PREF-GEOM; end-to-end Direct
reference and one-anchor extension contrast reported separately with confounds.
Retain negative tails; no GT routing, independent-confirmation claim or automatic
CURRENT/PAPER promotion. Real pixel review, anonymous privacy, fresh two-ref CAS,
remote HTTP/SHA/blob/mode/tree and archive check/snapshot/check before closure.
Qwen original Direct experiment remains separately authorized and resumes.
