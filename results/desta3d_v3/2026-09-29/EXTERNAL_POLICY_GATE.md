# Final external-evidence same-PTD policy gate — completed / independently audited

**Decision: NO-GO. Stop this DESTA-3D privileged-correction / OPD line under the user's bounded final gate.** The internal R16 direction-rescue chain remains stopped. This is a decision about further investment, not a theorem that all external privilege or latent correction is ineffective.

Exact exposed A0.5 Dev16:16 source parents,1 query each, fixed SHA-selected roster. Frozen PTD4B/B1,0 optimizer/backward. One fixed LLaVA-ST-Qwen2-7B supplies both temporal/spatial evidence roles; no independently tested TVG/RVOS pair. Pixel privileges retain original physical coordinate system: outside-time dim.25 and outside-box Gaussian blur8. Source development exposure and possible provider training overlap are explicit. No fresh388 or target input/GT.

|Policy|tIoU %|sIoU %|vIoU %|Δv vs B1 pp|
|---|---:|---:|---:|---:|
|B1|39.517189|36.243945|23.041295|—|
|external|33.271722|21.525494|13.224624|-9.816671|
|T|40.263876|35.789293|20.618530|-2.422765|
|S|35.151169|29.163829|22.151871|-0.889424|
|TS|36.399374|24.266130|19.079401|-3.961893|

External direct is the fixed generated evidence mapped/interpolated onto the observed grid, not an official benchmark reproduction. The policy comparison is the four free-native PTD arms.

TS Δv=-3.961893pp, paired-parent95% CI [-16.383700658846603, 5.097992150415363]. T has Δt +.746687pp, but Δv −2.422765pp. S has Δs −7.080116pp and Δv −.889424pp. The required TS positive v gate fails. TS does not meet the predeclared systematic-collapse definition (>=12/16 negative parents plus <=−1pp branch mean), even though its spatial mean drops11.98pp. CIs are descriptive/unadjusted; no universal or statistically confirmed route-level negative claim.

Wrong controls are conditional on preliminary benefit and were not run. Therefore evidence-correctness superiority over wrong evidence remains untested. OPD, expert/prompt/injection replacement and further rescue tuning were not started.

|Arm|v positive/negative/zero|v harm >5pp|v worst/best pp|B1 v>.5 retained|B1 t>.5 retained|
|---|---:|---:|---:|---:|---:|
|T|4/4/8|2|-65.329688/+20.724095|2/3|5/6|
|S|5/6/5|4|-22.396548/+16.890765|3/3|6/6|
|TS|5/7/4|5|-78.395626/+21.522665|2/3|6/6|

All positive cases are retained: TS best gains+21.522665 and+21.072626pp, while its largest loss is−78.395626pp. All64 native predictions have legal format; legal invalid-zero boxes are retained (B1/T/S/TS totals2/2/0/1). Interval changes T/S/TS=8/8/13; semantic reference changes=1/0/0. Actual pixel changes=12/16/16. All16 provider outputs have usable temporal/spatial evidence after the syntax-only parser repair; this is coverage, not correctness.

Engineering verification and preserved failures:

- smoke001 failed before generation: custom Qwen2 config default vocab151936 versus released checkpoint151660. Isolated v2 uses official AutoConfig; special300 time/height/width tokens use separate official embeddings. First CPU tokenizer equality assertion also retained in repair record.
- smoke002:8 actual calls across2 loaders×2 recipes×2 fixed inputs. Full parameter hashes, processed pixels, vision features and generated IDs exact across loaders. Greedy/official recipes were not identical on both inputs, so the preregistered rule chose official temp.01 seed20260928. Selected first2 raw outputs reused, remaining14 generated once.
- Original parser rejected identical repeated interval mentions and the official prose-header colon. CPU-only v3 recognizes these syntax constructs, preserves all original raw/v2 errors and all box/support values; conflicting spans/invalid boxes remain errors. No new generation or score-based repair.
- One synthetic scorer comparison failed on JSON list versus in-memory tuple; original code/error retained and corrected before native registration. Six CPU controls pass, including full16 synthetic seal/scoring/conditional gate, and public clone tests pass. These are engineering controls, not efficacy.
- All64 native outputs sealed before reading only16 source label records. Sixteen B1 replays match old full geometry/readout/physical fields/native-time logits exactly. Scalar/tensor240 metric checks maximum error2.220446e−16;195 independent aggregate/CI/tail/retention checks maximum3.552714e−15;720 root case/support checks pass.

Measured new allocations/wrappers 373.785564641s, cumulative 73571.557844824s, cap=null; all loading failures counted in receipts. Independent CPU work/initial model-hash validation is distinguished from GPU allocation timing. Actual artifacts 2,063,393,602bytes, free disk 126,620,311,552bytes >8GiB; no research deletion. No live research GPU work remains. Existing CURRENT and old paused queues remain.

Anonymous complete per-query readback:

|Query|B1 v%|T Δv pp|S Δv pp|TS Δv pp|
|---|---:|---:|---:|---:|
|Q01|7.868289|+0.000000|-7.868289|-7.868289|
|Q02|0.000000|+0.000000|+0.000000|+0.000000|
|Q03|35.391910|-0.587895|-8.629413|-1.318403|
|Q04|85.472333|+0.000000|-22.396548|-22.396548|
|Q05|78.395626|-65.329688|+14.876291|-78.395626|
|Q06|37.910062|+0.000000|+16.890765|+16.890765|
|Q07|0.000000|+0.000000|+0.000000|+0.000000|
|Q08|7.386671|+2.589678|+5.484015|+1.866004|
|Q09|37.513171|+4.172051|-3.731064|-10.173620|
|Q10|0.000000|+0.000000|+0.000000|+0.000000|
|Q11|8.798026|-0.217574|-8.798026|-8.798026|
|Q12|61.341130|-5.221245|+0.694827|+21.072626|
|Q13|0.898184|+0.000000|-0.898184|-0.898184|
|Q14|0.000000|+5.106342|+0.000000|+5.106342|
|Q15|7.685312|+20.724095|+0.144849|+21.522665|
|Q16|0.000000|+0.000000|+0.000000|+0.000000|

Reproduction: protocol `protocols/desta3d_v3_external_policy_gate_v1.md` plus loader-v2/parser-v3 engineering addenda. Provider_v2 prepare/launch -> provider audit -> CPU reparse -> native register/launch -> score -> independent summary/case audit. All paths are write-once; never rerun completed stages. Private media, labels, weights, raw tensors and source identities stay local.
