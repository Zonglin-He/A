# Frozen-profile CG-STVG full evaluation — HC complete, Vid running

The user explicitly requested the complete HC1,902-query and Vid10,303-query evaluation against Source. The user separately confirmed that Vid uses its prior domain-specific parameters; it does not inherit the HC conservative profile. This is a new independent full-evaluation namespace, not a resume or replacement of any prior partial/full method or parameter search.

| Domain | Queries / parent sources | gamma | unobserved share | box-IoU link | beta | kappa | context |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC | 1,902 / 227 | 0 | 0 | 0 | .5 | 64 | 1.5 |
| Vid | 10,303 / 732 | .5 | .5 | .25 | 1 | 64 | 1 |

HC explicitly retains Source on unobserved frames and disables adjacent-box path coupling; independent observed-frame decisions and temporal feedback remain. Vid keeps the original both-anchor, bidirectional complete-box geometry and exactDP. Every selected profile is frozen before full labels; this run cannot select new parameters. Production TA/eight/main config, every old result/root and Fig1 stay unchanged; third benchmark is deferred.

Original CG420 FP32, source-checkpoint EMA (HC target uses Vid weights; Vid target HC weights), official single native test grid and native postprocessing are unchanged. The prior audited independent-attention-window storage schedule is used; no video-frame or expert joint-context chunking, precision/resolution reduction, optimizer or new teacher. All eight SAME ReferDINO observations are selected from CG's nativeMAP interval. SourceNative uses original nativeMAP. The frozen internal key Source contains OUR added MBR on the SAME CG boxes and must be displayed as Source+MBR, while SourceNative is displayed as original Source. All original-Source, MBR-only and matched-MBR feedback comparisons are reported separately.

Both official sets have historical TA exposure. All227HC parents participated in prior CG development/CV (one query each); Vid64development queries selected its profile. This full evaluation is not an unseen independent confirmation. Public reports will include full query-equal means plus selection-query and remaining-query strata; remaining HC queries share development parents and cannot be called independent parent validation. No sample or failure is dropped.

Strictly compatible legacy CG acquisition covers227HC+320Vid queries. The other1,675HC+9,983Vid queries need actual native Source and same-expert8 work: at most11,658 new Source queries and11,658 new joint expert calls plus4extra native no-op reference forwards (first2new acquisitions/domain). A cache hit retains the old original acquisition cost/provenance; it is not zero-cost deployment. Every legacy capture/input/native grid/evidence context is SHA-bound and its new profile is genuinely recomputed and repooled. TA grids/logits/oldpooled features do not substitute for CG.

Independent preflight passed12real previously acquired cases (6/domain, two from each historical source family). HC outputs match the frozen100-profile selected arrays; Vid outputs match prior selected-feedback math and available confirmation arrays. All compact-to-cold replay outputs match exactly; preflight performs0newSource/teacher/model/GT work. Prior actual checkpoint bridges are referenced separately, not misrepresented as CPU model tests. First2new full acquisitions/domain execute original fake-label-unused-call versus observer/no-label/native-head no-op again.

Every domain must completely seal before separate CPU fixed-label scoring. HC CPU scoring may overlap Vid GPU work. Five arms: SourceNative, SourceMBR(named Source), SharedCorrection, Space and Full. Report full t/v/s means, all signed positive/negative changes, severe t/v/union (>5pp loss, union once), rescues, short/long event strata and parentcluster pointwise CIs. No new Source+.9 threshold or final-test parameter choice is introduced. HC results will be notified when valid, without waiting for Vid completion or final publication.

## Durable output and bounded storage

A full raw CG capture is about66–71MB/query, so duplicating raw features for12,205 queries would exceed local available storage. This new study checkpoints each paid raw Source before requesting the expert, and keeps the raw scratch until prediction commit. It saves compact geometry, auxiliary trajectory, actual repooled FP64 features, native FP32 logits/q0, Source, final predictions, teacher context/cache and acquisition receipts. Every compact state is read cold and all five arms replayed exactly before commit. Only after the new prediction/compact/receipt hashes are committed is that query's NEW raw scratch disposed with its original SHA/disposal receipt. All historical raw captures remain untouched. This sufficient-state cache reproduces this frozen profile; it does not claim to preserve arbitrary future raw-grid repooling.

Private output cap160GiB, minimum disk free80GiB, per-worker limit72hours. Each process/PID-start segment records real Source forwards, expert forwards, extra native references, load times and nested pipeline/feedback costs. Failed/unfinished paid Source/evidence remains available for inspection; no blind retry, skipped query, hidden fallback or hash waiver. A same-plan resume must verify owned process exit; an unresolved before-Source attempt without checkpoint blocks automatic repeat. New interface fixes require independent version/evidence; frozen code never patched in place.

Public code/protocol and all scalar signed results; videos/captions/GTcoordinates/raw features/masks/checkpoints/credentials private. Existing half-hour heartbeat watches actual process identity/lock/progress/log/resource state, quiet during healthy unchanged running. Complete per-domain results are published promptly; final closure requires every-domain score/report plus all-remote tree/file/mode/byteSHA/ancestor verification.

Entry: `studies/cgstvg_full_v1/run.py run`; never launch a duplicate. Fingerprint: `bee008c175fc359dc89947e1896882a0d9757bf83d1701403e42c93d3c63663d`. Private state/progress/launch/seals under `outputs/cgstvg_full_v1`.

## Explicit source display names (2026-10-24 correction)

Frozen data keySourceNative is displayedas **Source(originalnativeMAP)**; data keySource as **Source+MBR**. The originalmatched-readoutcontrast remainsvalid but cannotreplace nativeSource. Aftereachdomainseal+officialscore, root must also run `studies/source_semantics_v1/cg_full_reporting.py --dataset hc2` orvidstg; it only readsalready-scored metrics and providesnativeSource/Source+MBR/Full means,allpairedCI,signed andsevereagainstnativeSource. Notifyboth Full−Sourceoriginal andFull−Source+MBR. No inference/profile/hashchanges. See [semantic correction](../source_semantics_v1/README.md).


## Completed HC full evaluation

[HC1,902 complete result and audit](hc2/ANALYSIS.md): original Source t23.220480/v14.673932%, Source+MBR42.632982/26.153691%, Full44.899144/27.331087%. Matched-MBR gain +2.266163t/+1.177396v pp; original-Source total gain includes the much larger MBR readout effect. All signed/tails/CI/cost and exposed development-parent limitations are retained. Vid remains running; the deferred TA strict-deletion study has not started.
