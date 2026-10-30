# AVIS round2: audited early-query and audio-activity adaptation

No tested TTA mechanism is adopted. The early query latent has a measurable gradient and changes predictions, but does not improve the new confirmation cohort. An independent audio teacher exposes a real supporting-condition error—relative-to-best-class evidence suppresses weaker cooccurring sounds—but correcting that error still does not make activity adaptation outperform the frozen source.

The official AVISeg archive is now fully downloaded: 10,473,330,966 bytes, SHA256 `fe5ccdffe5756aaf33f44bf107b8b7e5c6359bdfff93ec1dc3f437618d745ec0`. Full gzip CRC/ISIZE verification passed; original sequentially extracted test.json exactly matches the earlier independent tail recovery SHA256. All205 official test videos,12,544 images,205 audio recordings and205 cached VGGish features are available privately. This is acquisition completeness, not205-video model evaluation.

## Protocol and independent confirmation

Round1 eight videos remain historical development and were excluded. Fixed metadata-hash ranking selected8 new development videos (480frames) and12 disjoint confirmation videos (768frames). All official frames were retained, approximately1FPS; confirmation lengths include longer clips, with no silent truncation or OOM fallback. Development supports9/26 categories and none/single/multi-source frame counts4/175/301; confirmation supports14/26 categories and47/257/464. These subsets are not the full benchmark, and their absolute scores should not be compared as if they were controlled domain shifts.

Early6-arm predictions and both7-arm audio-factor predictions were completely sealed and jointly checked before any confirmation outcome scoring. Source and early-latent copies have exactly shared hashes across studies. All subsequent analysis preserves every positive, negative and unchanged video. No parameters/prompts/thresholds were changed after confirmation. Whole JSON had previously been parsed for integrity/schema; this is an outcome-exposure protocol, not a claim that the file had never been loaded.

Primary sound-source measure is independently defined complete frame-instance-set correctness: area-zero masks are absent, category multiplicities must match, and a class-compatible perfect matching must satisfy each strict IoU threshold0.05:0.05:0.95. Report both frame-weighted and video-macro values. This is not nativeFSLA and does not perform global track assignment. NativeHOTA and separately named area-zero detection sensitivity measure tracking; mAP is secondary. NativeFSLA remains a historical compatibility column because its empty-RLE and single-source assignment logic have known limitations.

## Early query latent: real interface, ineffective feedback

The source backbone/pixel decoder is frozen. A zero-initialized100×256 per-video latent is added immediately after frame audio-spatial fusion and before nine local decoder layers. Eight uniform anchor frames receive frozen whole-video class/mask pseudo-supervision. The identity term uses the original transformed frame/clip raw-dot-product embedding location, with detached teacher identities and ambiguous negatives excluded. ThreeAdam steps atLR0.03, official2/5/5 class/BCE/Dice scales, identity0.5 and a source anchor are fixed. No pretrained network weights are updated.

The same-information no-update control directly fuses frozen local/whole-video masks. Old matched-reversal readout and late-LayerNorm entropy arms are retained. Real old-video engineering verified source repeatability, a bit-exact zero-latent/source bridge, nonzero gradients/latent changes, real post-update inference, and bit-exact source restoration after removing the latent. A GT-free audit found all6 used development negative track pairs cooccurred at all8 anchors; no never-cooccurring pseudo-negative repair was needed.

| Early arm | Dev exact-set frame% | Confirmation exact-set frame% | Confirmation HOTA native | Confirmation HOTA area0 | Confirmation mAP |
|---|---:|---:|---:|---:|---:|
| Frozen source |67.510965|46.299342|34.29|34.821955|50.88|
| Direct frozen-head fusion, no update |67.258772|46.580318|34.37|34.918325|50.90|
| Old reversal views, no update |67.510965|46.292489|34.29|34.820108|50.88|
| Old late-LN entropy3 |67.500000|46.292489|34.29|34.820108|50.88|
| Early latent + identity |67.521930|46.285636|34.29|34.821644|50.88|
| Early latent without identity |67.521930|46.285636|34.29|34.821644|50.88|

Development early adaptation gained only one frame×threshold check; confirmation lost two. The identity ablation has the same task results. Direct no-update fusion gains0.2740video-macro pp in confirmation, with a descriptive video-bootstrap95% interval[-0.1023,0.6995],5 improved/2 worsened/5 unchanged videos. This small uncontrolled-subset result is not evidence for a TTA mechanism.

The original development failure decomposition found90 count-mismatch frames and6 category-composition mismatches versus16 additional spatial failures atIoU>0.5; extra category instances128 versus missed18. Count/class mismatch blocked about61.6% of all threshold failures. The shared pseudo-teacher retained the same incorrect activity set, so the latent mostly perturbed boundaries rather than suppressing the main false sources. Fixed audio replacement changed early audio-conditioned queries substantially (mean relativeL2≈0.198), while final instance masks remained very similar (mean matchedIoU≈0.9987). This attenuation is observable; it does not prove that every error is caused by visual bias.

## Independent audio activity: supporting condition and strong controls

One fixed public teacher was used: official LAION-linked `laion/clap-htsat-unfused`, revision8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a,153,492,890parameters,614,525,833-byte weights with verified LFS SHA. Standard Transformers4.52.4 loads the state strictly with `weights_only=True`; no downloaded Python or remote code runs. No teacher/model/prompt search was performed.

Audio uses physically normalized PCMfloat, stereo averaging, rational48kHz resampling and centered3-second windows with declared noncausal context, then official repeat-padding to10seconds. AVISM still uses unchanged author-cached VGGish features under its original frontend condition. Fixed taxonomy prompts include broad person vocalizations; class-level audio evidence cannot tell which of several same-class people is speaking, establish onscreen visibility, or recover objects missing from source proposals.

The late activity latent is class×time, zero initialized with base logit3 so the zero-step output exactly equals source. It controls final per-frame candidate activity directly. TwentyAdam steps atLR0.5 optimize confidence-weighted Bernoulli pseudo-supervision, temporalTV0.1 and source-anchor0.01. Global source class/track scores and segmentation geometry are preserved; inactive frames become area-zero and fully inactive tracks are dropped. All direct and adapted arms use the same policy.

Two fixed evidence definitions form a matched supporting-condition factor:

- Competitive reference: sigmoid((class cosine−max(background cosine,best class cosine−0.04))/0.02).
- Independent reference: sigmoid((class cosine−background cosine)/0.02), changing only the reference. These values are uncalibrated relative evidence, not probabilities of a visible sound source.

Controls include directgate, five-second median-smoothed directgate, a no-temporal objective, and an analytic no-update source-prior gate. At the noTV gate boundaryr=−3, the weightedBCE/source-anchor derivative gives q*=0.5−sqrt(0.03)=0.32679491924311227. This threshold was derived and locked before inspecting teacher scores; it prevents static calibration from being misreported as adaptation information gain.

| Audio arm | Competitive dev exact% | Independent dev exact% | Competitive confirmation exact% | Independent confirmation exact% |
|---|---:|---:|---:|---:|
| Frozen source |67.510965|67.510965|46.299342|46.299342|
| Directgate |41.546053|62.390351|15.549616|44.579221|
| Smoothed directgate |40.668860|61.896930|14.686129|44.592928|
| Activity latent withTV |43.739035|63.125000|17.187500|44.962993|
| Activity latent withoutTV |43.980263|63.366228|16.837993|44.962993|
| Analytic source-prior gate |43.980263|63.366228|16.598136|44.880757|

Development noTV and analytic prediction files are byte-identical in both factors. In confirmation they differ slightly because20-step optimization is a finite approximation to the static objective; that difference supplies no new evidence. IndependentTV and noTV have the same confirmation task results, so there is no demonstrated temporal contribution. IndependentTV remains below source by1.33635frame-weighted pp;0 improved/3 worsened/9 unchanged confirmation videos, video-macro delta−1.32982pp, descriptive95% video-bootstrap interval[-2.75874,0]. Its worst video loses6.9681pp. CompetitiveTV worsens9/12 and improves none, with worst decline82.0175pp.

IndependentTV confirmation HOTAarea0 is34.241313 versus source34.821955; mAP49.29 versus50.88. It is not adopted even though it restores27.7755pp relative to the broken competitive evidence factor.

Development diagnosis isolates the structural condition: competitive evidence rejects129/216 true class-frames in multiclass mixtures, although many class similarities exceed background; independent evidence rejects0/216. Guitar retention changes17/118→118/118 and violin26/50→50/50. The remaining independent errors are concentrated in uncalibrated machine/transport/background distinctions and cannot be declared annotation errors. Correcting a supporting condition is not equivalent to demonstrating a useful TTA method.

## Audits, costs and limits

Original-paper/implementation checks are retained inTRICKS_AUDIT. Current checkout has a similarity-loss naming/weight mismatch and active150/100 debugging schedule instead of paper48k/32k; checkpoint training-code provenance is unknown, so these facts do not establish incorrect checkpoint training. The published PCM24 extraction path reproduces cached VGGish features, whereas physically correcting its amplitude changes the feature domain. The early/late query interfaces, window6 padding and reversal symmetry, prediction quality/confidence readout, CUDA dispatcher modernization and long-video CPU/GPU indexing repair are documented. No random initialization smoke run is counted as task evidence.

Confirmation single instrumented mean costs: source4.511s/video, early latent including source teacher8.862s, direct audio≈5.94s, independent activity≈5.99s. FrozenCLAP adds about1.4s/video and peaks below0.85GB allocatedGPUmemory; AVISM/latent peaks about4–5GB under this sequential configuration. These measurements include source/teacher inference and adaptation as specified, excluding input I/O, model/text initialization andRLE conversion; they are not repeated end-to-end latency benchmarks. Audio controls are diagnostic overhead, not part of the method latency. No OOM or silent input truncation occurred.

The scientific result is a bounded rejection: stable identity and self-consistency do not identify the correct active sound source; an external class-audio score can introduce damaging false absence; and optimizer improvements must be compared with an analytic same-information baseline. A future instance-activity mechanism needs independent, calibrated evidence and an abstention rule. Fine audiovisual synchronization is not automatically available at the released1FPS, and class-level audio alone cannot resolve same-class speaker association. This round does not claim that those missing conditions have been solved.

The public package contains only explicit reference code, sanitized lock metadata, aggregate/anonymous per-video scalar metrics, costs, hashes and provenance. No original videoIDs, media, annotations, per-frame activity labels, cosine arrays, embeddings, prediction masks or pretrainedweights are distributed. Private source data and two qualitative audit figures remain outside the export. Reference source files map back to the original workspace layout; official AVISM/Detectron2 and separately downloaded weights/data are required to run them. No automatic publishing or pushing is performed by this package.
