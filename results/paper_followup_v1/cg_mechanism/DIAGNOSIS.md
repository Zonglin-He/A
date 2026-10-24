# CG failure diagnosis and parameter roles

Scope: previous exposed CG development64parents/domain ONLY. The new35HC+128Vid confirmation labels were not read. `cg_mechanism.py` reconstructs all128 existing original Space paths byteexact from saved candidate geometry, then computes offline GT diagnostics. No new model/expert or new parameter configurations. All candidate/quality statistics are offline evidence, never an inference gate.

## The first damaged component is shared correction

Mean fixed-support unobserved box IoU (%) on the SAME sampled GT-supported frames:

|Domain|Source|SharedCorrection|Space|Full|
|---|---:|---:|---:|---:|
|HC|57.212527|48.308292|51.040084|50.464816|
|Vid|47.392838|44.437702|45.608398|45.970077|

Shared correction alone loses8.904236ppHC/2.955135ppVid. Spatial decoding restores2.731792/1.170695pp relative to that damaged initialization, but does not recover Source quality. These are fixed-support spatial statistics, so merely changing which frames are scored cannot explain them. It is wrong to assign the whole spatial loss to propagation or to temporal decoding.

The original8-anchor consensus estimates a common encoded shift and applies it through the entire track. A common offset can overcorrect a strong Source or misrepresent time-varying box errors; sparse erroneous anchors can also contaminate it. The results establish damage at this interface, not which of these subcauses dominates. Expert GT-supported anchor meanIoU is55.804077vsSource56.798167%HC (58queries with annotated anchors),57.065352vs48.190723%Vid (51queries); thus the evidence is not uniformly bad, and better sparse anchor quality does not establish safe full-video common correction. Unannotated anchors are missing evidence here, not scored as failures.

Gamma is the directly relevant parameter. Atgamma0 SharedCorrection=Source exactly; HC Space unobservedIoU54.8336%, versus51.0401 atoldgamma.5. Gamma0 still retains all8 expert boxes/masks in propagation and temporal support; it disables only global correction. Vidgamma.125 improves fixed spatial quality most among these values, butgamma.5 gives higher jointvIoU after temporal decoding. One spatial metric alone cannot select the final joint configuration. The CGDevRobust residual-scale/truncation profiles did not win; no evidence supports declaring scale calibration the sole problem.

## Candidate selection has additional headroom

Within original nativeMBR, unobserved GT-supported frames, per-query means (HC59 / Vid40queries with valid nodes):

|Domain|SourceIoU|EIoU|selectedSpaceIoU|best legal candidate per-frame IoU|candidate–selected gap|
|---|---:|---:|---:|---:|---:|
|HC|55.934309|47.152093|48.677852|60.928626|12.250774|
|Vid|52.433765|49.690801|52.437311|56.368262|3.930950|

The best-candidate value is a framewise GT diagnostic upper bound, not a realizable single trajectory or method score. Existing candidates sometimes contain better localization; lack of candidates alone cannot explain every loss. Within each query, appearance-unary/actual-candidate-IoU Spearman then averaged over definedqueries is.061474HC(58valid)/.067334Vid(36valid), descriptive only. This is weak alignment in the tested candidate pool, not proof that CG features in general are unusable. Exact DP is already solving its chosen objective exactly; making the same DP more exact cannot repair a misaligned score. Connection weight can change the objective but cannot supply missing object evidence or repair an incorrect affine fit.

## What the finite parameter controls actually show

Stage B holds the chosen A profile fixed and changes one factor:

- Link.25→.5 improvesHC developmentvIoU23.464375→23.691393, but harmsVid15.986841→15.524782. Removing links harms both (22.268480HC/15.336001Vid). Smoothness helps in these controls, but larger is not universally better.
- Beta changes the strength of visual temporal evidence relative to nativeq0, not the spatial affinity. HCbeta.25 yields23.696096vs23.464375 atbeta1; Vidbeta.25 sharply harms15.986841→13.784174. HCbeta.25 is only.004703pp above the link.5 alternative; that tiny development difference is not mechanism proof.
- Context only changes appearance pooling for temporal prototypes. Vidcontext1 improves15.986841→16.379100; context2 gives15.835019. Background contamination is a possible explanation, not established causality. This does not directly repair spatial affinity or geometric candidates.
- Kappa controls prototype shrinkage. Vidkappa256 drops to14.354809, whilekappa16 gives16.106159. More shrinkage is not automatically safer and cannot correct biasedE.

Frozen selected profiles: HCgamma0/link.25/context1.5/kappa64/beta.25; Vidgamma.5/link.25/context1/kappa64/beta1. DevelopmentvIoU23.696096HC/16.379100Vid versus original23.036291/15.926685. All20 trials and losses are published. Selection is explicitly target-development selection, not independent benefit. No posthoc parameter combinations or additional grid are introduced. New-CG-parent confirmation compares selectedFull and oldTransferredFull on the SAME newcohort; this is the next decision point.


## Confirmation update (35HC+128Vid CG-new parents, not used for selection)

The frozen gamma0HC profile removes SharedCorrection changes, but Source67.832981% fixed-support unobservedIoU falls toSpace63.982395 andFull63.029979. Residual failure is at spatial candidate/path selection; disabling global correction alone is insufficient. SelectedFull recovers3.058149pp over oldFull59.971831 CI[.816119,6.456549], but remains belowSource. HCv selected26.420382 vsold24.530040 vsSource28.737788; selected−old+1.890341 CIcross0, selected−Source−2.317406 CIcross0.

Vid selectedFull11.530333 vsold11.879313 vsSource9.192995; selected−Source+2.337337 CI[.412902,4.357343], selected−old−.348980 CIcross0. UnobservedIoU40.602997 vsSource40.143025 andold39.551544; spatial change small, while selected−old temporalIoU−.882525pp CI[−1.842741,−.085210]. This combines gamma/context changes and cannot identify context alone as the cause. No confirmation-driven parameter revision follows. These findings support reduced HC spatial damage and a positive Vid Source contrast in this cohort, not consistent superiority to the original transferred profile.
