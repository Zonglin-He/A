# Superseded method hypothesis — current candidate uses joint correction

**2026-09-28:** This original branch-isolated OPD proposal is preserved as history. Mandatory T/S gradient isolation and independently applied correction are no longer the principal-method hypothesis. The current candidate is [Decomposed Evidence, Joint Correction](JOINT_CORRECTION_CANDIDATE.md). The all16 saved-gradient attribution supports local complementary span capacity but does not establish the cause of finite native improvement, a learned mixer, external-evidence teacher quality, or OPD success. Existing scalar-mask STOP and all negative/positive results remain. No new model/OPD run is authorized by this document.

## Historical proposal below

# DESTA: evidence-conditioned branch-latent adaptation

Current mainline candidate, not an established method result. The source oracle has completed: temporal tIoU -1.785714pp, spatial sIoU +.218113pp but correct-minus-wrong spatial -.146974pp with CI crossing zero. Correct evidence advantage is not established. See `artifacts/desta3d_v3/latent_oracle_v1/oracle001/independent_readback_v1/REPORT.md` and `DECISION_AND_NEXT.md` for all results. Full-source fit remains cancelled at 140 query occurrences / 35 committed steps, retained for provenance only. It is not source-prepared or eligible as the teacher. Official external downloads were paused at the user's route change, then explicitly resumed to finish the pixel-view baseline; that baseline remains conditional on download/loader/qualification gates. No OPD optimizer has run.

## What is shared and what is decomposed

“Decouple the reasoning paths, not the spatio-temporal representation.” Here 3D means T×H×W, not metric XYZ. A shared video/query representation can support overlapping, task-conditioned event and referent evidence. It does not imply disjoint temporal/spatial evidence, or two physically asynchronous STVG timelines.

The computation is F=E_phi(V), Z=H_thetaC(F), Z_T=G_thetaT(Z,q), Z_S=G_thetaS(Z,q). Both readers see THW. Native PTD receives F_tilde_b=F+R_b(Z_b), where R includes the frozen output projection/bias and residual gate. Privilege replaces only the branch feature with P_b(Z_b;e_b), yielding F+R_b(P_b(Z_b;e_b)). It does not replace the original F or the final tube with external answers.

The first implemented operator weights post-reader/LayerNorm/SiLU features before out_proj: w=.25+.75M. Channelwise scalar scaling immediately before LayerNorm could be largely cancelled, hence that is not the insertion used. RGB, chronology and coordinates are preserved; feature distributions do change. The source masks are a physical event indicator and box-derived cell occupancy, not segmentation ground truth. Missing box annotations remain unknown/neutral.

PTD already has temporal and time-conditioned parallel spatial blocks, with access to shared video-query context. Decoupled decoding itself is not our novelty. Historical TTA already contained latent consistency, referent/event KL, source moments and parameter anchors; its limitations cannot be attributed solely to output-space compression. The unproven hypothesis is that suitable decision-specific evidence can create a native policy advantage that branch parameters can absorb.

## Three separately measured links

1. Teacher advantage: correct evidence must improve the corresponding native decision and outperform matched wrong evidence. The source oracle holds parameters fixed; Base is frozen PTD4B plus B1 no-update, not bare Frozen PTD. Report all t/s/v metrics, tails and original-good retention. S-only comparisons share Base reference/interval/anchors; T/dual final spatial effects include temporal conditioning.
2. Native actuation: separately registered source-GT controls test whether branch parameters can improve actual endpoints/coordinates at fixed finite updates. Prior GT-prefix CE controls did not establish this. A smaller loss is not native utility.
3. Absorption: only a qualified teacher and controllable branch justify registered OPD. Measure the final ordinary-input prediction with privilege withdrawn, and include provider cost during per-video adaptation. No expert in the final pass is not zero expert cost.

An oracle mask carries answer-related information. Oracle gains do not establish sufficiency of the unmodified latent or prove recovery of hidden evidence. If correct evidence clearly beats wrong controls, separately test preserved masks with disrupted feature content before claiming content recovery. Failure limits the specified operator/insertion/frozen-readout combination, not all latent adaptation.

## Future OPD contract (proposed, not implemented)

External models provide evidence only; the policy teacher is the same PTD under privileged branch latents. Freeze the episode-start teacher initially. At each student-visited native state, re-evaluate teacher and student on the same semantic reference, temporal block positions, student interval/time anchors, coordinate block state, valid action set and decoding constraints. H_T contains true endpoint positions; H_S contains true coordinate positions, not all NTP/MTP tokens indiscriminately.

A proposed branch loss averages KL(p_student || stopgrad p_teacher_privileged) on this identical support. Spatial blocks are parallel, not an invented autoregressive box trajectory. This is a native-state branch-conditional divergence; no claim of exact full-tube joint KL. Shared stem stays frozen initially; temporal loss has no direct gradient into spatial parameters and vice versa. Temporal adaptation still changes later spatial conditions. Direct update isolation does not eliminate all task coupling. Missing teacher support is declared, never replaced with target GT or privileged teacher's own superior interval while pretending supports match.

The contribution to test is the explicit correspondence among evidence, native decision, and adapted parameter scope, not an unqualified “3D + two readers + two KLs” combination.

## Primary-paper positioning, checked 2026-09-28

[DUST, section3.2 and theorem1](https://arxiv.org/html/2605.07910v1) shares canonical object representation while separating poses at genuinely asynchronous capture times. Its error statement assumes local linearized rendering. We borrow a variable-sharing design principle, not its theorem or dual-timeline physics for STVG.

[Vision-OPD](https://arxiv.org/html/2605.18740v2) already compares crop-conditioned and full-image policies of the same model on student rollouts. Its reported recipe selects EMA teacher and JSD; our frozen teacher/reverse-KL proposal is not an exact reproduction. The relevant candidate difference is structured STVG branch evidence and native conditional support, which still needs evidence.

[ViCuR, sections3.1–3.3](https://arxiv.org/html/2606.05718v1) motivates visual cues recoverable from ordinary inputs and adds a cue-recovery mechanism. Information availability does not guarantee successful recovery; our final no-privilege readout must demonstrate absorption.

[Decomposed OPD, section3](https://arxiv.org/html/2606.00564v1) separates language prior from visual grounding. This is distinct from event/referent branches; its gradient observations do not establish our temporal/spatial orthogonality.

[Video-OPD, TVDF](https://arxiv.org/html/2602.02994v3) uses GT temporal annotations for teacher reliability pre-validation. That gate cannot be imported into unlabeled target adaptation. Native rollout supervision can inform implementation, but data-use boundaries remain different.

Paper claims above were checked in the cited primary sections, not assumed from secondary summaries. No new architecture or optimization recipe is adopted just because it appears in a cited paper.
