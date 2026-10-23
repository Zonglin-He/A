# Final temporal-feedback repair: coordinate stability

The user explicitly chose continued repair before full evaluation, then limited this
work to the last repair round. One fixed primary is declared; there is no grid or
fallback promotion. All 256 cached rows from the preceding expanded study are now
exposed development data, including its former confirmation split.

The preceding common-mean scalar shrink preserves the direction of the prototype
contrast. In this final candidate, each fitting offset is divided into four chronological
groups. Delete each group, recompute the positive/negative Bernoulli log-odds contrast,
and use its variation as a coordinate sensitivity proxy. The fixed reliability is
`r = eta^2 / (eta^2 + V)` (zero when both are zero). Each coordinate's positive and
negative means move toward its fitted unconditional mean by that reliability. Only
one q0-seeded update is performed. No feedback round, block-count or gain is selected.

This is a hypothesis about locally concentrated pseudo-label feedback. Four temporal
blocks and overlapping offsets are dependent; stable incorrect semantics can survive.
The replication statistic is not a calibrated uncertainty or event probability.
The reliability formula is a new fixed heuristic. The [NASS delete-group report](https://data.nass.usda.gov/Education_and_Outreach/Reports,_Presentations_and_Conferences/allreports/Using_the_Delete-a-Group_Jackknife_Estimator_in_NASS_Surveys.pdf)
provides replication-method background, not evidence that the STVG adaptation works.

Original author code remains frozen. `models/pipeline.py` applies `t_spatial_clas` to
selected frames; `grounding_model/classifier.py` averages the word-head logits across
frames, and `models/criterion.py` uses multi-hot BCE. This is pooled word supervision,
not per-frame event supervision. Original sampling and Source forward remain intact;
no foreground/actionness outputs are used as new feedback.

Controls include the exact old one/two-round, kappa16 and beta.75 outputs; raw one-round
feedback rescaled to the primary's exact L2 strength; the same reliability histogram
on shuffled coordinates, also L2 matched; reversed time; matched-histogram random
semantic features; reflected context; and the same evidence with a uniform legal prior.
Reflected context is not a known-negative identity. All original geometry, physical
MBR, complete-box DP and eight expert observations remain unchanged. No new model,
RGB, expert observation or optimizer is required.

Seven CPU tests passed before plan preparation: original/no-op equality, held-out
parity isolation, constant-feature no-op, coordinate permutation, replication formula,
strength matching, and a synthetic stable-versus-local-accident positive control.
All 256 predictions must seal before separate CPU official-label scoring.

Only if every frozen gate passes will the unchanged candidate receive one confirmation
on 32 new parents per direction (excluding the 128 expanded and old HC96/Vid32 panels).
No retuning follows that read. Failure at either stage ends this temporal repair; no
new full or automatic original-paper fallback is authorized.

## Completed result and final decision

All256 exposed development predictions sealed before independent CPU official labels.
The frozen primary FAILED; this is the last authorized repair. No new confirmation
collection, no further candidate/parameter scan, no new full and no automatic original
paper fallback. Original paper remains paused at HC1902scored/Vid1373saved.

Each direction below contains128distinct parents/onequeryeach. These are not the
prior64confirmation-only figures and are not full-dataset results. All256 are exposed
development; no test-set generalization claim is made.

| Direction | Method | tIoU % | complete vIoU % | delta t / v vsG6 pp | delta t / v vsSource pp |
|---|---|---:|---:|---|---|
|hc2|Source|47.6781|24.5281|-1.0026 / -3.7155|+0.0000 / +0.0000|
|hc2|NativeG6|48.6807|28.2435|+0.0000 / +0.0000|+1.0026 / +3.7155|
|hc2|JackknifePrototype|48.8396|28.3242|+0.1589 / +0.0807|+1.1615 / +3.7961|
|vidstg|Source|36.4970|13.7207|-1.4216 / -4.3702|+0.0000 / +0.0000|
|vidstg|NativeG6|37.9187|18.0909|+0.0000 / +0.0000|+1.4216 / +4.3702|
|vidstg|JackknifePrototype|37.9509|18.2841|+0.0322 / +0.1932|+1.4538 / +4.5634|

| Direction | Source short t gain pp | Source long t gain pp | wins / harms / ties vsG6 | severe t / v vsG6 | OriginalLoop2 severe t / v |
|---|---:|---:|---|---|---|
|hc2|-0.1083|+1.6394|49 / 41 / 38|12 / 7|24 / 12|
|vidstg|+0.8515|+2.1154|44 / 35 / 49|12 / 4|18 / 8|

HC short events remain below Source. The primary reduces originaltwo-round severe
harms, but remains worse than kappa16 in both mean time and severe counts. Full-v
system gains against Source include the fixed spatial method; they cannot all be
attributed to the temporal repair.

Primary t gain versus G6 has95% parent-bootstrapCI HC[-.6418,+.9878]pp and
Vid[-1.0714,+1.0972]pp. No reliable incremental time benefit is established.

## Did coordinate stability repair the evidence?

| Direction | Primary minus same-L2 uniform t pp | minus shuffled-coordinate same-L2 t pp | minus matched-random t pp | minus reflected t pp |
|---|---:|---:|---:|---:|
|hc2|-0.0085|+0.4210|-0.6167|-0.7243|
|vidstg|-0.4183|-0.3713|-0.0435|+0.2124|

The primary fails the prespecified matched-control condition in both domains.
In Vid, it is worse than exact-strength uniform feedback by.4183pp tIoU and.2410pp
vIoU, with paired95%CIs belowzero for both. Correct coordinate sensitivity does
not yield the needed task-specific advantage. Controls are retained, not promoted.

The coordinate weights do differ: median per-query reliability SD is.2785HC/.2411Vid.
Median cosine between old and corrected prototype contrast is.9797HC/.9747Vid:
it changes direction, but only moderately in these inputs. Same-L2 control max
absolute norm error8.89e-16; no degenerate strength control occurred. Hence this is
not simply a disabled mechanism or unmatched weaker feedback.

Evidence-only matched candidate ranking: HC primary50.41%, random51.64%,
reflected51.86%; Vid primary58.64%, shuffled-coordinate60.84%. The candidate rule
uses top64q0 cells, equalgridspan and priorratio<=2. Physical duration can still vary.
Only queries with quality-disagreeing matched pairs enter ranks(123HC/88Vid); all128
queries per direction remain in final metrics. These are offline diagnostics,
never GT-selected inference or accept/reject rules.

## Concrete rescues and failures

Cases follow a fixed postseal reporting rule written after completion: three largest t harms and three
largest t gains per direction. All anonymous signed rows are published, with no
qualitative category inferred from captions.

| Query | Group | primary delta t vsG6 pp | uniform same-strength delta t pp | shuffled same-strength delta t pp |
|---|---|---:|---:|---:|
|HC2_P128_120|short_le5s|-19.9455|-19.9455|-24.7865|
|HC2_P128_073|short_le5s|-13.5021|-13.5021|-13.5021|
|HC2_P128_112|short_le5s|-12.1862|-12.1862|-14.3215|
|HC2_P128_023|long_gt5s|+11.4425|+15.6436|+19.3303|
|HC2_P128_013|long_gt5s|+12.7126|+12.7126|+10.2661|
|HC2_P128_041|long_gt5s|+17.5573|+11.4504|+11.4504|
|VIDSTG_P128_118|long_gt5s|-35.2113|-40.8451|-38.0282|
|VIDSTG_P128_100|short_le5s|-16.1290|-10.7527|-10.7527|
|VIDSTG_P128_085|long_gt5s|-15.2727|-15.2727|-15.2727|
|VIDSTG_P128_017|short_le5s|+12.2727|+12.2727|+12.2727|
|VIDSTG_P128_128|long_gt5s|+12.5765|+12.5765|+11.3291|
|VIDSTG_P128_011|long_gt5s|+32.6241|+31.9149|+1.4184|

HC120/073/112 remain severely harmed despite nonconstant reliability. HC073 has
contrast cosine.9985; primary and uniform read exactly the same harmful interval.
Vid118 still loses35.21pp time (less than uniform40.85pp); Vid100 loses16.13pp,
more than uniform10.75pp. Strong positive Vid011(+32.62pp) is almost reproduced
by uniform(+31.91pp). Some positive individual corrections exist, but stability
alone does not separate event correctness from stable pseudo-label errors.
These cases weaken this particular repair, not a theorem that all native features
lack event information or all prototype methods are impossible.

## Verification and costs

SevenCPUtests passed. All256 exact oldSource/G6/Loop1/Loop2/K16/Weak baseline outputs retained. 6656 official-versus-independent-dense checks, maxerror9.99e-16. Frozen48paper/434provider unchanged; predictor/scorer realPID+startticksexited. No GPU work or shared GPU lease was needed by this CPU-only method.

Actual CPU prediction worker 31.83s; independent CPU scoring 22.70s. These include research traces/multiplecontrols/DP and are NOT deploymentbenchmark or cached-replay speedup. Newmodel/RGB/text/fusion/expert/observations/optimizer0. All code/protocol/signedpositive+negative/severe/CI/ranking and verification retained.

The user has explicitly limited this to the last repair. Research stops here with qualificationfalse; mainline remains the original frozen BoxFixed+Global6+physicalMBR, and no full job is started or resumed.
