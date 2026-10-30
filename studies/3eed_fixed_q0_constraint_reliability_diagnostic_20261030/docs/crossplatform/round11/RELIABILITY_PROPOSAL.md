# R11 proposed reliability test: fixed targets, unlabelled evidence consistency

Prepared proposal only; no new model, GT, scoring, images or point arrays opened. Await root agreement before implementing/running the bounded pure-CPU module. R10 source/geometry code and every old prediction stay unchanged.

## Question and fixed estimand

R10 produced valid segmentation in157/160 but only12 center coordinate applications; the actual point-plane correction harmed the task while the matched RGB coordinate term had a small positive increment. Availability/observability and reliability are different questions. This study asks whether label-free evidence features rank useful versus harmful **applications of the already generated original G target**. It does not identify true object identity and does not certify a planar patch as an object's amodal bounding face.

The original fixed-q0 source box, full G, no-RGB-coordinate G, segmentation and acceptance remain byte-preserved baseline outputs. Reliability may (a) rank/select applications at a predeclared coverage or (b) interpolate a distinct attenuated arm `source + r_axis*(G-source)`. The latter changes applied output, not the stored target, and must not be called unchanged G. There is no teacher, EMA, training, query reselection, CLIP-confidence gate, point-count primary score, source residual primary gate, source correction amplification or forced shift.

## Evidence generation and dependence

Unique physical raw returns/XYZ are assigned to two deterministic spatially interlaced voxel partitions. A physical return never occurs in both halves. Each half independently estimates visibility, normals/planarity and source-backed GrabCut seeds. Normal graphs are never shared. Shared RGB and source q0/prior initialization are disclosed: the halves are disjoint point evidence, not independent object labels, views or scenes. A spatial partition should be fixed from metric cells and a sample-salted hash before inspecting any targets/labels; no alternate partitions are searched.

Compute extra reliability evidence only for base G actual-change candidates; zero-change outputs are identities and remain explicit zero/undefined units. Original G output is not recomputed from these partitions. Calls consume explicit arrays/camera and cannot read files, metadata, GT or scoring sidecars.

## Proposed distinct feature blocks

1. **RGB constraint stability, not identity:** mask-A/mask-B intersection and boundary agreement; A-seeded mask covering B-only projected visible returns and converse; record false stability possibility when both follow the same source/background error.
2. **Crossmodal ambiguity:** within the unchanged source crop and foreground, physical 3D connected components and depth-separated support competing for the same 2D foreground component. Report component fragmentation, competing support mass, depth separation and image-connected/3D-disconnected evidence. A largest component is a support hypothesis, never an instance-ID label. No RGB/fullcaption localizer exists in this round; original text only fixes q0.
3. **Face stability under spatial leave-out:** for each accepted original source-facing face, independently compute normal orientation/planarity, coordinate-plane median/MAD and tangential span from partitionA/B. Use plane-offset/normal disagreement and coverage loss, not agreement with source face offset, as primary stability inputs. Single-plane center remains conditional on unchanged source size; stable source-size error can survive all checks.
4. **Required simpler controls:** physical face support count and source agreement measured by normalized original G displacement. These are controls against reliability merely rediscovering density or preference for no change. They cannot be presented as independent correctness evidence.

Mask stability alone cannot qualify reliability. Joint scores must expose their contributing factors and undefined/rejected reason; a continuous heuristic product is not calibrated correctness probability. Exact formulas/constants require root approval and locking before actual inputs or labels are read. R10 point-normal and plane scales should be reused where meaningful rather than optimized on target scores.

## Units, controls and proposed finite comparisons

Geometry primary unit is a coordinate-application `(sample_id, axis_j)` with original accepted mask and actual float32 G displacement. It is not a causal leave-one-face-out effect. RGB unit is the sample-level coordinate constraint, retaining whether full-minus-noRGB actually changes the box. No face refit/causal effect is claimed unless separately registered and sealed.

Keep unconditional original source/full/off-coordinate. One predeclared fixed coverage is top ceil(n/2) among base actual-change units within platform; use the exact same eligibility and K for reliability, fixed-salt hash-random, source-agreement and physical-point-count rankings. Tie order is a fixed input-only hash. Do not scan coverage, select thresholds with GT or discard undefined scores. A continuous attenuation arm is separate from the fixed-coverage ranking test and uses the same stored original target. Keep all eligible, zero-change and undefined units in explicit counts; whole-cohort task metrics keep all samples. Ranking controls may preserve K even when score is undefined by placing undefined last under the predeclared tie rule; undefined fallback is not silently dropped.

Every source/G/reliability feature, score, selection and output must seal before GT. Post-seal task analysis compares exactly matched coverage/denominators and retains positive/negative/zero effects. Correlation with source correctness or IoU is then descriptive diagnostic evidence, not online identity knowledge.

## Cost and falsification

R10 actual four-arm CPU geometry cost145.241s/160. A fresh320-sample base run is approximately5min at similar input sizes; two extra half-seeded masks/independent point graphs cost at most a few additional whole-run equivalents, with extra work restricted to actual-change units. Root budget is20CPUminutes. No GPU or new pretrained model is needed for this module.

Failing outcomes include low or zero crossfit support, stable but wrong background surfaces, reliability reducing to point density/source agreement, no better ranking than matched random, positive matched RGB effects masked by harmful geometry, or attenuation improving only by suppressing nearly all corrections. None justifies a learned reliability or TTA claim. Preserve the outcome and stop this bounded configuration rather than searching gates.
