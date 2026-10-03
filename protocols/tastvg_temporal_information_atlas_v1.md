# TA-STVG Temporal Latent Information Atlas v1 (CPU P0)

This is representation characterization, not a new selector or TTA method.
Reuse immutable `tastvg_temporal_latent_quality_v1` source/target final temporal
decoder hidden states and existing candidates. No GPU, model/expert calls,
backward, new candidates, output changes, MLP, new media or downloads.
Query swap, layerwise and appearance/motion capture are deferred, not launched.

## Cohorts and labels

Official same-domain source checkpoints and exact previous source split: Vid
126 queries/sources (95 fit, 31 validation), HC2 64 (48 fit, 16 validation).
Source training GT comes from the previously audited `SOURCE_GT.json`.
Target uses only the 288 cached expert arrivals (240 corrupt, 48 clean), not
all 1152 arrivals. Vid has 16 search + 8 confirm cached sources; HC has 14 + 7.
One query/source, two orders, clean + five 5% corruptions, original Paper48
sampling, A spatial state and output unchanged. Target historical exposure and
the four cached target-span schema inspections in this turn are disclosed;
this is not a fresh test or a new first-ever GT barrier. Fitting processes may
not read target GT/metrics. Freeze both datasets' entire probe set before
target readout; seal target readout before diagnostic label join.

The coordinate domain is the existing observed clip: origin = first frame ID,
T = last frame ID + 1 - origin. GT stays in physical frame coordinates, including
parts outside that domain. Position is (frame-origin)/T; event label is s<=t<e;
signed start/end distances are (t-s)/T and (e-t)/T. Phase = (t-s)/(e-s), evaluated
only on observed GT-event frames (oracle-gated); no outside-event fill. Empty
event support and single-class event cases remain counted and disclosed.

Candidate precision = overlap/candidate length, recall = overlap/GT length,
tIoU = overlap/union; intervals keep [frame_i,frame_j+1), all 32 entries including
duplicates. Source native is candidate 0, target A8 is the saved anchor_index.
The source cache has no UVTG A8 decision. Delta-tIoU therefore uses a zero-
intercept *linear contrast*: feature(candidate)-feature(anchor), scaled using
source training differences; source labels relative to native, target labels
relative to actual A8. These anchors are explicitly different; this tests
relative linear accessibility and does not create a deployed relative scorer.
Also evaluate differences of the frozen absolute tIoU readout, so the direct
delta fit is not the only way of measuring target A8 differences.

## Probe family and controls

Frame views: 256D hidden; geometry position alone. Tasks: position, event
membership, signed start/end distance, phase. Continuous tasks use weighted
ridge; membership uses weighted logistic regression, no class balancing.
Each source has equal fitting weight irrespective of observed frame count.
Phase fitting uses GT-event frames only. Source-level intercept-only nulls
and event prevalence are reported. Start/end MAE after removing position is
reported separately: high distance R2 alone is not boundary evidence.

Candidate views: Endpoint(512), Inside(256), Context(512), Contrast(512),
Full(1792), Geometry(start,end,length:3). Same seven cached blocks as predecessor,
left/right context 1 second, missing side zero. Each view independently fits
P/R/tIoU and delta. Each source has equal weight (32 candidates per source).
No candidate/feature/view selected using target scores.

Alpha grid is fixed .001,.01,.1,1,10,100,1000 for all tasks. Fit normalization
uses source-fit rows only; normal ridge has a free intercept; delta has no
intercept and preserves zero at the anchor. Source-validation source-macro
MSE chooses continuous alpha, log loss chooses logistic alpha, smaller alpha
breaks exact ties; no source-validation refit. Logistic objective is weighted
mean binary log loss + alpha*||w||^2/2, FP64 L-BFGS, 1000 iterations, tolerance
1e-8; nonconvergence is reported, never silently treated as valid.

For every view/task add a single deterministic within-source *training-label*
shuffle with seed 20261003 and source/task hash. P/R/tIoU shuffle uses the same
candidate permutation; frame tasks use the same full-frame permutation, phase
permutes only event-supported frames. Delta shuffles keep anchor zero. Validation
and target labels stay real. The control preserves source label histograms and
between-source priors: it is NOT guaranteed to yield chance on every task.
Both geometry real/shuffled and latent real/shuffled are retained.

## Evaluation and interpretation

Report source validation, target clean/corrupt combined, and separate target
search/confirm, clean/corrupt and order readbacks. Validation participates in
alpha selection, so its result is descriptive. Average within frame/candidate
rows, then condition, then order, then source. Use 10000 paired source-bootstrap
draws (seed 20261003); never candidate/frame bootstrap. Primary regression
summary: source-balanced pooled R2 computed from source-weighted moments, MAE,
MSE. Also within-cell R2 with explicit zero-variance exclusions. Event AUROC and
AUPRC require both classes, with support/exclusions and prevalence reported;
never replace undefined AUC with .5. Publish paired latent-over-geometry and
real-over-shuffle differences, per-source statistics, selected alphas, all
source paths and heatmaps. No threshold/top1 deployment or scientific route
promotion is authorized by a high probe score.

Signed distances and phase contain position; geometry controls and position-
removed endpoint errors are necessary for interpretation. P/R prediction is
linear decodability in this frozen setting, not proof of semantics or query
specificity. A source-target performance gap does not uniquely identify
representation shift: label/prior/support/readout differences also matter.
Negative results do not prove absent nonlinear information or justify a new
expert. Frozen A predictions, source caches and all old failures remain intact.

## Completion/publication

Independent root checks bind all cached payload/label hashes, feature slices,
label arithmetic, source-only standardization/fitting/KKT and target readout.
Public audit recomputes anonymous sufficient statistics, source aggregation,
paired bootstrap and all reported differences; it does not reconstruct private
latents/weights. Publish code/protocol, all anonymous paths/statistics, figures,
positive/negative findings and limitations to Zonglin-He/A and verify every
remote file. Exclude raw spans/annotations, frame chronology, captions/media,
latent caches and trained weights. Update RESEARCH_HISTORY and run
research_archive.py check/snapshot/check before completion.
