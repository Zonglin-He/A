# A0.2 parameter-free global THW mean screen

Authorized 2026-09-29. A0.1 width and step rescues failed while fixed Q1 fitted
cosine .955796. Test only the missing global hidden context hypothesis, without
changing parameter count. Same sealed A0 Train128/95 parents and exposed
Dev64/16 parents, source-GT-derived cached evidence and oracle directions.
No new labels, pixels, PTD loading or native predictions during cached screen;
fresh31/388 untouched. Old data/code/pins/results remain immutable.

## Exact intervention and matched controls

Original inputfeature128/state33/evidence8/union256, internal128, radius
.13545580427763146. h0=silu(input(x)); hlocal=silu(DWConv3d(h0));
g=mean(h0,dims=T,H,W,keepdim=True); h=(h0+hlocal)+g;
coeff=output(silu(mix(h))). Zero added learnable parameters, no extra norm,
projection, attention, max pooling or new evidence. All107648 parameters and
initialization tensors equal original H128 before training.

Two fresh runs, seed20260928, same shuffle/pop order, batch4, AdamW lr.001/wd0,
clip1, sole per-query-global 1-cos loss. GMean-S200 fixed200 compared with
sealed Local-S200; GMean-S2000 fixed2000 compared with sealed Local-S2000.
The long run step200 is saved only for exact match to GMean-S200, not selection.
No LR/radius/step/architecture grid. Report complete query means/medians,
strict counts cosine>.1 and>.3, terminal mean1-cos, final training-batch loss,
preclip gradient min/max/last and clipping. Save every terminal coefficient;
independent NumPy cosine, frozen basis, integer/live Adam/counters audit.

Priority: S200 train median>=.3 and dev median>=.1 selects only S200; otherwise
S2000 with both selects S2000. A passing direction gate enables only a separately
registered Dev64 native stage at the same radius, using existing A0 harm/retention
and collapse rule (mean delta t or s<=-1pp AND >=12/16 negative parents).
Direction alignment alone never authorizes full/fresh/expert/OPD/target.

If both fail, stop width/steps/global-context architecture tuning and execute
only the conditional CPU structure audit below. No further architecture training.

## Conditional cached Oracle Field Structure Audit

All192 existing oracle_coeff256 tensors; no sample selection. Reshape [N,256],
N=THW, FP64 arithmetic. No centering for the requested raw-field rank.
- Global constant-channel mean projection and per-time spatial mean projection:
  energy ||P(A)||²/||A||² and cosine(A,P(A)); for an orthogonal projection these
  satisfy cos²=energy. Temporal includes the global component; report incremental
  temporal energy separately, do not sum nested energies as independent parts.
- Full256x256 channel Gram eigenspectrum, equivalent to all singular values²;
  cumulative energy at ranks1/4/8/16/32. Save complete spectrum, include zero-rank
  degeneracy explicitly. Numerical negative eigenvalues must be within relative
  1e-10 before clamping roundoff only. Independent Torch FP64 Gram/eigenspectrum
  compared with NumPy; synthetic SVD/projection controls before run.
- Adjacent T/H/W 256-D vector cosines, separately report mean/median and valid/
  undefined counts. Zero-norm pair undefined, never silently cosine0. Uniform
  observed-grid adjacency, not physical-time/optical-flow correspondence.
  Also finite-difference energy relative to field energy with explicit edge count.
- Per-query records and equal-query split mean/median; no token-count-weighted
  pooled replacement. Report Train128 and Dev64 separately. Low channel rank can
  coexist with complex/unsmooth THW variation and does not prove predictability
  from cached inputs. No rank-selected learned model or native interpretation.

## Resources

Two serial GPU allocations<=900s each,8GiB disk floor,new outputs<=4GiB.
Prior72513.02124451008s, cumulative cap=null; all measured loading/cache/readout
and nonoverlapping wrappers counted, failures retained, no silent rerun.
Conditional CPU audit<=900s,4 threads, incremental per-query processing, no GPU.
Existing Luna single read-only check; no extra long-lived timer for minute stage.
