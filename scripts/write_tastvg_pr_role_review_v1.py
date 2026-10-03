"""Deterministic tables and review from sealed audited anonymous results."""
import sys,json,csv,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tastvg_pr_role/2026-10-03'
read=lambda p:json.loads(Path(p).read_text())
SS={ds:read(OUT/ds/'SUMMARY.json') for ds in ['vidstg','hc2']}
ROLES=['P_A','R_A','P_W','R_W'];FITS=['all','role']
def num(v,scale=1):return 'undefined' if v is None else f'{v*scale:.4f}'
def cell(v,scale=1):
    return num(v['mean'],scale)+(f" [{num(v['ci95'][0],scale)}, {num(v['ci95'][1],scale)}]" if v['ci95'] else ' [undefined]')
def metric(z,fn,field,k):return z['regression'][fn][field]['metrics'][k]
def matrix(panel,k):
    lines=['| Dataset | Role | M-all | M-role | Paired role − all |','|---|---|---:|---:|---:|']
    for ds in SS:
        z=SS[ds][panel]
        for role in ROLES:
            lines.append(f'| {ds} | {role} | {cell(metric(z,"all",role,k))} | {cell(metric(z,"role",role,k))} | {cell(z["paired_role_minus_all_regression"][role][k])} |')
    return '\n'.join(lines)
def decisions(panel):
    lines=['| Dataset/model | AUROC | Balanced accuracy | Helpful/harmful accepts | Severe v-harms | Expert ΔvIoU pp vs A |','|---|---:|---:|---:|---:|---:|']
    for ds in SS:
        z=SS[ds][panel]
        for fn in FITS:
            d=z['decision'][fn];c=d['counts']
            lines.append(f'| {ds}/{fn} | {cell(d["metrics"]["auc"])} | {cell(d["metrics"]["balanced_accuracy"])} | {c["accepted_helpful"]}/{c["accepted_harmful"]} | {c["accepted_severe_v_harm"]} | {cell(d["utility"]["delta_v"],100)} |')
    return '\n'.join(lines)

report=[]
def add(t):report.append(t.strip())
add('''# Role-conditioned P/R readout audit

The population-only intervention fits A/W roles much better in-sample but
does not recover held-out absolute P/R quality. Every confirmation-corruption
role MAE rises in point estimate, with all paired MAE intervals crossing zero;
all eight role R2 values remain negative. HC's analytic decision loses helpful
accepts and admits more harmful corrections. Retain Vid's local severe-harm
rejection, but do not promote M-role or infer that latent information is absent.

This is explicitly target-search-GT-supervised CPU diagnosis on fixed cached
A8/L32 choices, not an unsupervised TTA method, new prediction stream or frozen
backbone performance comparison. No GPU, experts, candidates, replay or updates.

## Matched setting and provenance

Predecessor: d408f154fb753764fef4aac2ca510f9b3b8fd36a. Same source lists,
checkpoints Vid fbb1ed88 /HC ee72f0d9, sixth temporal hidden, Inside[512:768] ->
precision /Endpoint[0:512] -> recall, fixed source-selected ridge alpha Vid1/1
and HC1/.1. The original 32 search +16 confirm sources/dataset/one query/two
orders/six conditions/25% expert design yields 288 cached expert cells: Vid
16 search +8 confirm independent expert sources, HC14+7, each96+48 cells.
Corruption contains80 search +40 confirm cells/dataset; clean16+8. All have
historical exposure; source-disjoint confirmation is not fresh project testing.

M-all reuses its four sealed target-search fits trained on3072 candidate rows
per dataset. M-role fits four shared heads on192 ordered [A,W] rows/dataset;
retain identical rows if A=W, without labels filtering helpful/harmful. Both
arms have source-equal weights summing to1, unpenalized bias and identical
weighted normalization procedure. New mean/std/bias are learned only from the
role population, so normalizer, label prior and candidate diversity change
together; this is not a coefficients-only intervention under frozen statistics.
No role-ID feature, separate A/W heads, alpha search or decision threshold.
The quality readout never changes the original L32 winner or A's1792 spatial
parameter trajectory, VidK1/HCK8, two-offset pixels or expert positions.

Separate processes sealed role models, then all288 GT-free scalar readouts
before confirmation GT join. M-all predictions reproduce predecessor arrays.
No confirmation features/labels entered fitting. Hash integrity reads and
historical label exposure are disclosed. Private coefficients/normalizers,
latents, GT spans, media and original weights are excluded from publication.

## Absolute role readout: primary confirmation corruption

Raw predictions are not clipped for regression. Conditions -> orders -> equal
source aggregation and paired whole-source10000 bootstrap, seed20261003.
Intervals are exploratory, unadjusted; undefined draws remain undefined.''')
add('### MAE (lower is better)\n\n'+matrix('confirm_corrupt','mae'))
add('### R2\n\n'+matrix('confirm_corrupt','r2'))
add('### Pearson correlation\n\n'+matrix('confirm_corrupt','rho'))
add('''## Search fitting is successful in-sample

Search roles are the training population of M-role, not an independent
validation. Better fitting alone does not prove generalizable P/R encoding.''')
add(matrix('search_corrupt','r2'))
add('''## Fixed analytic correction decision

Clip P/R only for F=PR/(P+R-PR), with F(0,0)=0. Accept the already fixed W only
when eligible and predicted F(W)-F(A)>0; no-op retains A. Binary truth is cached
true delta_t>+/-1e-12; neutral/no-op remain but are excluded from binary metrics.
AUROC is oriented without sign flipping and equal-source-weighted within each
class. Utility is the existing expert subset under a frozen readout simulation,
not full-flow benefit or newly executed online adaptation.''')
add(decisions('confirm_corrupt'))
lines=['| Dataset | Paired AUROC | Paired BA | Paired expert ΔvIoU pp | Paired accepted precision |','|---|---:|---:|---:|---:|']
for ds in SS:
    z=SS[ds]['confirm_corrupt'];d=z['paired_role_minus_all_decision']
    lines.append(f'| {ds} | {cell(d["auc"])} | {cell(d["balanced_accuracy"])} | {cell(z["paired_role_minus_all_utility"]["delta_v"],100)} | {cell(d["accepted_precision"])} |')
add('\n'.join(lines))
add('''HC helpful accepts9->6 and harmful11->18; severe10->9 does not offset those
errors. Paired expert delta_v=-.5395pp[-1.2868,-.0357], accepted precision
-.1709[-.3622,-.0067] and source-balanced accuracy-.2405[-.4429,-.0571]; AUROC
and BA deltas have intervals crossing zero. These are local exploratory
results from seven sources. The helpful-source TPR loss is identical in this
small class composition; its degenerate bootstrap interval is not universal
certainty or a new reliability guarantee.

Vid helpful accepts remain19, harmful7->6, severe1->0. It also switches which
helpful cases it accepts: expert utility+.3339pp->+.2522pp vs A, paired change
-.0817pp[-1.1283,1.1228]. Thus the local harm rejection is preserved without
declaring net improvement or role mapping recovery.''')
add('## Clean control\n\n'+decisions('confirm_clean'))
add('## Per-order confirmation corruption (point estimates only)')
lines=['| Dataset/order | M-all AUROC/BA | M-role AUROC/BA |','|---|---:|---:|']
for ds in SS:
    for o,z in SS[ds]['confirm_corrupt']['orders'].items():
        entries=[]
        for fn in FITS:entries.append('/'.join(num(z['decision'][fn]['metrics'][k]['mean']) for k in ['auc','balanced_accuracy']))
        lines.append(f'| {ds}/{o} | {entries[0]} | {entries[1]} |')
add('\n'.join(lines))
add('HC order2 has no helpful class; its binary AUROC/BA are undefined, not chance. Delete-one-source outputs are influence of frozen predictions, not cross-validation or refits.')
add('''## Why pooled R2 versus role R2 is not sufficient root-cause proof

Changing selected-role population may matter, but selecting roles also changes
label variance and support. R2=1-MSE/Var(truth); a high pooled value alone
does not establish precise A/W estimates or a causal selection mismatch.
The following true variances and matched M-all pooled/role statistics make
that denominator change explicit; M-role's all32 application is extrapolation
outside its chosen training support, not its training objective.''')
lines=['| Dataset/population | True variance | M-all R2/MAE | M-role R2/MAE |','|---|---:|---:|---:|']
for ds in SS:
    z=SS[ds]['confirm_corrupt']
    for f in ['P_pool','P_A','P_W','R_pool','R_A','R_W']:
        a=z['regression']['all'][f];b=z['regression']['role'][f]
        lines.append(f'| {ds}/{f} | {num(a["truth_variance"])} | {num(a["metrics"]["r2"]["mean"])}/{num(a["metrics"]["mae"]["mean"])} | {num(b["metrics"]["r2"]["mean"])}/{num(b["metrics"]["mae"]["mean"])} |')
add('\n'.join(lines))
add('''## Preserved work and failure cases

The cases are selected by fixed delta_v extremes and largest changed-decision
gains/losses, not chosen to hide exceptions. Full288 anonymous rows remain.

* Vid source36/frame_drop/order1: true delta_t=-.07438, delta_v=-5.4593pp.
  M-all predicts+.12718 and accepts; M-role predicts-.06450 and rejects.
  This is a genuine local severe-harm rejection.
* Vid source36/motion_blur/order1: true delta_t=+.25219, delta_v=+17.3445pp.
  M-all predicts+.28857 and accepts; M-role predicts-.05934 and rejects.
  The same source's useful correction is lost; source-specific effects differ
  by observation condition, so a global protection claim is unsupported.
* HC source34/frame_drop/order1: true delta_t=-.05974, delta_v=-4.4188pp.
  M-all's-.01012 correctly rejects; M-role's+.05565 now accepts. This undoes
  the predecessor's local relative-decision repair.
* HC source33/motion_blur/order1: true delta_v=-6.0932pp. M-role rejects the
  previously accepted severe harm; retain this positive case despite negative
  aggregate results. HC source33/frame_freeze's+5.0708pp correction is also
  rejected by M-role, while M-all accepted it.

## Evidence and decision

The simple hypothesis that replacing generic candidate training by deployment
A/W roles suffices to recover held-out quality is weakened in this fixed
setting. Search fitting is strong, confirmation role measurement remains
poor and relative execution does not improve consistently. It does not
identify a unique deeper cause: few independent sources, correlated repeated
cells, fixed source-selected alpha, population/normalization changes, role
label range and linear-family generalization remain competing explanations.

Pause further rule/threshold construction on this final-layer linear P/R
readout in its current configuration. Preserve A and CURRENT; no supervised
weight promotion. No automatic MLP, layerwise, motion/appearance, new expert,
loss or GPU phase is started. The result does not prove that the latent lacks
information, that P/R structure is useless, or that nonlinear alternatives
would succeed. No additional oracle ladder was introduced.

## Verification and resources''')
root=read(OUT/'ROOT_AUDIT.json');public=read(OUT/'PUBLIC_AUDIT.json');resource=read(OUT/'RESOURCES.json')
add(f'''10 meaningful CPU controls pass. Root audit independently verifies eight
ridge systems (four reused M-all plus four M-role) using SciPy SPD solve,
all immutable inputs, training memberships, physical labels, unlabelled
prediction parity, state/choice hashes and every metric/bootstrap:
**{root['total_scalar_checks']} scalar checks**. Public audit verifies all
anonymous regression/acceptance/statistics without private latents, weights
or GT spans: **{public['total_scalar_checks']} checks**.

Four new scientific ridge fits are distinct from eight verification refits.
Scientific CPU wall: fit{resource['fit_CPU_wall_seconds']:.3f}s,
readout{resource['readout_CPU_wall_seconds']:.3f}s,
diagnose{resource['diagnose_CPU_wall_seconds']:.3f}s;
root{root['CPU_wall_seconds']:.3f}s/public{public['CPU_wall_seconds']:.3f}s
verification are separate. All new GPU/backbone/expert/candidate/replay/
backward/online-stream/production-update counts are zero.

Figures: [search/confirmation role MAE](../results/tastvg_pr_role/2026-10-03/figures/role_fit_generalization.png),
[fixed decision](../results/tastvg_pr_role/2026-10-03/figures/role_confirmation_decision.png),
[paired confirmation MAE](../results/tastvg_pr_role/2026-10-03/figures/role_paired_confirmation_mae.png).
All have PDF equivalents and source-data bindings. Full scalar outputs:
[anonymous rows](../results/tastvg_pr_role/2026-10-03/ROWS.json),
[Vid summary](../results/tastvg_pr_role/2026-10-03/vidstg/SUMMARY.json),
[HC summary](../results/tastvg_pr_role/2026-10-03/hc2/SUMMARY.json),
[decision](../results/tastvg_pr_role/2026-10-03/DECISION.json),
[protocol](../protocols/tastvg_pr_role_v1.md).''')
p=ROOT/'docs/TA_PR_ROLE_CONDITIONED_REVIEW.md';assert not p.exists();p.write_text('\n\n'.join(report)+'\n')
csvpath=OUT/'REGRESSION.csv';assert not csvpath.exists()
with csvpath.open('w',newline='') as f:
    writer=csv.writer(f,lineterminator='\n');writer.writerow(['dataset','panel','model','field','metric','mean','ci_low','ci_high','bootstrap_defined','bootstrap_undefined'])
    for ds,panels in SS.items():
        for panel,z in panels.items():
            for fn,fields in z['regression'].items():
                for field,stats in fields.items():
                    for k,v in stats['metrics'].items():
                        writer.writerow([ds,panel,fn,field,k,v['mean'],*(v['ci95'] or [None,None]),v['bootstrap_defined'],v['bootstrap_undefined']])
owned=['protocols/tastvg_pr_role_v1.md','docs/tastvg_pr_role_v1/EXECUTION.md',
    'scripts/run_tastvg_pr_role_v1.py','scripts/tastvg_pr_role_math_v1.py','scripts/test_tastvg_pr_role_v1.py',
    'scripts/audit_tastvg_pr_role_v1.py','scripts/draw_tastvg_pr_role_v1.py','scripts/write_tastvg_pr_role_review_v1.py',
    'scripts/tastvg_pr_accessibility_math_v1.py','scripts/audit_tastvg_pr_accessibility_v1.py']
p=OUT/'CODE_BINDING.json';assert not p.exists()
p.write_text(json.dumps(dict(time=time.time(),code={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in owned}),indent=2)+'\n')
print('ROLE_REVIEW_WRITTEN',str(csvpath))
