"""Deterministic tables, curve values and CSV from independently audited results."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tastvg_pr_loso/2026-10-03'
SS={ds:json.loads((OUT/ds/'SUMMARY.json').read_text()) for ds in ['vidstg','hc2']}
ROLES=['P_A','R_A','P_W','R_W'];V=['all/in_sample','all/nmax','role/in_sample','role/nmax']
def number(v,scale=1):return 'undefined' if v is None else f'{v*scale:.4f}'
def cell(v,scale=1):return number(v['mean'],scale)+(f" [{number(v['ci95'][0],scale)}, {number(v['ci95'][1],scale)}]" if v['ci95'] else '')
def regression_table(metric):
    lines=['| Dataset | Role | M-all in-sample | M-all LOSO | M-role in-sample | M-role LOSO |','|---|---|---:|---:|---:|---:|']
    for ds,z in SS.items():
        for r in ROLES:lines.append('| '+ds+' | '+r+' | '+' | '.join(cell(z['corrupt']['regression'][v][r]['metrics'][metric]) for v in V)+' |')
    return '\n'.join(lines)
report=['''# Source-held-out P/R readout audit

Both populations lose substantial accuracy when the evaluated search source
is excluded from fitting: all sixteen paired role MAE increases have intervals
above zero. M-role's in-sample R2 is .914-.996 in HC and .922-.948 in Vid, but
all eight M-role LOSO role R2 are negative. M-all does not provide a reliable
escape: HC's four R2 are negative; Vid precision retains small positive R2
(.130/.043), while its recall R2 is negative. This establishes a source-held-out
generalization gap inside the search cohort, so a separate confirmation-cohort
shift is not needed to reproduce the failure. It supports source sensitivity /
overfitting in this fixed readout configuration, without uniquely assigning
it to high dimensionality, literal memorization or representation absence.

The count curve offers qualified positive evidence in Vid: increasing12 to15
training sources reduces precision MAE for both populations (paired intervals
below zero), though intermediate points are nonmonotone and recall is weaker.
HC shows no consistent improvement from4 to13 and all12-to13 paired MAE
intervals cross zero. This short, single-sequence curve does not establish an
asymptote, or prove that more independent sources cannot help HC. Do not
jump to layerwise/MLP or a new gate from this diagnostic.

## Fixed cohort and supervision boundary

Predecessor c1558266eaefee13b745202f332f7f52bffa1748. Each dataset has96 cached
search expert arrivals (80 corrupt /16 clean), Vid16 /HC14 independent sources.
Original design32 search sources, one query/source, two orders, six conditions,
25% expert schedule; no nonexpert features added. All historically exposed.
Holdout applies to quality-readout supervision; cached persistent A states
can have encountered those sources without quality labels. No new confirmation
fit or evaluation; prior confirmation findings are not another endpoint here.

TA same-domain checkpoints Vidfbb1ed88 /HCee72f0d9, sixth temporal hidden,
Inside256 ->precision /Endpoint512 ->recall, fixed alpha Vid1/1 /HC1/.1.
M-all trains all32 candidates/cell, M-role ordered[A,W] with no-op duplicates.
Shared heads and equal-source weighted MSE with total weight1, unpenalized
bias and per-fold training-only normalization unchanged. A1792 Uniform
Rank-RKL VidK1/HCK8, Paper48 original pixels/two offsets, Old8/Expanded32,
A8/L32 winner W, expert schedule and spatial state trajectory stay fixed.

## Source holdout and nested count protocol

All conditions/orders/roles of one source are excluded together. Remaining
IDs have a locked SHA order per(dataset,held source). Nested4/8/12/remaining
source prefixes are shared by both models and P/R. Maximum means15 Vid /
13 HC sources, never includes the held source. One sequence per fold, no
subset search/repeats, no alpha/layer/threshold/dimension tuning.

Thirty source holdouts x4 counts x2 populations x2 heads =480 scientific CPU
fits. Eight prior heads provide in-sample controls, without scientific refit.
All models seal, then192 GT-free held-source scalar readouts seal before label
join. Extraction creates source-specific supervised packs; each fit denies
opening the held-source labelled pack. Independent verification refits are
counted separately. This is supervised diagnosis, not unsupervised TTA gain.

## Primary: raw search-corruption role regression

All regimes use identical test labels; MAE/MSE are unbounded predictions.
Conditions ->orders ->equal source. Whole-source10000 bootstrap seed20261003,
conditional on fixed OOF predictions and overlapping training folds; no
bootstrap refits or multiple-comparison adjustment. Undefined draws remain.
R2 compares with a test-label mean, not a deployed train-mean predictor.''']
for m in ['mae','r2','rho']:report.append('### '+m.upper()+'\n\n'+regression_table(m))
lines=['| Dataset/population | Role | 4 sources | 8 sources | 12 sources | Remaining sources |','|---|---|---:|---:|---:|---:|']
for ds,z in SS.items():
    for pop in ['all','role']:
        for r in ROLES:lines.append('| '+ds+'/'+pop+' | '+r+' | '+' | '.join(cell(z['corrupt']['regression'][pop+'/'+l][r]['metrics']['mae']) for l in ['n4','n8','n12','nmax'])+' |')
report.append('## Held-source count curve: MAE\n\n'+'\n'.join(lines))
lines=['| Dataset/population | Role | LOSO − in-sample MAE | Max − 12 MAE |','|---|---|---:|---:|']
for ds,z in SS.items():
    for pop in ['all','role']:
        for r in ROLES:
            a=z['corrupt']['paired'][pop+'/nmax minus '+pop+'/in_sample']['regression'][r]['mae']
            b=z['corrupt']['paired'][pop+'/nmax minus '+pop+'/n12']['regression'][r]['mae']
            lines.append(f'| {ds}/{pop} | {r} | {cell(a)} | {cell(b)} |')
report.append('## Paired gap and last count increment\n\n'+'\n'.join(lines))
lines=['| Dataset/model | AUROC | BA | Helpful/harmful accepted | Severe harmful accepted | Expert delta vIoU vs A (pp) |','|---|---:|---:|---:|---:|---:|']
for ds,z in SS.items():
    for v in V:
        d=z['corrupt']['decision'][v];c=d['counts'];lines.append(f'| {ds}/{v} | {cell(d["metrics"]["auc"])} | {cell(d["metrics"]["balanced_accuracy"])} | {c["accepted_helpful"]}/{c["accepted_harmful"]} | {c["accepted_severe_v_harm"]} | {cell(d["utility"]["delta_v"],100)} |')
report.append('''## Secondary fixed analytic decision

Clip only for F=PR/(P+R-PR). Keep the frozen eligible W iff predicted deltaF>0.
No new winner, threshold or gate. Utility reuses cached official dense scores
for expert arrivals; this is not a new full-stream prediction or25%-scaled gain.
Helpful/harmful uses true delta tIoU, severe harm is accepted delta vIoU<-.05.

'''+ '\n'.join(lines))
report.append('''## Controls, interpretation and deliverables

Clean, both orders, individual conditions, source moments, frozen-prediction
delete-source influence and undefined metrics are in SUMMARY.json. All192
anonymous scalar rows and original A/W identity hashes are in ROWS.json;
CASES.json retains smallest/largest held-source absolute errors and both
directions of correction-decision changes, without filtering training labels.
The learning curve changes source count in one nested sequence, not only an
IID row count. No small-source curve establishes an asymptote or a guarantee
that more data fixes the mapping. Population differences include candidate
diversity, label prior and fitted normalizers. Fixing source-selected alpha
tests transfer of this configuration, not optimal regularization at every n.

No nonlinear/layerwise/dimension/alpha sweep or new expert is initiated. A
and CURRENT remain unchanged; quality weights are private diagnostic models.
Protocol, CONFIG/FOLDS/FIT_SUMMARY, anonymous predictions, complete metrics,
negative cases, ROOT/PUBLIC_AUDIT, resources and figures accompany this review.
Root review records the measured interpretation after these tables are read.
''')
res=json.loads((OUT/'RESOURCES.json').read_text());report.append('''## Concrete work and failure cases

Vid source6/exposure/order2: M-role's mean four-role absolute error rises
.1305 in-sample ->.8772 held-out. True anchor P/R=.9733/1, but OOF predicts
.0545/-.0827; a previously rejected correction with delta vIoU=-21.5919pp
is accepted after analytic clipping. Vid source11/freeze/order2: M-all true
anchor and winner recall both.9643 become negative OOF predictions, losing a
helpful +10.2685pp correction. These failures survive within search-source
holdout, rather than requiring a separate confirmation panel.

Preserve work cases: Vid source3/exposure/order2 M-role held-out four-role MAE
.0824 and still rejects a harmful -7.2796pp correction. HC source14/freeze/
order1 M-all OOF MAE.0793 and still accepts +7.0399pp; M-role OOF MAE is even
smaller .0450 but rejects the same helpful correction. Thus low absolute error
need not yield correct relative zero-threshold choice. HC source26/drop/order1
has small OOF error and no-op stays A. HC source4/blur/order2 has true P/R=0
for both fixed intervals but role OOF estimates recall>1 and precision>.5;
this is a neutral candidate pair, retained in regression and excluded from
helpful/harmful binary counts, not silently labelled harmful.

M-role has fewer severe accepted harms than M-all (Vid10->8, HC6->3), but
does not establish safe benefit over A: expert delta vIoU=-.4789pp in Vid and
+2.1250pp in HC, with both intervals spanning zero. Vid role-minus-all utility
is +.7416pp [.0893,1.7718], a conditional exploratory improvement over another
imperfect readout, not a TTA promotion. All success/failure and clean/order
values are preserved. Different orders give different A/W contexts and the
same source-level split excludes both. The primary result remains absolute
and decision generalization loss, not an oracle-selected subset.

## Decision scope

All four in-sample-to-LOSO AUROC gaps (two datasets x two populations) have
negative paired intervals. Poor LOSO in both populations weakens a pure
all32-versus-A/W mismatch fix. However the paired M-role-minus-M-all absolute
MAE intervals all span zero, so this audit does not prove that reducing
candidate supervision is the unique variance cause. Fixing alpha tests this
family/configuration; dimensionality, regularization, finite independent
sources, candidate/label diversity and source-conditioned latent-quality
mapping remain competing explanations. No new dimension/regularization/data/
layer/nonlinear experiment is launched automatically. Retain A and CURRENT.
''')
report.append('## Actual resource accounting\n\n```json\n'+json.dumps(res,indent=2)+'\n```')
p=ROOT/'docs/TA_PR_SOURCE_HELD_OUT_REVIEW.md';assert not p.exists();p.write_text('\n\n'.join(report)+'\n')
q=OUT/'REGRESSION.csv';assert not q.exists()
with q.open('w',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['dataset','panel','variant','role','metric','mean','low95','high95','defined','undefined'])
    for ds,z in SS.items():
        for panel in ['all','corrupt','clean']:
            for v,d in z[panel]['regression'].items():
                for r,a in d.items():
                    for k,m in a['metrics'].items():w.writerow([ds,panel,v,r,k,m['mean'],*(m['ci95'] or [None,None]),m['bootstrap_defined'],m['bootstrap_undefined']])
print('LOSO_REPORT_AND_CSV_WRITTEN')
