"""Source-backed two-PDF report and additive paper claim revision."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_stvg_opd_temporal_spatial_v1 import BASE,OUT,SPATIAL,GROUPS,read,write,sha,rows_read,verify

def pp(v):
    return f"{v['mean']*100:+.3f} [{v['ci95'][0]*100:+.3f}, {v['ci95'][1]*100:+.3f}]"

def run():
    verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass'
    s=read(OUT/'SUMMARY.json');q=read(OUT/'Q0_DIAGNOSTIC_SUMMARY.json');rows=rows_read(OUT/'ROWS.jsonl.gz')
    lines=['# Temporal × spatial cached factor study and Q0 mean readout', '',
        'Existing Actionness Projection improves OPD under the two native source checkpoints. The interaction with OPD versus frozen Native is not established: both primary interaction intervals span zero. At projected time, the OPD-versus-output-capped-Direct contrast also spans zero in both primary panels; the two-order Vid secondary panel favors capped Direct. These cached measurements do not establish a new joint method or justify Query-level OPD or a temperature search.', '',
        '## Actual scope and input availability', '',
        'All 1,792 Common-State cells and 1,536 distinct packed inputs were checked before scoring. Exactly 128 clean cells match the complete P6 temporal captures in source-model state SHA, pixels, physical frame IDs, corruption, Native interval and frozen boxes. They represent the original 64 P6 parents (32 per target), repeated under two orders. The primary panel is their original order1 (64 cells); the secondary is both orders (128 cells), not new independent parents. All 192 panel rows / 1,536 logical eight-way readouts were hash-sealed before the new offline GT scoring.', '',
        'The remaining 1,664 O cells lack exact two-offset frozen temporal logits, raw actionness and physical-offset frame records. That full extension was not executed. No time signal was borrowed across source checkpoints or corruptions; no new GPU, backbone, optimizer, teacher or model call was made.', '',
        'T0 is frozen Native WHEN. T1 exactly replays original P6 Actionness Projection (HC center fraction 1; Vid 0.5; original prior 0.1 and epsilon 1e-6). Native is the frozen-source spatial output. OPD, Direct and Capped Direct share the original Full initial state, observations and their existing saved fits. Native-versus-adapted differences therefore include existing state history; the three adapted arms are conditional on Full history. The 576 original P6 Native/projection/SpatialOPD scalar controls were reproduced.', '',
        '## Complete eight-arm mean vIoU (%)', '',
        '| Panel | Target | Spatial readout | Native time T0 | Projection T1 | T1−T0 (pp), 95% CI | >5 pp temporal-harm parents |',
        '|---|---|---|---:|---:|---|---:|']
    for panel in s:
        for ds,b in s[panel].items():
            m=b['metrics']
            for arm in SPATIAL:
                v=m[f'delta_T_{arm}_v']
                lines.append(f"| {panel} | {ds} | {arm} | {m[f'{arm}_T0_v']['mean']*100:.3f} | {m[f'{arm}_T1_v']['mean']*100:.3f} | {pp(v)} | {v['harm_5pp']} |")
    lines+=['','## Primary and secondary paired contrasts','','| Panel | Target | Contrast | Difference (pp), 95% CI | >5 pp harm parents |','|---|---|---|---|---:|']
    contrast=['delta_T_OPD_v','interaction_OPD_Native_v','OPD_minus_Capped_Direct_T1_v','OPD_minus_Direct_T1_v']
    for panel in s:
        for ds,b in s[panel].items():
            for f in contrast:
                v=b['metrics'][f];lines.append(f"| {panel} | {ds} | {f} | {pp(v)} | {v['harm_5pp']} |")
    lines+=['','## Coverage and scoring boundary','',
        'Official dense scoring and an independent NumPy readback agree. Interpolation stays within the original full decoder frame grid; frames outside it receive zero IoU. There is no GT box completion, extrapolation or branch choice. Fixed-GT-event sIoU is identical under T0 and T1 for each spatial arm; tIoU is identical for all four spatial arms under each time choice. Complete mvIoU, mtIoU, msIoU, strict R@0.3/R@0.5 and all positive/negative parent rows are in ROWS and SUMMARY.', '',
        '| Target | Mean event coverage T1−T0 (pp), CI | Total newly included annotated frames outside Native | Total lost event frames | Gained frames outside decoder support |',
        '|---|---|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['panel']=='P6_primary' and r['dataset']==ds];m=s['P6_primary'][ds]['metrics']
        lines.append(f"| {ds} | {pp(m['delta_coverage'])} | {sum(r['gained_outside_Native_frames'] for r in rr)} | {sum(r['lost_event_frames'] for r in rr)} | {sum(r['gained_unsupported_frames'] for r in rr)} |")
    lines+=['','## Q0 signed-weight cancellation and Jensen gap','',
        'All 64 corrected-Native episodes, 192 direction-group readouts, 2,808 candidates and 1,308 signed pairs were read. Exact softmax, asymmetric-pair decomposition, float32 stored displacement and actual parameter-mean versus weighted-box decode were checked. The old full-video RMS budget qualification is not part of this scope. The current pooled Native budget and original target tau remain fixed.', '',
        'For actual unequal amplitudes, with d_mid=(delta+ + delta−)/2, d_half=(delta+ − delta−)/2 and pair mass m, the contribution is m[d_mid+tanh((R+−R−)/(2τ))d_half]. The simple equal-amplitude tanh formula alone would omit a real midpoint term. Zero-denominator ratios remain undefined; nine Vid groups per direction family have no nonzero candidate displacement. There is no reweighting, candidate reselection or temperature sweep.', '',
        '| Target | Group | Pairs | Unequal amplitudes | Median abs reward margin/τ | Median tanh retention | Median mean/top1 displacement | Median native Jensen RMS | Median true-mean native RMS |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds in q:
        for group,b in q[ds].items():
            st=b['statistics'];pair=b['pair_statistics']
            lines.append(f"| {ds} | {group} | {b['pairs']} | {b['unequal_amplitude_pairs']} | {pair['absolute_margin_over_tau']['median']:.5f} | {pair['tanh_retention']['median']:.5f} | {st['mean_over_top1_norm']['median']:.5f} | {st['native_Jensen_RMS']['median']:.7f} | {st['mean_native_RMS']['median']:.7f} |")
    lines+=['',
        'Observable Query pair margins have medians 0.02056 (HC) and 0.01162 (Vid) relative to tau; corresponding tanh retention is 1.028% and 0.581%. Actual mean/top1 displacement medians are 1.364% and 4.214%, with unequal amplitudes and direction summation accounting for differences from a one-pair illustration. Native Jensen RMS medians are 0.000414 and 0.001611; full-decoder medians are 0.004918 and 0.002424. True mean Native RMS medians are 0.0000482 and 0.0000459. This supports substantial shrinkage/nonlinear readout differences in these saved episodes. It does not identify their unique causal contribution to vIoU or prove tau must be changed.', '',
        '## Evidence and paper decision', '',
        'Retain Full unchanged as the reproducible research baseline and retain the P1 system improvements of HC +2.566 and Vid +2.968 pp. The common-state O controls do not establish stable Full-over-Direct superiority; correct Q0 does not establish stable Query-over-LN mean benefit. The new factorial study supports useful temporal coverage recovery, with no established OPD-specific interaction. Do not promote the query residual as indispensable or describe descriptive inherited-state decomposition as the causal effect of writeback. Expanded state, LN-only and full deployment timing still need the original S/C/N queues.', '',
        'The anonymous full draft has an additive evidence revision; the original CVPR project is preserved. A standalone claims/evidence source is supplied for built-in LaTeX preview. The manuscript is a working draft, not a submission-ready claim that all calibration/latency/state experiments have finished.', '',
        'All cohorts are historically exposed. Each target contributes 32 parents, the secondary panel repeats those parents, bootstrap uses 10,000 paired parent resamples with seed 20261012, and intervals are unadjusted for multiplicity. Positive-control P6, primary interaction and secondary negative results must be reported together. This finite cached study does not establish a full-population joint method, exact output-norm equivalence of the cap, or general impossibility of OPD/Query adaptation.', '',
        'ROOT_READBACK records complete source/readout/independent projection/dense/statistical and Q0 pair verification. Root readiness was initially dispatched before CPU_COMPLETION and correctly rejected; the original capture is preserved, then the same root source ran after scoring. Actual PDF pixel view, GitHub remote verification and archive closing have separate receipts.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    dest=ROOT/'output/pdf/stvg_opd_temporal_spatial_complementarity';dest.mkdir(parents=True,exist_ok=False)
    def forest(ax,metrics,fields,labels):
        for i,f in enumerate(fields):
            v=metrics[f];lo,hi=np.array(v['ci95'])*100;mean=v['mean']*100
            ax.plot([lo,hi],[i,i],color='#286da8',lw=2);ax.plot(mean,i,'o',color='#286da8')
        ax.set_yticks(range(len(fields)),labels);ax.invert_yaxis();ax.axvline(0,color='.5',lw=1);ax.grid(axis='x',alpha=.2);ax.set_xlabel('vIoU difference (pp), paired 95% CI')
    with PdfPages(dest/'factorial_readouts.pdf') as pdf:
        fig,axes=plt.subplots(2,2,figsize=(12.3,8.6),constrained_layout=True)
        for j,ds in enumerate(['hc2','vidstg']):
            m=s['P6_primary'][ds]['metrics'];ax=axes[0,j];x=np.arange(4)
            for offset,t,color in [(-.10,'T0','#65788c'),(.10,'T1','#207c82')]:
                means=np.array([m[f'{arm}_{t}_v']['mean'] for arm in SPATIAL])*100
                ci=np.array([m[f'{arm}_{t}_v']['ci95'] for arm in SPATIAL])*100
                ax.errorbar(x+offset,means,yerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=3,label=t,color=color)
            ax.set_xticks(x,['Native','OPD','Direct','Capped Direct']);ax.set_ylabel('Mean vIoU (%)');ax.set_title(('HC-STVG-v2' if ds=='hc2' else 'VidSTG')+' - 32 historical parents');ax.legend();ax.grid(axis='y',alpha=.2)
            forest(axes[1,j],m,contrast,['OPD: T1 - T0','Interaction: OPD - Native','At T1: OPD - cap','At T1: OPD - Direct'])
        fig.suptitle('Original P6 cohort: eight fixed time/spatial readouts',fontsize=15)
        pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12.3,8.6),constrained_layout=True)
        for j,ds in enumerate(['hc2','vidstg']):
            m=s['O_matched_secondary'][ds]['metrics'];forest(axes[0,j],m,contrast,['OPD: T1 - T0','Interaction: OPD - Native','At T1: OPD - cap','At T1: OPD - Direct'])
            axes[0,j].set_title(('HC' if ds=='hc2' else 'Vid')+' - both orders, same 32 parents')
            rr=[r for r in rows if r['panel']=='P6_primary' and r['dataset']==ds];ax=axes[1,j]
            for i,arm in enumerate(SPATIAL):
                vals=sorted(r[f'delta_T_{arm}_v']*100 for r in rr)
                ax.scatter(np.arange(32),vals,label=arm,s=18,alpha=.8)
            ax.axhline(0,color='.5',lw=1);ax.axhline(-5,color='#a74736',ls='--',lw=1);ax.set_xlabel('Parent rank within each spatial arm');ax.set_ylabel('Primary T1 - T0 vIoU (pp)');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle('Repeated-order sensitivity and complete temporal-harm tails',fontsize=15);pdf.savefig(fig);plt.close(fig)
    with PdfPages(dest/'q0_mean_readout_geometry.pdf') as pdf:
        fig,axes=plt.subplots(2,2,figsize=(12.3,8.6),constrained_layout=True)
        for j,ds in enumerate(['hc2','vidstg']):
            x=np.arange(3);dat=q[ds]
            for offset,field,label,color in [(-.1,'tanh_retention','Pair tanh retention','#286da8'),(.1,'mean_over_top1_norm','Mean / top1 displacement','#b85b30')]:
                vals=[dat[g]['pair_statistics'][field]['median'] if field=='tanh_retention' else dat[g]['statistics'][field]['median'] for g in GROUPS]
                axes[0,j].bar(x+offset,np.array(vals)*100,width=.2,label=label,color=color)
            axes[0,j].set_xticks(x,['Q-random','Q-observable','LN-observable']);axes[0,j].set_ylabel('Median retention (%)');axes[0,j].set_title('HC' if ds=='hc2' else 'Vid');axes[0,j].legend(fontsize=8);axes[0,j].grid(axis='y',alpha=.2)
            for offset,field,label,color in [(-.2,'native_Jensen_RMS','Native Jensen gap','#b85b30'),(0,'full_decoder_Jensen_RMS','Full decoder Jensen gap','#765aa4'),(.2,'mean_native_RMS','True parameter-mean native movement','#286da8')]:
                vals=[dat[g]['statistics'][field]['median'] for g in GROUPS]
                axes[1,j].bar(x+offset,vals,width=.2,label=label,color=color)
            axes[1,j].set_yscale('log');axes[1,j].set_xticks(x,['Q-random','Q-observable','LN-observable']);axes[1,j].set_ylabel('Normalized output RMS, median');axes[1,j].legend(fontsize=8);axes[1,j].grid(axis='y',alpha=.2)
        fig.suptitle('Correct Native Q0: actual pair cancellation and nonlinear readout',fontsize=15)
        pdf.savefig(fig);plt.close(fig)
    write(BASE/'FIGURE_EXPORT.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',PDFs=2,pages=3,
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in dest.glob('*.pdf')},
        CPU_only=True,GT_read=True,time=time.time()))
    revise_paper(s)

def revise_paper(s):
    original=ROOT/'paper/stvg_opd_reviewer_revision_20261011.tex';old=original.read_text();text=old
    abstract=r'''Spatio-temporal video grounding must recover an event interval and the trajectory
of a language-specified instance under deployment shifts. We study sparse
specialist-guided spatial test-time adaptation through a 1,792-parameter
decoder interface. On-policy localization actions are scored against one
frozen grounding specialist, and their reward-weighted distribution guides
the current-query spatial update while the native temporal interval is retained.
The fixed system increases mean video IoU from 20.017 to 22.583 on HC-STVG-v2
and from 11.129 to 14.097 on VidSTG. Common-state comparisons distinguish the
system gain from the spatial objective: they do not establish stable OPD
superiority over direct regression, and output-budget controls reveal
dataset-dependent tradeoffs. A cached temporal--spatial factorial study finds
that an existing actionness projection adds 3.823 and 2.079 percentage points
to OPD on two historical 32-parent panels, with no established OPD-specific
interaction. These results identify a reproducible compact adaptation system
and clarify which benefits are attributable to sparse feedback, spatial
readout and temporal coverage.'''
    start=text.index('\\begin{abstract}')+len('\\begin{abstract}');end=text.index('\\end{abstract}',start)
    text=text[:start]+'\n'+abstract+'\n'+text[end:]
    text=text.replace('One World, Two Timescales: Test-Time Adaptation via On-Policy Distillation\n\tfor Spatio-Temporal Video Grounding',
        'Sparse Specialist Feedback for Spatial Test-Time Adaptation\n\tin Spatio-Temporal Video Grounding')
    text=text.replace('We propose \\emph{One World, Two Timescales}, a specialist-guided','We study a specialist-guided')
    begin=text.index('Our contributions are threefold:');finish=text.index('\\section{Related Work}',begin)
    contributions=r'''Our contributions are threefold:
\begin{itemize}
\item We formulate sparse specialist-guided spatial adaptation for STVG,
keeping the temporal interval explicit and distinguishing frame-level
reference agreement from complete language-conditioned tube accuracy.
\item We implement on-policy spatial distribution distillation through a
compact decoder interface and evaluate the fixed system on complete official
cross-domain populations, including complete negative results and matched
direct-regression controls.
\item We provide controlled evidence separating spatial objective, parameter
readout and temporal coverage. Common-state and output-budget controls delimit
OPD's advantage; cached Query/LN diagnostics and temporal--spatial factorial
readouts identify useful mechanisms without claiming universal superiority or
an established benefit from state inheritance.
\end{itemize}

'''
    text=text[:begin]+contributions+text[finish:]
    text=text.replace('\\subsection{Dual-Timescale State Evolution}','\\subsection{Query Reset and LayerNorm Writeback}')
    text=text.replace('These two roles require different update speeds:\nthe current query benefits from the full correction, whereas the state shared\nacross queries should evolve more conservatively.',
        'The implementation assigns these roles different lifetimes: both parameter\ngroups form the current output, while the query residual and optimizer reset\nbefore the next arrival. A small LayerNorm displacement is retained by the\nfixed writeback rule. Its utility is an empirical question for matched\nstate-lifecycle controls.')
    text=text.replace('The query residual provides high-capacity, instance-specific correction at the\nfast timescale, while the LayerNorm state forms a low-dimensional path through\nwhich decoder adaptation can accumulate gradually across the stream.\nConsequently, a single query can strongly correct its own tube without forcing\nthe complete query-specific state onto subsequent video--query pairs.',
        'The query residual supplies 256 local coordinates and the selected\nLayerNorm parameters supply 1,536 shared coordinates. This describes the\nimplemented state lifecycle; it does not establish that both groups are\nnecessary or that retaining LayerNorm state causally improves grounding.')
    text=text.replace('while the query residual adds a\nclear benefit in the VidSTG transfer.',
        'while the query residual has a positive contrast in this particular\nVidSTG panel. Its indispensable contribution is not established: the enlarged\nmatched state controls and independent LN-only stream remain outstanding.')
    text=text.replace('These results show that temporal refinement\nand spatial adaptation address complementary errors.',
        'These Native-space controls demonstrate deployable temporal headroom.\nTheir interaction with spatial adaptation requires a matched factorial comparison.')
    newsection=r'''\subsection{Common-State Spatial and Parameter-Mean Controls}
The new common-state study uses every original matched cell, preserving Full's
initial state, observations and saved reference. All four primary Full-minus-
Direct intervals span zero. Full-minus-Capped-Direct is negative in both Vid
settings: cross-domain $-0.410\;[-0.718,-0.117]$ pp and same-domain corruption
$-0.303\;[-0.620,-0.006]$ pp. The cap bounds output movement; it does not force
exact norm equality. These controls are conditional on Full history rather
than three independent online streams.

The corrected Native-support Query observability study uses all 64 existing
development parents. Query-observable parameter-mean gains are $+0.002$ pp on
HC and $-0.004$ pp on Vid, while HC Query-minus-LN mean is
$-0.147\;[-0.238,-0.072]$ pp. Candidate oracle and expert-top1 readouts are
distinct from the deployed true parameter mean. Complete signed-pair replay
finds median reward-margin-to-temperature ratios of 0.02056 and 0.01162,
corresponding to 1.028\% and 0.581\% pair retention. Unequal positive/negative
amplitudes contribute a midpoint term; the equal-amplitude tanh illustration
alone is insufficient. These observations support mean-readout shrinkage in
this scope without identifying its unique causal role or selecting a new
temperature. Full Query-level OPD is not introduced.

\subsection{Matched Temporal--Spatial Readouts}
We cross frozen Native WHEN (T0) and the unchanged existing Actionness
Projection (T1) with frozen Native, Full OPD, common-state Direct and Capped
Direct. Exact cache matching admits all original P6 parents (32 per target)
and both existing orders, totaling 128 Common-State cells. The primary panel
uses original order1; the secondary clusters both orders within the same
parents. The other 1,664 cells lack complete corresponding temporal captures
and are not represented as a full extension. Every readout is sealed before
offline scoring; missing decoder support has zero IoU and no GT completion.

'''
    newsection+='\\begin{table}[t]\n\\centering\\small\n\\caption{Primary cached time--space contrasts (pp), 32 historical parents per target; 10,000 paired parent bootstrap, unadjusted 95\\% CI.}\\label{tab:time-space}\n\\begin{tabular}{lcc}\\toprule\nContrast & HC & Vid \\\\\n\\midrule\n'
    for field,label in [('delta_T_OPD_v','OPD T1 $-$ T0'),('interaction_OPD_Native_v','OPD/Native interaction'),('OPD_minus_Capped_Direct_T1_v','OPD $-$ cap at T1')]:
        def fmt(ds):
            v=s['P6_primary'][ds]['metrics'][field];return f"${v['mean']*100:+.3f}\\;[{v['ci95'][0]*100:+.3f},{v['ci95'][1]*100:+.3f}]$"
        newsection+=label+' & '+fmt('hc2')+' & '+fmt('vidstg')+' \\\\\n'
    newsection+='\\bottomrule\\end{tabular}\\end{table}\n'
    newsection+=r'''Actionness Projection improves the OPD readout in both targets, but neither
primary interaction interval excludes zero. OPD-minus-cap at projected time
also spans zero in both primary panels; the two-order Vid comparison is
$-0.771\;[-1.680,-0.020]$ pp. Mean annotated event coverage increases by
19.146 and 5.315 pp, with 1,087 and 1,275 newly included annotated frames
outside Native and no added frames outside decoder support. Four HC and one
Vid parents lose more than 5 pp under OPD's temporal change. Fixed-event
spatial IoU stays unchanged for each spatial readout. This supports useful
temporal coverage recovery without establishing an OPD-specific joint
mechanism or an independently validated new temporal adaptation method.

\section{Conclusion}
Sparse specialist feedback can improve a compact STVG spatial adaptation
system under the fixed cross-domain protocol. The current evidence supports
the complete system gain and distinguishes temporal coverage, spatial
objective and parameter readout; stable OPD-over-Direct superiority,
indispensable Query residuals and causal gains from inherited state remain
unestablished. Enlarged state, calibrated baseline and full deployment timing
claims require completion of their original locked evaluations. We retain the
fixed Full method and report conditional negative controls rather than
promote a new Query or temporal method from these diagnostics.

'''
    where=text.index('\n\t{\\small\n\t\t\\bibliographystyle')
    text=text[:where]+'\n'+newsection+text[where:]
    dest=ROOT/'paper/stvg_opd_evidence_revision_20261012.tex';assert not dest.exists();dest.write_text(text)
    # Standalone editable evidence preview deliberately requires no project assets.
    preview=ROOT/'paper/stvg_opd_revised_claims_20261012.tex'
    portable=r'''\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{amsmath,amssymb,booktabs,hyperref}
\title{Sparse Specialist Feedback for Spatial Test-Time Adaptation in Spatio-Temporal Video Grounding}
\author{Anonymous working revision}
\date{12 October 2026}
\begin{document}\maketitle
\begin{abstract}
'''+abstract+'\n\\end{abstract}\n'+contributions.replace('Our contributions are threefold:','\\section{Revised contributions}\nOur contributions are threefold:')+newsection.replace('\\subsection','\\section')+r'''
\section{Revision status}
This standalone file previews the revised claims and completed O/Q0/cached
factorial evidence. The additive full manuscript preserves the original
CVPR style, figure and bibliography references, which require the original
project assets. Calibration, expanded state, LN-only and complete deployment
timing remain in the original S/C/N queues; no results for them are invented.
\end{document}
'''
    assert not preview.exists();preview.write_text(portable)
    write(BASE/'PAPER_REVISION_RECEIPT.json',dict(status='revised_working_sources_pending_builtin_preview',
        original_sha256=sha(original),source_files={str(p.relative_to(ROOT)):sha(p) for p in [dest,preview]},
        preserves_original_source=True,submission_ready=False,S_C_N_complete=False,time=time.time()))
    claimmap='''# Claim-to-evidence revision, 2026-10-12

| Claim | Decision | Evidence or required completion |
|---|---|---|
| Fixed sparse specialist spatial system improves P1 cross-domain vIoU | Retain | HC +2.566 pp, Vid +2.968 pp; complete original P1 population |
| OPD stably beats direct regression | Remove universal claim | O common-state four primary CIs span zero; two Vid cap contrasts negative |
| Query residual is indispensable / Query OPD stably beats LN | Remove | Correct Q0 true mean near zero and HC Query−LN negative; enlarged LN study pending |
| Retaining LayerNorm state causally provides the inherited component | Remove inference | Descriptive Before−Frozen is not an intervention; original C and N controls pending |
| Spatial OPD repairs temporal coverage | Remove | Primary spatial method fixes Native WHEN |
| Existing temporal projection benefits OPD | Retain in bounded historical cache scope | HC +3.823 pp, Vid +2.079 pp; exact64 P6 parents; complete tails |
| OPD has a special positive interaction with temporal projection | Not established | Both primary interaction intervals span zero; cap comparator retained |
| Eight-arm study covers all Common-State cells | Reject | 128/1792 cells available, same64 parents; 1664 missing time captures |
| Small pair reward margin explains all task failure | Do not assert causality | Algebra demonstrates shrinkage/asymmetry/Jensen; no tau intervention |
| New complete latency or calibrated TENT/SAR results | Pending | Original S/C/N queues unchanged |

Reverse outline: task under shift -> sparse spatial interface -> measured system
gain -> matched mechanisms -> temporal support recovery -> bounded conclusions.
The full original method description and equations are retained; state lifecycle
is described as implemented rather than as a proven separate contribution.

Adversarial self-review:
1. Contribution: revised to reproducible sparse-feedback system plus controlled
   mechanism evidence; recombining existing temporal projection is not a new method.
2. Writing: distinguish Native source/Before/common Full history and candidate
   oracle/top1/true parameter mean; keep one evidence claim per paragraph.
3. Strength: retain P1 gains, Direct/cap negatives, zero-crossing interactions and
   temporal-harm tails together; no hidden winner selection.
4. Completeness: S/C/N baseline/state/latency and unavailable full O extension
   remain explicit gaps; preview is not a submission-ready completeness claim.
5. Soundness: no GT branch selection, unsupported interpolation, new training or
   temperature rescue. Historical exposure and unadjusted bootstrap remain clear.
'''
    (ROOT/'docs/STVG_OPD_CLAIM_EVIDENCE_REVISION_20261012.md').write_text(claimmap)

if __name__=='__main__':run();print('REPORT_PDFS_PAPER_WRITTEN',flush=True)
