"""Render only the already-public anonymous scalar reports; no private input access."""
import json, statistics
from pathlib import Path
from xml.sax.saxutils import escape
def build_pdf(main, diag, steps, readback, path, optimization):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.lineplots import LinePlot

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

    path.parent.mkdir(parents=True, exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title2', parent=styles['Title'], fontName='DejaVu-Bold', fontSize=20, leading=23, textColor=colors.HexColor('#13324B'), spaceAfter=6))
    styles.add(ParagraphStyle(name='Sub', parent=styles['Normal'], fontName='DejaVu', fontSize=9, leading=11, textColor=colors.HexColor('#516273'), spaceAfter=5))
    styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontName='DejaVu-Bold', fontSize=12.5, leading=15, textColor=colors.HexColor('#13324B'), spaceBefore=6, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name='Bodyx', parent=styles['BodyText'], fontName='DejaVu', fontSize=8.1, leading=10.5, spaceAfter=4))
    styles.add(ParagraphStyle(name='Smallx', parent=styles['BodyText'], fontName='DejaVu', fontSize=7.0, leading=8.5))

    def table(data, widths, fs=7.0):
        t=Table(data, repeatRows=1, colWidths=widths, hAlign='CENTER')
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173E5E')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'DejaVu-Bold'),
            ('FONTNAME',(0,1),(-1,-1),'DejaVu'),
            ('FONTSIZE',(0,0),(-1,-1),fs),
            ('LEADING',(0,0),(-1,-1),fs+1.2),
            ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#C7D1DA')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F8')]),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('ALIGN',(1,1),(-1,-1),'RIGHT'),
            ('TOPPADDING',(0,0),(-1,-1),1.6),
            ('BOTTOMPADDING',(0,0),(-1,-1),1.6),
            ('LEFTPADDING',(0,0),(-1,-1),4),
            ('RIGHTPADDING',(0,0),(-1,-1),4),
        ]))
        return t

    story=[]
    story.append(Paragraph('Temporal Caption/Query Adaptation with Spatial OPD',styles['Title2']))
    story.append(Paragraph('Eight-case post-seal development readout | 14 October 2026',styles['Sub']))
    story.append(Paragraph('<b>Scope.</b> Four historically exposed HC cases and four historically exposed VidSTG cases. GT metrics were read after prediction and diagnostic seals. This is not a fresh confirmation cohort; teacher-training overlap is unknown.',styles['Bodyx']))
    story.append(Paragraph('<b>Main readout.</b> T improves aggregate m_vIoU and m_tIoU over F and Projection in both datasets, with a small incremental gain over Projection on VidSTG. S and TS retain lower dense-GT m_sIoU than F in this panel. Spatial supervision used a student-selected pseudo-reference, not an independent geometry teacher; this outcome does not establish a general OPD failure.',styles['Bodyx']))
    story.append(Paragraph(f"<b>Optimization issue.</b> The fixed temporal KL+L2 objective ended above its initial value in {optimization['final_loss_above_initial_count']}/8 cases. Initial-minus-final is negative when the objective rose. Spatial loss is re-sampled on-policy, not a fixed-target convergence trace. No tuning or GT step selection followed.",styles['Bodyx']))
    story.append(Paragraph('Dataset-level arm averages',styles['H2x']))
    data=[['Set','Arm','vIoU %','tIoU %','dense sIoU %','d-vIoU vs F','d-tIoU vs F','d-sIoU vs F']]
    for r in main['arm_averages']:
        m,d=r['metrics_percent'],r['delta_vs_F_pp']
        data.append([r['dataset'],r['arm'],f"{m['m_vIoU']:.2f}",f"{m['m_tIoU']:.2f}",f"{m['m_sIoU_dense_GT']:.2f}",f"{d['m_vIoU']:+.2f}",f"{d['m_tIoU']:+.2f}",f"{d['m_sIoU_dense_GT']:+.2f}"])
    story.append(table(data,[22*mm,25*mm,24*mm,24*mm,30*mm,30*mm,30*mm,30*mm],7.0))
    story.append(Spacer(1,5))
    story.append(Paragraph('T0 preserves Projection. Temporal adaptation has 512 active parameters and five fixed steps; spatial OPD has 1,792 active parameters and five actual steps. The final step is reported, not a GT-selected checkpoint. Per-case metrics and all six-step scalars are in the adjacent JSON exports.',styles['Smallx']))
    delta_data=[['Case','T-F v/t (pp)','T-Projection v/t (pp)','S-F sIoU (pp)','TS-F sIoU (pp)']]
    for c in main['cases']:
        a=c['arms']
        delta_data.append([c['case'],
            f"{100*(a['T']['m_vIoU']-a['F']['m_vIoU']):+.2f} / {100*(a['T']['m_tIoU']-a['F']['m_tIoU']):+.2f}",
            f"{100*(a['T']['m_vIoU']-a['Projection']['m_vIoU']):+.2f} / {100*(a['T']['m_tIoU']-a['Projection']['m_tIoU']):+.2f}",
            f"{100*(a['S']['m_sIoU_dense_GT']-a['F']['m_sIoU_dense_GT']):+.2f}",
            f"{100*(a['TS']['m_sIoU_dense_GT']-a['F']['m_sIoU_dense_GT']):+.2f}"])
    story.append(Paragraph('Per-case contrasts (percentage points)',styles['H2x']))
    story.append(table(delta_data,[22*mm,43*mm,58*mm,43*mm,43*mm],6.4))

    story.append(PageBreak())
    story.append(Paragraph('Per-case outcomes',styles['Title2']))
    story.append(Paragraph('All 20 case-by-arm rows are shown for each dataset. Values are percentages; positive and negative tails are preserved.',styles['Bodyx']))
    domain_tables=[]
    for ds in ('HC','VidSTG'):
        data=[[f'{ds} / Case','Arm','vIoU','tIoU','dense sIoU']]
        for c in [x for x in main['cases'] if x['dataset']==ds]:
            for arm in ('F','Projection','T','S','TS'):
                m=c['arms'][arm]
                data.append([c['case'],arm,f"{100*m['m_vIoU']:.2f}",f"{100*m['m_tIoU']:.2f}",f"{100*m['m_sIoU_dense_GT']:.2f}"])
        domain_tables.append(table(data,[20*mm,26*mm,22*mm,22*mm,28*mm],6.8))
    side=Table([[domain_tables[0],domain_tables[1]]],colWidths=[124*mm,124*mm],hAlign='CENTER')
    side.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),1),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
    story.append(side)
    story.append(Spacer(1,6))
    story.append(PageBreak())
    story.append(Paragraph('Optimization path and diagnostics',styles['Title2']))
    story.append(Paragraph(f"The fixed KL+L2 temporal objective finished above its initial value in {optimization['final_loss_above_initial_count']}/8 cases; negative initial-minus-final means it rose. Spatial loss is resampled on-policy. All six snapshots per case remain in JSON.",styles['Bodyx']))
    loss_data=[['Case','Initial loss','Final loss','Initial - final']]
    for r in optimization['records']:
        loss_data.append([r['case'],f"{r['initial_loss']:.5f}",f"{r['final_loss']:.5f}",f"{r['initial_minus_final']:+.5f}"])
    story.append(table(loss_data,[28*mm,36*mm,36*mm,40*mm],6.4))
    # Mean per-case temporal objective trajectory by dataset; spatial losses are per-step resampled.
    plot=Drawing(370,65)
    lp=LinePlot();lp.x=40;lp.y=12;lp.width=305;lp.height=40
    lp.data=[]
    line_series=[]
    for ds in ('HC','VidSTG'):
        selected=[r for r in steps['cases'] if r['dataset']==ds]
        means=[]
        for i in range(6):
            vals=[r['steps'][i]['temporal_loss'] for r in selected if r['steps'][i]['temporal_loss'] is not None]
            means.append((i,statistics.mean(vals) if vals else None))
        line_series.append((ds,[(x,y) for x,y in means if y is not None]))
    lp.data=[[list(pt) for pt in row[1]] for row in line_series]
    for i,col in enumerate((colors.HexColor('#147D92'),colors.HexColor('#D97928'))):
        lp.lines[i].strokeColor=col;lp.lines[i].strokeWidth=2
    lp.xValueAxis.valueMin=0;lp.xValueAxis.valueMax=5;lp.xValueAxis.valueStep=1
    lp.yValueAxis.labelTextFormat='%.2f'
    lp.xValueAxis.labels.fontName='DejaVu';lp.xValueAxis.labels.fontSize=6
    lp.yValueAxis.labels.fontName='DejaVu';lp.yValueAxis.labels.fontSize=6
    plot.add(lp)
    plot.add(String(40,56,'Mean temporal objective (initial state + five fixed updates)',fontName='DejaVu',fontSize=6.5,fillColor=colors.HexColor('#516273')))
    story.append(plot)
    story.append(Paragraph('HC temporal | VidSTG temporal',styles['Smallx']))
    story.append(Paragraph('Gradient comparison and within-query candidate correlations. QA/SSL comparisons use endpoint-NLL gradients on 512 frozen hidden-state parameters, not a vIoU derivative; GT gradients were not used for updates. Correlations are descriptive across four sources and 16 candidate rows per dataset.',styles['Bodyx']))
    grad_data=[['Case','QA-GT S0','SSL-GT S0','QA-GT native','SSL-GT native']]
    for g in main['gradient_diagnostics']:
        grad_data.append([g['case'],f"{g['QA_vs_GT_shared_S0']['cosine']:+.3f}",f"{g['SSL_replay_vs_same_GT_shared_S0']['cosine']:+.3f}",f"{g['QA_vs_GT_native_score']['cosine']:+.3f}",f"{g['SSL_replay_vs_GT_native_score']['cosine']:+.3f}"])
    grad_table=table(grad_data,[16*mm,22*mm,22*mm,25*mm,25*mm],6.1)
    corr_data=[['Set','Sources','Rows','Reward-tIoU within-query r','Gain-tIoU within-query r']]
    for g in diag['within_query_centered_visual_diagnostics']:
        corr_data.append([g['dataset'],str(g['independent_sources']),str(g['candidate_rows']),f"{g['within_query_centred_reward_tIoU_correlation']:+.3f}",f"{g['within_query_centred_gain_tIoU_correlation']:+.3f}"])
    corr_table=table(corr_data,[20*mm,16*mm,16*mm,32*mm,32*mm],6.1)
    side=Table([[grad_table,corr_table]],colWidths=[112*mm,120*mm],hAlign='CENTER')
    side.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
    story.append(side)
    gg=[['Set','Mean G (nats/token)','Positive G / candidates','Mean QA-GT cosine','QA T - shuffled T (pp)']]
    for z in diag['reward_shuffle_groups']:
        gg.append([z['dataset'],f"{z['mean_visual_gain_nats_per_token']:.4f}",f"{z['positive_visual_gain_count']}/16",f"{z['mean_cosine_QA_GT']:+.4f}",f"{z['QA_T_minus_shuffled_vIoU_pp']:+.3f}"])
    story.append(Paragraph('Visual gain and fixed reward-shuffle readback',styles['H2x']))
    story.append(table(gg,[22*mm,48*mm,48*mm,43*mm,48*mm],6.7))
    story.append(Paragraph('The SSL replay is a 512-parameter replay control, not the historical 66,306-parameter/32-panel fit. Higher visual-information gain G does not imply higher tIoU.',styles['Smallx']))

    story.append(PageBreak())
    story.append(Paragraph('Measured cost and evidence limits',styles['Title2']))
    cost=diag['cost'];data=[['Stage','Calls / cases','Input / output tokens','Seconds','Peak GPU memory (GiB)']]
    for key,label in [('method_teacher','Teacher method'),('auxiliary_diagnostic','Auxiliary teacher diagnostics')]:
        x=cost[key];data.append([label,str(x['calls']),f"{x['input_tokens']} / {x['output_tokens']}",f"{x['seconds']:.3f}",f"{x['GPU_peak_bytes']/1024**3:.2f}"])
    x=cost['student'];data.append(['Student readout',f"{x['cases']} cases; {x['temporal_optimizer_steps']} T + {x['spatial_optimizer_steps']} S steps",'--',f"{x['seconds']:.3f}",f"{x['GPU_peak_bytes']/1024**3:.2f}"])
    story.append(table(data,[44*mm,55*mm,43*mm,28*mm,38*mm],6.4))
    story.append(Spacer(1,4))
    story.append(Paragraph('The 96 auxiliary teacher requests comprise 64 paired-condition visual forced-likelihood requests, 8 reverse/replacement likelihood checks, 12 counterfactual caption calls, and 12 alignment calls. Eight reward-shuffle fits are CPU-only and add no teacher calls or full-model forwards. The student stage performed 8 temporal and 8 spatial fits with 40 optimizer steps in each branch.',styles['Bodyx']))
    story.append(Paragraph('Counterfactual description controls (hashes only)',styles['H2x']))
    cf=[['Case','Control','Reward delta']]
    for r in diag['counterfactual_description_hashes']:
        cf.append([r['case'],r['control'],f"{r['reward_delta']:+.3f}"])
    story.append(table(cf,[28*mm,55*mm,35*mm],6.2))
    story.append(Paragraph('Only hashes and reward deltas are included; description and query text are omitted.',styles['Smallx']))
    story.append(Paragraph('Evidence boundaries',styles['H2x']))
    limits=[
        'Historically exposed eight-case development panel; not fresh confirmation. Teacher-training overlap unknown.',
        'Ground-truth metrics were read after prediction seals; this was post-seal evaluation, not a no-GT run.',
        'Spatial reference was a student-selected pseudo-reference, not independent teacher geometry.',
        'QA/SSL gradients are endpoint-NLL gradients, not discrete vIoU derivatives; no GT gradient update.',
        'Visual-information gain and candidate correlations do not establish higher tIoU or causality.',
        'Negative spatial outcomes are retained but do not prove a universal OPD failure; no method promoted.',
    ]
    story.append(Paragraph(' '.join(escape(z) for z in limits),styles['Smallx']))
    story.append(Paragraph(f"Root receipt: {readback['root_receipt']['count']} cases, {readback['root_receipt']['metric_readouts']} metric readouts; official-metric reconstruction maximum absolute error {readback['official_metric_reconstruction_max_abs_error']:.3g}. Sanitized provenance and source hashes are in the JSON package.",styles['Smallx']))

    def footer(canvas, doc):
        canvas.saveState();w,h=landscape(A4)
        canvas.setStrokeColor(colors.HexColor('#CFD7DE'));canvas.line(12*mm,10*mm,w-12*mm,10*mm)
        canvas.setFont('DejaVu',6.8);canvas.setFillColor(colors.HexColor('#526273'))
        canvas.drawString(12*mm,6*mm,'Anonymous scalar report | post-seal development readout | 2026-10-14')
        canvas.drawRightString(w-12*mm,6*mm,f'Page {doc.page}')
        canvas.restoreState()
    doc=SimpleDocTemplate(str(path),pagesize=landscape(A4),rightMargin=12*mm,leftMargin=12*mm,topMargin=10*mm,bottomMargin=14*mm,title='Temporal Caption Query Spatial OPD Anonymous Scalar Report',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__ == '__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 def read(n):return json.loads((a.results/n).read_text())
 build_pdf(read('ANONYMOUS_RESULTS.json'),read('ADDITIONAL_DIAGNOSTICS.json'),read('STEP_SCALAR_TRAJECTORIES.json'),read('READBACK_METADATA.json'),a.output,read('OPTIMIZATION_PATH_DIAGNOSTICS.json'))
