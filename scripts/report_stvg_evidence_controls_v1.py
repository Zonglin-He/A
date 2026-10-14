"""Anonymous complete control/score export and finite diagnostic PDF."""
import sys,json,hashlib,shutil,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stvg_evidence_controls_inputs_v1 import ROOT,BASE,sha,read

def export():
 report=read(BASE/'POSTSEAL_REVIEW.json')
 anon={cid:('HC' if cid.startswith('hc2') else 'V')+f'{int(cid.rsplit("_",1)[1])+1:02d}' for cid in report['scores']}
 rows=[]
 for r in report['control_rows']:
  aliases={r['baseline']:'baseline',r['alternative']:'alternative'}
  def safe(v):return aliases.get(v,v) if v in ['Tie','Uncertain',None] or v in aliases else 'unrecognized'
  rows.append({**{k:v for k,v in r.items() if k not in ['case_id','baseline','alternative','controls','stable_choice']},'case':anon[r['case_id']],'baseline':'baseline','alternative':'alternative','controls':[safe(v) for v in r['controls']],'stable_choice':safe(r['stable_choice'])})
 scores={anon[cid]:{k:v for k,v in r.items() if k in ['dataset','Projection','Native']} for cid,r in report['scores'].items()}
 for cid in report['scores']:
  for branch in ['temporal','spatial','coarse']:
   rr=next(x for x in report['control_rows'] if x['case_id']==cid and x['branch']==branch);scores[anon[cid]][branch]={'baseline':report['scores'][cid][branch][rr['baseline']],'alternative':report['scores'][cid][branch][rr['alternative']]}
 selection=[{k:v for k,v in s.items() if k not in ['case_id','candidate_id']}|{'case':anon[s['case_id']]} for s in report['selections']]
 public=dict(status='complete_finite_P0_1_P0_2_diagnostic',unique_development_queries=8,historically_exposed=True,fresh_confirmation=False,teacher_training_overlap='unknown',format_and_query_diagnostics=read(BASE/'EXTRA_FORMAT_AND_QUERY_DIAGNOSTICS.json'),groups=report['groups'],control_rows=rows,scores=scores,output_validity=report['output_validity'],selections=selection,actual_cost=report['actual_cost'],candidate_metric_count=report['candidate_metric_count'],official_independent_max_abs_error=report['official_independent_max_abs_error'],actual_frozen_teacher_calls=report['count'],actual_student_fits=0,student_adapter_executed=False,conditional_fit_count=len(report['conditional_fit_case_roster']))
 # ANONYMOUS_RESULTS is immutable. Reconstruct the expected public object in
 # memory and require exact equality; never replace the sealed file.
 existing=read(BASE/'ANONYMOUS_RESULTS.json')
 assert existing==public, 'immutable ANONYMOUS_RESULTS differs from reconstructed anonymous report'
 response_rows=read(BASE/'ANONYMOUS_RESPONSE_ROWS.json')
 assert response_rows['status']=='complete' and response_rows['count']==232 and len(response_rows['records'])==232
 assert response_rows['raw_text_or_RGB_published'] is False
 expected_row_keys={'case','dataset','branch','representation','control','task','format_status','canonical_preference','valid_fact_field_count','output_truncated_at_cap','input_tokens','output_tokens','total_seconds','GPU_peak_bytes','native_example_time_copy','raw_response_sha256','marked_RGB_sha256'}
 assert all(set(r)==expected_row_keys for r in response_rows['records'])
 assert all(__import__('re').fullmatch(r'(?:HC|V)\d{2}',r['case']) for r in response_rows['records'])
 root_readback=read(BASE/'ROOT_READBACK.json')
 assert root_readback['status']=='pass' and root_readback['count']==232 and root_readback['hash']==sha(BASE/'POSTSEAL_REVIEW.json')
 # Independent scalar and shape equality audit, including negatives and unavailable values.
 assert public['groups']==report['groups'] and public['actual_cost']==report['actual_cost'] and len(public['control_rows'])==len(report['control_rows'])
 for rr,orig in zip(public['control_rows'],report['control_rows']):
  for k in ['stable','decisive','alternative_minus_baseline_vIoU_pp','informative_GT_pair','base_correct_when_informative','stable_correct_when_informative']:assert rr[k]==orig[k]
 for cid,orig in report['scores'].items():
  assert scores[anon[cid]]['Native']==orig['Native'] and scores[anon[cid]]['Projection']==orig['Projection']
  for b in ['temporal','spatial','coarse']:
   rr=next(x for x in report['control_rows'] if x['case_id']==cid and x['branch']==b)
   assert scores[anon[cid]][b]['baseline']==orig[b][rr['baseline']] and scores[anon[cid]][b]['alternative']==orig[b][rr['alternative']]
 return public

def report_pdf(pub):
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
 from reportlab.lib.styles import getSampleStyleSheet
 from reportlab.lib import colors
 from reportlab.pdfbase import pdfmetrics
 from reportlab.pdfbase.ttfonts import TTFont
 pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
 styles=getSampleStyleSheet()
 for st in styles.byName.values():
  st.fontName='DejaVu'
  if hasattr(st,'leading'):st.leading=max(st.leading,st.fontSize*1.3)
 styles['Title'].fontName='DejaVuBold';styles['Heading1'].fontName='DejaVuBold';styles['Normal'].fontSize=9;styles['Normal'].leading=13
 story=[]
 def p(t,style='Normal'):story.append(Paragraph(t,styles[style]));story.append(Spacer(1,8))
 def table(rows,widths):
  t=Table(rows,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),8),('LEADING',(0,0),(-1,-1),11),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf4')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#ccd3dd')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,12))
 p('Sparse STVG-R1: evidence and task controls','Title');p('Finite development diagnostic | 8 exposed queries | frozen STVG-R1-7B | 2026-10-14')
 p('P0-1: same-candidate input representation','Heading1');p('Temporal: paired sparse snippets versus two contiguous local boundary windows sampled uniformly at stride four. Spatial: same eight physical frames, paired small panels versus independent 448-square full-scene views. Different evidence/pixel budgets are reported; this is not isolated resolution causality.')
 g=[r for r in pub['groups'] if r['task']=='native_style' and r['branch']!='coarse'];rows=[['Domain / branch','View','ID same','Order same','Stable','Base correct']]
 for r in g:rows.append([r['dataset']+' / '+r['branch'][:1].upper(),'small' if r['representation']=='paired_small' else 'high',f"{r['ID_same_canonical_count']}/{r['pairs']}",f"{r['order_same_canonical_count']}/{r['pairs']}",f"{r['stable_count']}/{r['pairs']}",f"{r['base_correct']}/{r['base_informative_decisive']}"])
 table(rows,[132,67,58,73,53,74]);p('ID/order consistency counts all four fixed pairs per domain. Correctness counts only informative, decisive comparisons; different denominators must not be mixed. Stable Tie/Uncertain is not a usable update direction.')
 story.append(PageBreak());p('P0-2: Query-conditioned task controls','Heading1');p('Native-style adapts official Target ID / Time range syntax to supplied hypotheses; it is not native open-world benchmark accuracy. Simplified alignment receives partial Query-free subject/action facts plus the same visuals and Query. Boundaries are never judged without Query.')
 rows=[['Domain / branch','Task','ID same','Order same','Stable dir.','Stable correct']]
 for r in pub['groups']:
  if r['representation']!='independent_high' or r['branch']=='coarse':continue
  rows.append([r['dataset']+' / '+r['branch'][:1].upper(),'native' if r['task']=='native_style' else 'align',f"{r['ID_same_canonical_count']}/{r['pairs']}",f"{r['order_same_canonical_count']}/{r['pairs']}",f"{r['stable_decisive_count']}/{r['pairs']}",f"{r['stable_correct']}/{r['stable_informative_decisive']}"])
 table(rows,[132,67,58,73,53,74]);p('Output validity')
 table([['Task','Responses','Accepted','Truncated']]+[[k,str(v['total']),str(v['valid']),str(v['truncated'])] for k,v in pub['output_validity'].items()],[220,80,80,77])
 d=pub['format_and_query_diagnostics'];p(f"Native time-copy diagnostic: {d['native_example_time_copy_count']}/136 answers exactly copied the prompt example's 0.00-to-1.00 range. All {d['wrong_query_calls']}/{d['wrong_query_calls']} wrong-query calls kept the base ID preference; replacement queries were byte-distinct but may share visual semantics. Neither statistic alone proves complete Query blindness.")
 story.append(PageBreak());p('Coarse geometry and conditional adaptation','Heading1')
 table([['Domain','ID same','Order same','Stable dir.','Stable correct']]+[[r['dataset'],f"{r['ID_same_canonical_count']}/4",f"{r['order_same_canonical_count']}/4",f"{r['stable_decisive_count']}/4",f"{r['stable_correct']}/{r['stable_informative_decisive']}"] for r in pub['groups'] if r['branch']=='coarse'],[137,80,80,80,80])
 p('The coarse control truncates a student box by 55% on its right side. Neither candidate is asserted to be a complete correct referent. Post-seal truth checks quantify actual quality differences. Local candidates are not different object instances.')
 p(f"Conditional update cases: {pub['conditional_fit_count']}. Student fits executed: {pub['actual_student_fits']}. Zero fallback improvement is not an adaptation measurement. Temporal fitting to Native is a no-op; a reliable Projection target is needed to adapt the original Native logits.")
 c=pub['actual_cost'];p(f"Actual frozen evaluator calls: {pub['actual_frozen_teacher_calls']}; preprocess + generation {c['seconds']:.2f} s; input tokens {c['input_tokens']:,}; output tokens {c['output_tokens']:,}; peak allocated GPU {c['peak_GPU_bytes']/1024**3:.2f} GiB. Excludes source capture, view decoding/loading, and any student adaptation; not end-to-end TTA cost.")
 p('Limits and decision scope','Heading1');p('Only four exposed queries per direction, unknown teacher training overlap, no fresh confirmation or online carry. Stable preferences do not alone establish geometric correctness. Full-scene box-center labels differ from official mask-center IDs; glyph placement uses default PIL anchoring. Physical global offsets are supplied as text for separate clips. All positive, negative, abstaining and unknown outcomes remain in the anonymous JSON export. The finite diagnostic does not settle all MLLM feedback or OPD methods.')
 path=ROOT/'output/pdf/stvg_sparse_r1_evidence_task_controls.pdf';path.parent.mkdir(parents=True,exist_ok=True);SimpleDocTemplate(str(path),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=48).build(story);return path
if __name__=='__main__':
 pub=export();print(report_pdf(pub))
