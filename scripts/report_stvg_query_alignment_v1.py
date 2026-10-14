"""Anonymous scalar PDF; every page requires independent actual pixel review."""
from pathlib import Path
import json,shutil
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/stvg_hierarchical_evidence_guided_v1'
D=BASE/'public_stage/results/stvg_query_aligned_sparse_r1/2026-10-14'
s=json.loads((D/'summary.json').read_text());out=ROOT/'output/pdf/stvg_query_aligned_sparse_r1_P0AB.pdf';out.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Body','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
c=canvas.Canvas(str(out),pagesize=(612,792));c.setTitle('Query-Aligned Sparse STVG-R1: fixed development comparison')
def text(x,y,t,size=9,bold=False,color='#253649'):
 c.setFont('Bold' if bold else 'Body',size);c.setFillColor(HexColor(color));c.drawString(x,y,t)
def lines(y,seq,size=9):
 for t in seq:text(42,y,t,size);y-=15
 return y
def section(y,t):text(42,y,t,12,True,'#143958');return y-23
def title(page,sub):
 c.setFillColor(HexColor('#143958'));c.rect(0,713,612,79,fill=1,stroke=0)
 text(42,753,'Query-Aligned Sparse STVG-R1',18,True,'#ffffff');text(42,730,sub,10,False,'#dce7f0')
 text(42,27,'2026-10-14 | Frozen R1 + cold cross-domain TA-STVG | Historical development only',8)
 text(538,27,f'{page}/3',8)
title(1,'Initial pass: actual geometry contracts and direct versus explicit alignment')
y=section(680,'Finite scope and preserved controls')
y=lines(y,['Eight fixed historical cases: four HC-STVG2 and four VidSTG. A reuses the original',
'48 outputs. B sees Query + paired video; C first describes marked candidates without',
'Query, then aligns the facts to Query using the same frozen R1 in a text-only call.',
'B / C use exactly the same eight-frame visual payload per paired request. ID swaps,',
'panel-order swaps and mismatched-query controls are retained. No prompt retries.'])
y=section(y-16,'P0-A: actual old 32 spatial candidates')
y=lines(y,['13 / 32 original candidates contain out-of-bounds xyxy coordinates. Their original',
'bytes / hashes are preserved; a fixed legal mapping is declared before new inference.',
'Independent coordinate clipping, action inversion and original-z Gaussian checks pass.',
'The sparse interface takes K <= 4 observed, verified slots. Missing receipts give K = 0.',
'CPU forced receipts test the interface only: zero actual teacher anchors / student fits.',
'Mapped geometry is shared by visualization, candidate readout and reward contract.'])
y=section(y-16,'Actual output validity (exact schema, no semantic field repair)')
for kind,label in [('direct','B direct pairwise'),('facts','C query-free facts'),('alignment','C text-only alignment')]:
 v=s['output_validity'][kind];text(42,y,label,10,True);text(315,y,f"{v['valid']} / {v['total']} valid",10);text(475,y,f"{v['truncated']} truncated",9);y-=23
y=lines(y-4,['The denominator counts actual calls; invalid upstream facts skip alignment and',
'remain explicit missing evidence. All 162 C replies omitted only the schema field;',
'independent raw readback found Tie / Uncertain only, no directional preference.'])
y=section(y-16,'Do candidate-conditioned descriptions differ?')
text(42,y,'Fact field',9,True);text(238,y,'Different A/B text',9,True);text(428,y,'Unknown values',9,True);y-=20
for f,v in s['fact_dimension_discrimination'].items():
 text(42,y,f,9);text(248,y,f"{v['different_candidate_text_count']} / {v['valid_pairs']}",9);text(461,y,str(v['Unknown_value_count']),9);y-=19
y=lines(y-5,['Text variation is not automatically correct localization supervision. Identity,',
'boundary and geometry remain separate. 178 / 236 order facts say left / right.',
'Six relation facts echo the renderer caption rather than describe a physical relation.'],8)
c.showPage();title(2,'Initial pass: post-seal localization, control consistency and actual cost')
y=section(680,'Stable selections relative to mapped Frozen + Projection')
y=lines(y,['Selections are sealed before GT: a unique alternative must win consistently across',
'base / ID / order views; unsupported or conflicting evidence falls back to baseline.'],8)
for x,t in zip([42,207,270,349,446],['Domain / branch','Method','Update','Delta pp','Pair accuracy']):text(x,y,t,8,True)
y-=21
for g in s['groups']:
 label=('HC2' if g['dataset']=='hc2' else 'VidSTG')+' / '+g['branch'];acc=g['ranking_accuracy'];a='unknown' if acc is None else f"{100*acc:.1f}% ({g['informative_stable_decisive_count']})"
 for x,t in zip([42,216,285,359,446],[label,g['protocol'],str(g['directional_selection_count']),f"{g['mean_selected_minus_mapped_Projection_vIoU_pp']:+.3f}",a]):text(x,y,t,9)
 y-=23
y=lines(y-2,['Accuracy excludes GT ties and nondecisive / inconsistent pairs, whose counts stay',
'in the full anonymous table. Fallback readouts are not invented teacher choices.',
'A keeps its old unmodified-geometry scores; B / C are the matched new comparison.',
'B base pair accuracy is 75%, 58%, 64%, 75% (HC2 T/S, VidSTG T/S); ID agreement',
'is only 9/12, 0/12, 0/12, 2/12. Base accuracy alone does not establish robust feedback.'],8)
y=section(y-14,'Actual frozen-verifier cost (all positive / negative / unknown calls)')
a=s['actual_cost'];y=lines(y,[f"Actual calls: {s['calls']} / maximum {s['maximum_calls']}; preprocess + generate: {a['seconds']:.2f} s.",
 f"Tokens: {a['input_tokens']:,} input / {a['output_tokens']:,} output; peak: {a['peak_GPU_bytes']/2**30:.2f} GiB.",
 f"Model load: {s['teacher_model_load_seconds']:.2f} s; no observed OOM or guard stop.",
 'C adds text alignment cost. Frame extraction / student capture / future fits and final',
 'decode are not included in these verifier costs; this is not end-to-end TTA latency.'],9)
y=section(y-15,'Decision and limitations')
decision=s['technical_decision'].replace('_',' ')
# A fixed short label, with the exact machine decision retained in the scalar export.
y=lines(y,['This fixed feedback protocol did not trigger downstream GPU adaptation.' if not s['conditional_feedback_triggers'] else 'Feedback shows a development signal; downstream support / fit evidence is separate.',
'Current saved TA-STVG output provides one trajectory, not validated different-instance',
'hypotheses. Decoder-layer outputs are not automatically different object candidates.',
'A local-pool oracle bounds candidate selection, not the possible effect of parameter OPD.',
'Zero new student fits / backward / optimizer steps; OPD and temporal learning unmeasured.',
'This exposed eight-case cohort is not a fresh confirmation; reviewer GT-search incident',
'is recorded. Raw facts, queries, RGB, annotations and model state remain private.',
'No new DINO / YOLO / SAM2, no carry sweep, no CURRENT promotion or old queue restart.'],9)
c.showPage();title(3,'One display revision, exact C schema completion and final finite decision')
f=json.loads((D/'final_summary.json').read_text())
y=section(680,'Supported interface corrections; no candidate or parameter search')
y=lines(y,['The initial UI / fact audit found panel-position and caption copying. One fixed',
'display revision removed UI prose, added the same legal box in the crop and defined',
'event-order / boundary / geometry fields. The same old-eight source cohort remained.',
'A source audit then found an optional-version parser path and a non-JSON placeholder.',
'Both were corrected separately with CPU contracts. No semantic response was filled.',
'Sixteen source-role pairs were fixed: Projection vs Native; Native vs old large-plus.'])
y=section(y-15,'Actual follow-up outputs and reused evidence')
y=lines(y,['Display revision: 112 actual calls, 27 / 64 strict-JSON B replies, 0 / 48 valid facts.',
'Exact bare categorical readback makes all 64 B replies usable as original preferences;',
'their raw replies, prior stored parse, prompt and marked-pixel hashes remain preserved.',
'C completion: 48 fresh facts calls; 64 exact B reuses cost zero additional inference.',
'No complete facts record passed; therefore zero fresh text-alignment calls occurred.',
'Failures include numeric sequence / ID content and malformed JSON, not missing data.'])
y=section(y-15,'Keep partial facts; do not turn one bad field into a universal claim')
p=f['latest_partial_fact_diagnostic'];y=lines(y,[f"Among 40 complete JSON packets: subject {p['subject']['valid_values']} / 80 valid values; action {p['action']['valid_values']} / 80.",
f"Boundary: {p['boundary']['candidate_differences']} candidate differences; geometry: {p['geometry']['candidate_differences']} differences.",
'Event-order values were numeric sequences, so no validated event order was obtained.',
'Some semantic content exists. It is not a verified geometric or boundary reward.',
'Full C protocol failure does not prove query alignment impossible or worse than B.',
'Four short clips / paired views differ from native R1 input; this use remains unqualified.'])
y=section(y-15,'Total finite evidence and scope of the stop')
a=f['actual_cost'];y=lines(y,[f"658 fresh teacher calls total (498 + 112 + 48); old A cached responses: 48.",
f"Verifier preprocess + generate: {a['seconds']:.2f} s; peak GPU: {a['peak_GPU_bytes']/2**30:.2f} GiB.",
f"Tokens: {a['input_tokens']:,} input; {a['output_tokens']:,} output. No observed OOM or guard stop.",
'Zero accepted stable improvement selections. Zero new student fits / backward / optimizer.',
'P0-C pool expansion and P1 Direct / OPD / temporal / Combined / old-DINO control',
'conditions were not triggered. Their effects remain unmeasured; no old queue restarts.',
'The stop applies to this fixed feedback/input recipe only, not the whole research route.',
'All anonymous positive / negative / neutral / unknown results and revisions are retained.'],9)
c.save();shutil.copy2(out,D/out.name);print(out)
