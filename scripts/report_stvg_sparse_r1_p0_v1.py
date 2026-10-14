"""Render the anonymous finite screen from published scalar tables, no inference."""
import json, shutil
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("Body", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_sparse_stvg_r1_semantic_opd_v1'
D=BASE/'public_stage/results/stvg_sparse_stvg_r1_semantic_opd/2026-10-14'
s=json.loads((D/'summary.json').read_text())
OUT=ROOT/'output/pdf/stvg_sparse_r1_P0_screen.pdf';OUT.parent.mkdir(parents=True,exist_ok=True)
c=canvas.Canvas(str(OUT),pagesize=(612,792));c.setTitle('Sparse STVG-R1: finite P0 candidate verification screen')
navy=HexColor('#17324d');ink=HexColor('#263445');muted=HexColor('#556879');line=HexColor('#dde4ed');red=HexColor('#a63434')
def text(x,y,t,size=10,color=ink,bold=False):
 c.setFillColor(color);c.setFont('Bold' if bold else 'Body',size);c.drawString(x,y,t)
def lines(y,seq,size=9,step=15):
 for t in seq:text(44,y,t,size);y-=step
 return y
def title(p,sub):
 c.setFillColor(navy);c.rect(0,716,612,76,fill=1,stroke=0)
 text(44,753,'Sparse STVG-R1 Semantic Verification',17,HexColor('#ffffff'),True)
 text(44,729,sub,10,HexColor('#dce8f3'))
 text(44,27,'2026-10-14 | Frozen teacher + cold TA-STVG student | Historical development cohort',8,muted)
 text(535,27,f'{p} / 2',8,muted)
def section(y,label):text(44,y,label,12,navy,True);return y-22

title(1,'P0: 8 fixed cases, 48 actual teacher requests; no student parameter updates')
y=section(684,'Decision: stop this fixed input protocol before P1 / P2')
y=lines(y,['The finite screen is complete. Its evidence readback passed, but the candidate-ID',
'protocol did not qualify for expansion. This is not a general failure of MLLMs or OPD.',
'Teacher: STVG-R1-7B (c0db00d), frozen BF16/SDPA. No DINO, YOLO or SAM2.',
'P1, P2 and P3 were not run; CURRENT was not promoted.'],10)
y=section(y-20,'What was measured')
y=lines(y,['Each domain contributed four source-ranked cases: HC-STVG2 and VidSTG.',
'Per case: temporal and spatial comparison, each with base / swapped-ID / wrong-query.',
'Each request used eight fixed 448 x 448 marked frames and four student candidates.',
'Temporal: Native, Projection and two interval perturbations; fixed Native boxes.',
'Spatial: Native and three bounded box perturbations; fixed Projection interval.',
'GT was opened only after all 48 responses and source/runtime seals were complete.'])
y=section(y-20,'Output format and candidate-ID controls')
for label,count,desc in [('Original bare JSON + full schema',s['strict_bare_valid'],'No repairs; original parser retained'),('Exact fence removal + full schema',s['full_wrapper_schema_valid'],'Lossless wrapper removal only'),('Minimal legal top-choice readback',s['minimal_choice_valid'],'No missing field or candidate-ID repair')]:
 text(44,y,label,10,bold=True);text(300,y,f'{count} / 48',11,red,True);y-=15;text(44,y,desc,9,muted);y-=22
y=lines(y,['All outputs had JSON fences. Most omitted some required ranks/evaluations.',
'Invalid evidence-frame units were retained as errors, not silently converted.'],9,14)
y=section(y-15,'Swapping candidate IDs while preserving geometry')
headers=['Domain / branch','Same','Changed','Unavailable']
for x,h in zip([44,303,373,473],headers):text(x,y,h,8,muted,True)
y-=19
for g in s['groups']:
 label=('HC-STVG2' if g['dataset']=='hc2' else 'VidSTG')+' / '+g['branch']
 for x,t in zip([44,319,402,509],[label,str(g['ID_same']),str(g['ID_changed']),str(g['ID_unavailable'])]):text(x,y,t,10)
 y-=20
text(44,y-6,'9 / 12 comparable pairs agreed; 4 / 16 pairs had no comparable legal choice.',9,muted)
text(44,y-23,'Prelocked requirement: at least 6 / 8 same choices per domain; observed 5 / 8 and 4 / 8.',9,red)
c.showPage()
title(2,'Post-seal diagnostic quality, actual teacher cost and interpretation limits')
y=section(681,'Base-query choices versus matched Projection')
y=lines(y,['vIoU changes below use only cases with an actual legal teacher choice.',
'Unavailable choices are not scored as zero. Pool headroom uses all four cases per domain.'],9,14)
y-=10
for x,h in zip([44,220,295,414],['Domain / branch','Choices','Delta (pp)','Pool headroom (pp)']):text(x,y,h,8,muted,True)
y-=22
for g in s['groups']:
 vals=[('HC-STVG2' if g['dataset']=='hc2' else 'VidSTG')+' / '+g['branch'],f"{g['accepted_choices']} / 4",f"{g['delta_vIoU_pp_matched_Projection']:+.2f}",f"{g['pool_oracle_headroom_pp']:+.2f}"]
 for x,t in zip([44,233,308,461],vals):text(x,y,t,10,red if x==308 else ink)
 y-=23
y=lines(y-10,['Projection base means: HC-STVG2 27.85%; VidSTG 22.02%.',
'The VidSTG spatial choices (2 / 4) both selected the Native spatial trajectory.',
'No full-rank correlation is claimed: incomplete ranks are missing evidence.',
'192 candidate GT scores are diagnostic readouts, not 192 fits or complete teacher ratings.'],9,14)
y=section(y-22,'Cost of the 48 frozen-verifier requests')
a=s['cost'];y=lines(y,[f"Preprocess + generation total: {a['preprocess_plus_generate_seconds_total']:.2f} s; generation: {a['generate_seconds_total']:.2f} s.",f"Median request: {a['median_request_seconds']:.2f} s; range: {a['min_request_seconds']:.2f} - {a['max_request_seconds']:.2f} s.",f"Input tokens: {a['input_tokens_total']:,}; output tokens: {a['output_tokens_total']:,}.",f"Peak allocated GPU memory: {a['max_GPU_allocated_bytes']/2**30:.2f} GiB. Model load: {a['model_load_seconds']:.2f} s.",'No OOM or resource-guard stop occurred in the completed P0 run.',
'These are cached-input verifier costs, not end-to-end TTA latency.',
'Cold student generation, adaptation, final decode and artifact rendering are not all timed.'],10,15)
y=section(y-22,'What this screen can and cannot establish')
y=lines(y,['The spatial pool consists of nearby perturbations of one Native trajectory.',
'A selected pseudo-reference is a student candidate, not independent teacher geometry.',
'The pool cannot introduce a correct object absent from every candidate.',
'Some wrong-query responses identified absent referents in reviewed frames, but a',
'mismatched query is not automatically false; evidence unit errors remain unknown.',
'Four cases per domain are historically exposed, not a fresh unseen confirmation.',
'Teacher training overlap is not resolved. No statistical significance is claimed.',
'Zero new student fits / backward / optimizer steps were executed. OPD effect is unmeasured.',
'Future reopening requires a materially different, explicit input protocol and matched test.'],9,14)
y=section(y-17,'Evidence and reproducibility')
y=lines(y,['All 48 anonymous response rows, 192 candidate diagnostics, controls and timings are saved.',
'Official and independent metrics agree to a maximum absolute error of 1.23e-15.',
'Original raw outputs, sparse media and historical engineering revisions remain private.',
'Public code, protocol, negative / unknown results and root receipts accompany this report.'],9,14)
c.save();shutil.copy2(OUT,D/OUT.name)
print(OUT)
