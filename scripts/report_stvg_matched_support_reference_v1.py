"""Anonymous fixed-support diagnostic tables and private actual-pixel contacts."""
import json,hashlib,statistics
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];B=R/'artifacts/stvg_matched_support_reference_v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ROLES=['source','r1','pool_oracle','gt_oracle'];LABELS=['Source ref.','R1 ref.','Pool Oracle*','GT Box Oracle*']
def alias(c):return ('HC'if c.startswith('hc2')else'V')+f'{int(c[-3:])+1:02d}'
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 d=read(B/'ROOT_POSTSEAL_REPORT.json');assert d['status']=='pass'and d['metric_count']==80
 assert read(B/'ROOT_POSTSEAL_INDEPENDENT_READBACK.json')['status']=='pass'
 groups=d['groups'];rows=d['rows'];refs=read(B/'REFERENCE_LOCK.json');bar=read(B/'STUDENT_GLOBAL_BARRIER.json')
 table=['| Domain | Reference | Spatial delta vs Frozen (pp) | T+S delta vs T (pp) | S positive / negative / unchanged |','|---|---|---:|---:|---:|']
 for domain in ['hc2','vidstg']:
  for role in ROLES:
   g=next(g for g in groups if g['dataset']==domain and g['arm']==role);t=next(g for g in groups if g['dataset']==domain and g['arm']=='T_'+role)
   table.append(f"| {domain} | {role} | {g['mean_delta_pp']:+.6f} | {t['mean_delta_pp']:+.6f} | {g['positive']}/{g['negative']}/{g['neutral']} |")
 per=['| Historical source | Support | Source ref. | R1 ref. | Pool Oracle | GT Box Oracle |','|---|---:|---:|---:|---:|---:|']
 for ref in refs['records']:
  cid=ref['case_id'];a=[next(x for x in rows if x['case_id']==cid and x['arm']==k)for k in ROLES]
  per.append('| '+alias(cid)+' | '+str(len(ref['references']['source']['anchors']))+' | '+' | '.join(f"{v['delta_vIoU_pp']:+.6f}"for v in a)+' |')
 text='''# Matched-support spatial references: candidate, selection and transfer diagnosis

## Scope and evidence

Eight historically exposed sources, four per direction; 32 old QA centers. Nineteen centers have all displayed candidate boxes, Source, a real annotation, and a valid non-abstaining sealed R1 choice. These 19 centers form identical ordered support for all four reference roles. Thirteen centers are excluded by availability, never by the sign or size of IoU. HC01 has zero support and exact F/T reuse. No missing annotation is filled or interpolated into supervision.

Source and R1 are separately fitted, including R1's Source choices (17/19 anchors). Candidate-Pool Oracle selects the largest center IoU among already displayed candidates; GT Box Oracle uses the true sparse annotation. Both Oracles are explicitly label-assisted offline diagnostics. GT availability is also used to define this diagnostic support. These are not deployable results or independent confirmation. The pool maximum at sparse centers is not an Oracle for full-Tube vIoU after adaptation.

All nonempty roles use the same source checkpoint, identical initial 1792 Query/LN coordinates, original domain config, same Gaussian key, five original rounds and fresh Adam. All 28 fits, 140 actual updates and 28 final full-model redecodes completed. Eight source captures completed in the successful run; one additional prefit capture belongs to a preserved engineering stop. No new teacher call, candidate generation or temporal training was used in this comparison. F and old T are exactly reused. The five cases with identical Source/R1 anchors have separately executed fits whose saved dynamics and output match exactly.

## Full-Tube results

Domain means include all four sources, including HC01 no-update. T+S combines the newly decoded spatial boxes with the exact old T interval. The two columns have different paired baselines and must not be interchanged.

'''+ '\n'.join(table)+'\n\nPer-source Spatial-only changes from Frozen (pp):\n\n'+'\n'.join(per)+'''

## What the controlled comparison establishes

GT Box Oracle improves all seven nonempty sources. The same update interface therefore has demonstrated capacity under correct sparse geometry on this fixed panel. Candidate-Pool Oracle improves only two sources and degrades five; its domain gains are much smaller. This supports a candidate-reference quality/transfer bottleneck relative to the GT control, not a claim that OPD can never work.

Pool selection loses available benefit to R1 in HC03/HC04, but geometry and transfer also matter. V02's pool reference is fitted more closely (about +20.10 pp mean reference IoU at supervised positions), while its full-Tube score changes by about -0.43 pp. Better fitting of a candidate target is therefore not task improvement. The locally highest-IoU candidate is not guaranteed to be the most useful sparse target for the full trajectory. V04 remains a positive real-reference example.

Source-reference self-distillation itself moves parameters and is not a zero-update baseline. Its average losses must remain visible. GT Oracle fitting and complete-Tube gain do not establish a generally correct gradient for an arbitrary pseudo-reference. No learning-rate, temperature, amplitude, best step or prompt was selected from these labels.

The old V01/V03 harmful anchors had no GT and are excluded from this common-support comparison. Their new Source/R1 equality does not prove that the old anchors were correct or incorrect. Sparse reference IoU, reference-fitting progress, supervised/unsupervised geometry and full-Tube vIoU are reported separately. Window counts are not independent sample counts.

## One conditional candidate test

The GT-versus-pool gap supports the already proposed single local visual-motion region test. It is a new candidate-construction mechanism only: camera-compensated residual motion, connected regions and reciprocal short-window matching on the existing cache. It uses no new detector/segmenter network, no copied non-Source grid seeds and no new frames. Candidate outputs must be sealed before the fixed offline coverage comparison. R1/OPD is conditional on actual improvement, and independent >=32/source-per-direction confirmation remains untriggered until a fixed GT-free spatial rule is supported. No fourth tracker or parameter sweep is authorized by this report.

## Readback, cost and limits

The CPU reader independently checks source/config/hash/frame identity, matched support, samples, rewards, likelihood, distillation weights, stored gradients, Adam arithmetic and final cached-versus-full decode. It does not independently run decoder backward. Metrics are independently recomputed with separate scalar arithmetic using the same official label loader. All 80 metrics, 20 groups, 76 sparse-reference qualities, 28 movement/fit-progress records and all 32 support decisions are preserved, including negative, neutral and unknown entries.

The worker directly opens no annotation file but consumes GT-derived reference locks; the overall experiment is GT-assisted. Successful worker wall time is '''+f"{bar['end_to_end_seconds']:.6f}"+''' seconds (includes source capture, fitting and saved IO, excludes independent review/publication and the earlier failed launch). This is diagnostic cost, not production end-to-end TTA latency. Preparation receipt-schema issues and the first prefit pixel-hash scope mismatch were preserved and resolved without changing support, references or science. CURRENT and stopped historical queues are unchanged.
'''
 (R/'docs/STVG_MATCHED_SUPPORT_REFERENCE_V1.md').write_text(text)
 (B/'figures').mkdir(exist_ok=True);pdf=B/'figures/matched_support_reference_v1.pdf'
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
 with PdfPages(pdf)as pp:
  fig,axs=plt.subplots(1,2,figsize=(11.5,5.8))
  for ax,domain in zip(axs,['hc2','vidstg']):
   x=np.arange(4);a=[next(g['mean_delta_pp']for g in groups if g['dataset']==domain and g['arm']==k)for k in ROLES];t=[next(g['mean_delta_pp']for g in groups if g['dataset']==domain and g['arm']=='T_'+k)for k in ROLES]
   ax.bar(x-.18,a,.36,label='S minus Frozen',color='#426b9b');ax.bar(x+.18,t,.36,label='T+S minus T',color='#53958a');ax.axhline(0,c='#444',lw=.7);ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set_title(domain+' | 4 historical sources');ax.set_ylabel('Full-Tube vIoU change (pp)');ax.legend(fontsize=9)
  fig.suptitle('Same support and optimizer; only reference geometry changes');fig.text(.07,.025,'* Label-assisted offline diagnostics. 19 shared centers, 28 actual fits; HC01 has no support.',fontsize=9);fig.tight_layout(rect=[0,.06,1,.95]);pp.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(11.5,7));ax.axis('off');data=[]
  for ref in refs['records']:
   c=ref['case_id'];data.append([alias(c),len(ref['references']['source']['anchors'])]+[f"{next(x['delta_vIoU_pp']for x in rows if x['case_id']==c and x['arm']==k):+.3f}"for k in ROLES])
  tab=ax.table(cellText=data,colLabels=['Source','Support',*LABELS],cellLoc='center',loc='center');tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2);ax.set_title('All source-level Spatial-only effects versus Frozen',pad=15);fig.text(.06,.1,'GT improves 7/7 nonempty cases; Pool Oracle improves 2/7. These are exposed development sources.',fontsize=10);fig.text(.06,.06,'Old GT-unknown harmful anchors in V01/V03 are not evaluated by the new matched support.',fontsize=10);fig.tight_layout(rect=[0,.12,1,.95]);pp.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(11.5,7));colors=['#888','#bc7848','#547ba4','#498a72']
  for k,col,label in zip(ROLES,colors,LABELS):
   rr=[r for r in d['fit_movement']if r['role']==k];xx=[r['reference_IoU_delta_pp']for r in rr];yy=[next(v['delta_vIoU_pp']for v in rows if v['case_id']==r['case_id']and v['arm']==k)for r in rr];ax.scatter(xx,yy,label=label,c=col,s=45)
   for r,x,y in zip(rr,xx,yy):
    if r['case_id']in ('vidstg_001','vidstg_003'):ax.annotate(alias(r['case_id']), (x,y),xytext=(4,3),textcoords='offset points',fontsize=8)
  ax.axhline(0,c='#777',lw=.7);ax.axvline(0,c='#777',lw=.7);ax.set(xlabel='Mean IoU change toward the supplied sparse reference (pp)',ylabel='Full-Tube vIoU change from Frozen (pp)',title='Fitting the supplied reference is distinct from improving the task');ax.legend();fig.text(.06,.025,'Different supports from the earlier one-anchor experiment; no GT-based parameter selection.',fontsize=9);fig.tight_layout(rect=[0,.05,1,.97]);pp.savefig(fig);plt.close(fig)
 save(B/'FIGURE_EXPORT.json',dict(status='rendered_pending_pixel_review',pages=3,pdf_sha256=sha(pdf),report_sha256=sha(R/'docs/STVG_MATCHED_SUPPORT_REFERENCE_V1.md'),metric_count=80))
 private_contacts()
 print('report, three-page PDF and eight private contacts rendered')
def private_contacts():
 import torch
 from PIL import Image,ImageDraw,ImageFont
 old=R/'artifacts/stvg_short_event_tracklets_v1';manifest=read(old/'PREPARED_MANIFEST.json');refs=read(B/'REFERENCE_LOCK.json');bar=read(B/'STUDENT_GLOBAL_BARRIER.json');refmap={r['case_id']:r for r in refs['records']};out=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for f in bar['files']:
  p=R/f['path'];assert sha(p)==f['sha256'];z=torch.load(p,map_location='cpu',weights_only=False);cid=z['case_id'];m=next(x for x in manifest['cases']if x['case_id']==cid);cache=np.load(old/'private_rgb'/f'{cid}.npz');idx={int(v):i for i,v in enumerate(cache['physical_frame_ids'])};grid={int(v):i for i,v in enumerate(z['frame_ids'])};canvas=Image.new('RGB',(1200,980),'white');dd=ImageDraw.Draw(canvas)
  dd.text((10,7),alias(cid)+' | orange Frozen; blue adapted; green supplied reference (GT row is diagnostic)',fill='black',font=font)
  for row,role in enumerate(ROLES):
   anchors={a['physical_frame_id']:a for a in refmap[cid]['references'][role]['anchors']}
   for col,center in enumerate(m['centers']):
    fid=center['center_physical_frame_id'];im=Image.fromarray(cache['rgb_uint8'][idx[fid]]);im.thumbnail((290,205));w,h=im.size;draw=ImageDraw.Draw(im)
    for arm,color in [('F','#e79124'),(role,'#287dcc')]:draw.rectangle(tuple((np.asarray(z['arms'][arm]['boxes'])[grid[fid]]*[w,h,w,h]).tolist()),outline=color,width=2)
    a=anchors.get(fid)
    if a:
     cx,cy,bw,bh=a['box_cxcywh'];draw.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),outline='#26b54a',width=2)
    x=col*300+5;y=row*235+35;canvas.paste(im,(x,y));dd.text((x,y+208),role+f' / slot {col} / '+('supported'if a else'excluded'),fill='black',font=font)
  p=B/'private_views'/f'{alias(cid)}.png';p.parent.mkdir(exist_ok=True);canvas.save(p);out.append(dict(case_id=alias(cid),path=p.relative_to(R).as_posix(),sha256=sha(p)))
 save(B/'PRIVATE_CONTACT_EXPORT.json',dict(status='rendered_pending_pixel_review',count=8,rows_per_case=4,center_slots_per_case=4,records=out))
if __name__=='__main__':main()
