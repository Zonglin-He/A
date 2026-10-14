"""Private actual-pixel readback of every fixed teacher-selected anchor reference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from scripts.stvg_temporal_caption_query_v1 import ROOT,read,roster,catalog
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
B=ROOT/'artifacts/stvg_box_scale_global_evidence_v1';assert read(B/'TEACHER_GLOBAL_BARRIER.json')['status']=='complete' and read(B/'source_position_control001/TEACHER_GLOBAL_BARRIER.json')['status']=='complete';records=read(B/'GRID_INPUT_BARRIER.json')['records'];base={r['case']:r for r in read(B/'ROOT_PRELIMINARY_POSTSEAL_METRICS.json')['rows']};ctrl={r['case']:r for r in read(B/'source_position_control001/ROOT_READBACK.json')['rows']};cases={r['case_id']:catalog(r)[0]for r in roster()};truth,_=_truths_after_seal(cases);out=Image.new('RGB',(1792,1000),'white');drawer=ImageDraw.Draw(out)
for i,r in enumerate(records):
 cid=r['case_id'];alias=('HC'if cid.startswith('hc2')else 'V')+f"{int(cid.split('_')[-1])+1:02d}";canvas=Image.open(B/'private_inputs'/cid/'original.png').convert('RGB');d=ImageDraw.Draw(canvas);W,H=r['source_wh'];scale=min(448/W,448/H);rw,rh=round(W*scale),round(H*scale);ox,oy=(448-rw)//2,(448-rh)//2
 def rect(cx,cy,w,h,color):d.rectangle([ox+(cx-w/2)*rw,oy+(cy-h/2)*rh,ox+(cx+w/2)*rw,oy+(cy+h/2)*rh],outline=color,width=2)
 rect(*r['grid']['student_baseline']['cxcywh'],'cyan')
 for protocol,color in [('mask','orange'),('marked','magenta')]:
  keep=ctrl[alias][protocol+'_grid_plus_source_keeps_source'];id=base[alias]['mask_selected_window_id'if protocol=='mask'else 'marked_selected_window_id'];w=next(w for w in r['grid']['windows']if w['window_id']==id)
  if keep:rect(*r['grid']['student_baseline']['cxcywh'],color)
  else:rect(w['cx'],w['cy'],w['w'],w['h'],color)
 gt=truth[cid]['truth'].get(r['physical_frame_id'])
 if gt is not None:d.rectangle([ox+gt[0]/W*rw,oy+gt[1]/H*rh,ox+gt[2]/W*rw,oy+gt[3]/H*rh],outline='lime',width=3)
 x,y=i%4*448,i//4*500;out.paste(canvas,(x,y));drawer.text((x+8,y+454),alias+' | cyan Source / orange Mask / magenta Mark',fill='black');drawer.text((x+8,y+474),'green GT'if gt is not None else 'GT unknown at fixed anchor',fill='black')
p=B/'private_views/selected_reference_all_contact.png';out.save(p);print(str(p))
