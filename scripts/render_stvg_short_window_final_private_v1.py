"""Private actual QA-center readout contacts, only after full postseal scoring."""
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
B=R/'artifacts/stvg_short_event_tracklets_v1';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(B/'ROOT_POSTSEAL_REPORT.json')['status']=='pass'
 bar=read(B/'STUDENT_GLOBAL_BARRIER.json');man=read(B/'PREPARED_MANIFEST.json');cases={r['case_id']:load_source_case(R/r['source_case_path'])for r in man['cases']};truth,_=_truths_after_seal(cases);rows=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 for f in bar['files']:
  p=R/f['path'];assert sha(p)==f['sha256'];z=torch.load(p,map_location='cpu',weights_only=False);cid=z['case_id'];m=next(x for x in man['cases']if x['case_id']==cid);rgb=np.load(B/'private_rgb'/f'{cid}.npz');idx={int(v):i for i,v in enumerate(rgb['physical_frame_ids'])};grid={int(v):i for i,v in enumerate(z['frame_ids'])};canvas=Image.new('RGB',(1320,580),'white');draw=ImageDraw.Draw(canvas)
  draw.text((12,8),cid+' | Orange: Frozen; blue: local OPD; green: GT if annotated',font=font,fill='black')
  for j,c in enumerate(m['centers']):
   fid=int(c['center_physical_frame_id']);im=Image.fromarray(rgb['rgb_uint8'][idx[fid]]);im.thumbnail((320,245));w,h=im.size;x=10+330*j;y=48;dd=ImageDraw.Draw(im)
   for role,color in [('F','#db862f'),('local_R1','#2778b8')]:
    a=np.asarray(z['arms'][role]['boxes'])[grid[fid]];dd.rectangle(tuple((a*np.asarray([w,h,w,h])).tolist()),outline=color,width=3)
   gt=truth[cid]['truth'].get(fid)
   if gt is not None:
    a=np.asarray(gt)/[cases[cid]['width'],cases[cid]['height'],cases[cid]['width'],cases[cid]['height']];dd.rectangle(tuple((a*np.asarray([w,h,w,h])).tolist()),outline='#298651',width=2)
   canvas.paste(im,(x,y));draw.text((x,305),f'QA slot {j}; frame {fid}',font=font,fill='black');draw.text((x,330),'GT known'if gt is not None else'GT unknown',font=font,fill='black')
  yy=390
  for role in ['F','T','local_R1','T_local_R1']:
   draw.text((12,yy),role+': interval '+str(z['arms'][role]['interval']),font=font,fill='black');yy+=34
  out=B/'private_views'/f'{cid}_final_readout.png';canvas.save(out);rows.append(dict(case_id=cid,path=out.relative_to(R).as_posix(),sha256=sha(out)))
 (B/'PRIVATE_FINAL_CONTACT_EXPORT.json').write_text(json.dumps(dict(status='rendered_not_reviewed',count=8,center_slots=32,records=rows),indent=2)+'\n');print('8 private final contacts rendered')
if __name__=='__main__':main()
