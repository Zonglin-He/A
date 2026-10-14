"""Private actual candidate-pixel contact; never part of public bundle."""
from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];B=R/'artifacts/stvg_video_conditioned_global_referent_v1';z=json.loads((B/'CANDIDATE_INPUT_BARRIER.json').read_text());dst=B/'private_views';dst.mkdir(exist_ok=True)
for index,r in enumerate(z['records']):
 alias=('HC'if r['dataset']=='hc2'else'V')+f'{index%4+1:02d}';n=len(r['candidates']);can=Image.new('RGB',(960,36+n*256),'white');d=ImageDraw.Draw(can);font=ImageFont.truetype('DejaVuSans.ttf',13);d.text((8,8),alias+' | actual four-frame hypotheses | no GT shown | rejected '+str(len(r['rejected'])),font=font,fill='black')
 for row,c in enumerate(r['candidates']):
  d.text((8,36+row*256),f"ID {c['candidate_id']} / Source"if c['window_id']==0 else f"ID {c['candidate_id']} / window {c['window_id']}",font=font,fill='black');a=np.load(B/'private_inputs'/r['case_id']/f"candidate{c['candidate_id']}_ID{c['candidate_id']}.npy",allow_pickle=False)
  for k,frame in enumerate(a):can.paste(Image.fromarray(frame.transpose(1,2,0)).resize((240,240),Image.Resampling.LANCZOS),(k*240,52+row*256))
 can.save(dst/f'{alias}_candidate_contact.png')
print('8 actual private contacts generated')
