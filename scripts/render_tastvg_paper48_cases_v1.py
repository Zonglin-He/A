"""CPU-only fixed-hash qualitative frames; private media stays local."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scripts.tastvg_paper48_common_v1 import BASE,read,load,write

def run():
 import ijson
 from vg_tta.exact_frame_decode_audit_v2 import decode
 from scripts.run_tastvg_full_b1_experts_v1 import observation
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 assert read(BASE/'P1/COMPLETION.json')['status']=='completed'
 p=read(BASE/'P1/PLAN.json');chosen=read(BASE/'P4_PLAN.json')['parents'][:6];keys={p['rows'][i]['key'] for i in chosen}
 with (ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json').open('rb') as f:labels={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
 canvas=Image.new('RGB',(960,len(chosen)*216),(245,245,245));audit=[]
 for n,parent in enumerate(chosen):
  row=p['rows'][parent];arrival=p['orders']['order1'].index(parent);x=load(BASE/'P1/online/frame_drop_5/order1'/f'{arrival:05}.pt');frames,ids=decode(row['input']);shifted,pixel,_=observation(row,'frame_drop_5',frames);assert pixel==x['pixel_sha256'];g=labels[row['key']];positions=np.rint(np.linspace(0,len(ids)-1,3)).astype(int).tolist()
  for col,pos in enumerate(positions):
   im=Image.fromarray(shifted[pos]).resize((320,180));d=ImageDraw.Draw(im)
   for arm,b,idx,color in [('Frozen',x['source_native']['boxes'][pos],x['source_native']['indices'],'orange'),('Ours',x['slow']['boxes'][pos],x['final_indices'],'red')]:
    if idx[0]<=pos<=idx[1]:d.rectangle(xyxy([b.numpy()],320,180)[0].tolist(),outline=color,width=2)
   if g['valid'][pos]:d.rectangle(xyxy([g['boxes'][pos]],320,180)[0].tolist(),outline='lime',width=2)
   canvas.paste(im,(col*320,n*216+30))
  d=ImageDraw.Draw(canvas);d.text((8,n*216+8),f'S{parent+1:04} | frame_drop5 | orange Frozen, red Ours, green GT | frames '+','.join(str(ids[j]) for j in positions),fill='black');audit.append(dict(parent=parent,arrival=arrival,frame_ids=[ids[j] for j in positions],condition='frame_drop_5',order='order1'))
 canvas.save(BASE/'PRIVATE_QUALITATIVE.png');write(BASE/'QUALITATIVE_AUDIT.json',dict(status='completed',selection='first6 from locked P4 source hash; three uniform input positions; no score-based selection',cases=audit,private_media_not_for_public_export=True,CPU_only=True))
if __name__=='__main__':run()
