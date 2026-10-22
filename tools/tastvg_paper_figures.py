"""Prepare private continuous-frame sheets; do not publish dataset media."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tastvg_paper_full import ROOT,PROVIDER,RUN,PUBLIC,read,atomic,check

def main():
 sys.path.insert(0,str(PROVIDER));from stvg.decode import decode
 p=check();out=RUN/'qualitative';selection=[];names=['Source','SharedInit','Propagation+Full']
 for ds,j in p['jobs'].items():
  report=read(RUN/ds/'metrics.json');rows=[json.loads(s)for s in(PROVIDER/j['manifest']).read_text().splitlines()];by={r['id']:r for r in rows};scored=report['private_rows'];rank=lambda key:hashlib.sha256(('paper-qual-20261022|'+key).encode()).hexdigest()
  # Candidate pools, not confirmed semantic categories. Root verifies actual frames.
  pools={
   'scale_change_candidates':[r for r in scored if r['values']['OursS']['v']>r['values']['Source']['v']],
   'pose_or_occlusion_candidates':[r for r in scored if any(w in by[r['id']]['metadata']['caption'].lower()for w in ('turn','sit','bend','behind','stand'))],
   'similar_object_candidates':[r for r in scored if any(w in by[r['id']]['metadata']['caption'].lower()for w in ('another','other','two','people','woman','man'))],
   'failure_candidates':[r for r in scored if r['values']['OursS']['v']<r['values']['Source']['v']-.05]}
  for category,pool in pools.items():
   for item in sorted(pool,key=lambda x:rank(x['id']))[:2]:
    rec=by[item['id']];frames,ids=decode(rec['metadata'],ds);f=RUN/ds/'predictions'/(item['id']+'.npz')
    with np.load(f,allow_pickle=False)as z:
     pos=list(map(int,z['positions8']));center=pos[len(pos)//2];start=max(0,min(center-3,len(ids)-6));nodes=list(range(start,start+6));W=256;H=int(frames.shape[1]*W/frames.shape[2]);canvas=Image.new('RGB',(W*6,(H+28)*3),'white');draw=ImageDraw.Draw(canvas)
     for row,name in enumerate(names):
      for col,t in enumerate(nodes):
       image=Image.fromarray(frames[t]).resize((W,H));canvas.paste(image,(col*W,row*(H+28)+28));d=ImageDraw.Draw(canvas);boxes=[(z['Source'][t],'red')]if row==0 else[(z['SharedInit'][t],'orange')]if row==1 else[(z['complete_boxes'][t,s],'cyan')for s in (0,1)if z['complete_available'][t,s]]+[(z['Full'][t],'lime')]
       if t in pos:boxes.append((z['native_boxes'][pos.index(t)],'magenta'))
       for b,color in boxes:
        cx,cy,bw,bh=b;x0=(cx-bw/2)*W+col*W;x1=(cx+bw/2)*W+col*W;y0=(cy-bh/2)*H+row*(H+28)+28;y1=(cy+bh/2)*H+row*(H+28)+28;d.rectangle([max(col*W,x0),max(row*(H+28)+28,y0),min((col+1)*W-1,max(col*W,x1)),min((row+1)*(H+28)-1,max(row*(H+28)+28,y1))],outline=color,width=2)if x0<(col+1)*W and x1>=col*W and y0<(row+1)*(H+28)and y1>=row*(H+28)+28 else None
       d.text((col*W+3,row*(H+28)+5),f'{name} / node {t}',fill='black')
     target=out/ds/(category+'_'+rank(item['id'])[:8]+'.png');target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target);selection.append(dict(dataset=ds,category=category,query_id=item['id'],path=str(target.relative_to(RUN)),nodes=nodes,requires_root_visual_review=True))
 atomic(out/'manifest.json',dict(status='private_candidate_sheets_ready',items=selection,not_published_media=True,final_groups_required=4,root_review_required=True));atomic(PUBLIC/'qualitative_status.json',dict(status='private_candidate_sheets_ready_for_root_review',candidate_sheets=len(selection),final_groups_target=4,raw_dataset_media_exported=False))

if __name__=='__main__':main()
