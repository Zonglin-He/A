"""Private actual-frame currentMAP illustrations, post-hoc selection disclosed."""
from common import *
import argparse
Q=D/'qualitative'
def prepare():
 p=freeze('qualitative',dict(cases=3,selection='Post-hoc illustrations from complete sealed TA MAP scalars: HC largest v gain with positive unobserved gain and both t/v gains; Vid largest t gain under sameconditions; Vid most negative v gain. Deterministic ID ties. Extremes, not representative frequency.',figure='Each case6actualdecodedframes with2observed+4unobserved where available; all8observedmarkers on timeline, nativeSource/finalMAP eventintervals; Sourceblue/Fullred rectangles. No GTboxoverlay, no new scoring/model/teacher.',privacy='All RGB/captions/coordinates remain private; public selection hashes/scalar changes/protocol only. Root must view actualPNG before marking figure visuallychecked.'),[Path(__file__),S/'common.py']);print(p['fingerprint'])
def run():
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from stvg.decode import decode
 p=validate('qualitative');assert not(Q/'render_receipt.json').exists();cases=[]
 for ds,mode in [('hc2','v_gain'),('vidstg','t_gain'),('vidstg','v_harm')]:
  rr=read(MAP/'TA'/ds/'metrics.json')['private_rows'];good=[]
  for x in rr:
   v=x['values'];delta={k:v['FullMAP'][k]-v['Source'][k]if v['FullMAP'][k]is not None and v['Source'][k]is not None else None for k in ['t','v','unobserved_actual_sIoU']}
   if mode.endswith('gain')and not(delta['t']>0 and delta['v']>0 and delta['unobserved_actual_sIoU']is not None and delta['unobserved_actual_sIoU']>0):continue
   score=delta['t']if mode=='t_gain'else delta['v'];good.append((score,x['id'],delta))
  chosen=sorted(good,key=lambda z:((-z[0]if mode!='v_harm'else z[0]),z[1]))[0];cases.append(dict(dataset=ds,id=chosen[1],selection=mode,delta=chosen[2]))
 Q.mkdir(parents=True,exist_ok=True);fig=plt.figure(figsize=(18,13),layout='constrained');outer=fig.add_gridspec(6,6,height_ratios=[3,1,3,1,3,1]);private=[]
 for ci,c in enumerate(cases):
  ds=c['dataset'];row=next(x for x in records(ds)if x['id']==c['id']);frames,ids=decode(row['metadata'],ds);path,h=current_MAP(ds,c['id'])
  with np.load(path)as z:src=z['Source'].copy();full=z['FullMAP'].copy();pos=z['positions8'].tolist();iv0=z['Source_interval'].tolist();iv1=z['FullMAP_interval'].tolist();assert z['frame_ids'].tolist()==ids
  unobs=[i for i in range(len(ids))if i not in pos];observed=[pos[0],pos[-1]];unselected=[unobs[i]for i in np.linspace(0,len(unobs)-1,4).round().astype(int)];shown=sorted(set(observed+unselected));assert len(shown)==6
  for col,t in enumerate(shown):
   ax=fig.add_subplot(outer[ci*2,col]);ax.imshow(frames[t]);H,W=frames[t].shape[:2]
   for b,color,name in [(src[t],'deepskyblue','Source'),(full[t],'orangered','FullMAP')]:
    x,y,w,hh=b;ax.add_patch(Rectangle(((x-w/2)*W,(y-hh/2)*H),w*W,hh*H,fill=False,edgecolor=color,lw=2,label=name))
   ax.set_title(f"{ds} case{ci+1}, frame {ids[t]}\n"+('Expert observed'if t in pos else'Unobserved'),fontsize=10);ax.axis('off')
   if col==0:ax.legend(fontsize=8,loc='lower left')
  ax=fig.add_subplot(outer[ci*2+1,:]);fps=row['metadata']['fps'];origin=ids[0];conv=lambda x:(np.asarray(x)-origin)/fps
  ax.hlines(0,*conv([ids[0],ids[-1]+1]),color='gray',lw=1);ax.scatter(conv([ids[t]for t in pos]),np.zeros(8),color='purple',marker='v',label='Eight expert observations',s=40)
  ax.broken_barh([(conv([iv0[0]])[0],(iv0[1]-iv0[0])/fps)],(.3,.25),facecolors='deepskyblue',label='Native Source interval');ax.broken_barh([(conv([iv1[0]])[0],(iv1[1]-iv1[0])/fps)],(.7,.25),facecolors='orangered',label='Final MAP interval');ax.set_yticks([]);ax.set_xlabel('Seconds from first sampled physical frame');ax.set_ylim(-.15,1.1);ax.set_title(f"Post-hoc {c['selection']}; Δt={100*c['delta']['t']:+.2f}pp, Δv={100*c['delta']['v']:+.2f}pp; blue/red boxes are predictions, no GT overlay",fontsize=10)
  if ci==0:ax.legend(loc='upper right',ncol=3,fontsize=8)
  private.append({**c,'prediction_path':str(path),'prediction_sha256':h,'positions8':pos,'shown_positions':shown,'frame_ids':ids,'native_interval':iv0,'FullMAP_interval':iv1})
 fig.suptitle('Eight observations, unobserved-frame trajectory changes and event boundaries',fontsize=16);fig.savefig(Q/'current_MAP_qualitative.png',dpi=150);fig.savefig(Q/'current_MAP_qualitative.pdf');plt.close(fig);atomic(Q/'selection.json',private)
 out=dict(status='rendered_private_pending_root_visual_inspection',cases=3,selection='Post-hoc complementary gain extremes plus one harm, not a representative sample',PNG_sha256=sha(Q/'current_MAP_qualitative.png'),PDF_sha256=sha(Q/'current_MAP_qualitative.pdf'),new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0,fingerprint=p['fingerprint']);atomic(Q/'render_receipt.json',out);atomic(P/'qualitative/render_receipt.json',{**out,'case_scalar_deltas':[{k:v for k,v in c.items()if k!='id'}for c in cases]});print(out)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);z=a.parse_args();dict(prepare=prepare,run=run)[z.command]()
