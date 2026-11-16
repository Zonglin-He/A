"""PRIVATE local good/bad atlas; projections are geometric evidence, not identity labels."""
from pathlib import Path
import json,sys,itertools,textwrap,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
R=Path(__file__).resolve().parents[2];H=R/'research/round112';O=R/'crossplatform/round112/atlas_PRIVATE';rd=lambda p:json.loads(Path(p).read_text());sys.path.insert(0,str(R/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points,raw_to_model_points
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
COL={'original_source':'#0066bb','real_frozen_fusedABC':'#111111','native_real_fusedABC':'#777777','noPA_correct_fusedABC':'#e07900','noEarlyRGB_correct_fusedABC':'#9c30b9','neither_correct_fusedABC':'#d43c2f','oracle':'#c9a000'};EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]
def corners(b):return b[:3]+np.array(list(itertools.product([-1,1],repeat=3)))*b[3:6]/2
def main():
 O.mkdir(exist_ok=False);cases=rd(H/'GOOD_FAILURE_CASES_PRIVATE.json');chosen=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   for kind in ['good','failure']:
    matches=[c for c in cases if c['source']==src and c['target']==tgt and c['kind']==kind and c['control']=='real_frozen_fusedABC']
    if matches:chosen.append(matches[0] | {'case':'CASE'+str(len(chosen)+1).zfill(4)})
 rows={t:{r['id']:r for r in rd(R/f'research/round99/input/TARGET_{t.upper()}128_PRIVATE.json')} for t in ['waymo','drone','quad']};gtindex={t:rd(R/f'research/round96/experiment/OFFICIAL_{t.upper()}_VAL_ORDER_PRIVATE.json') for t in rows};metadata=[]
 for c in chosen:
  src,tgt,sid=c['source'],c['target'],c['row_PRIVATE'];r=rows[tgt][sid];assert not rd(R/f'crossplatform/round112/{src}_to_{tgt}/records_PRIVATE/{sid}.json')['protected_alias_fallback'];f=R/f'crossplatform/round112/{src}_to_{tgt}/samples_PRIVATE/{sid}.npz';z=np.load(f);boxes=z['source_boxes'].astype(float);probs=dict(zip(z['arm_names'],z['scores']));gt=np.array(gtindex[tgt][r['anno_index']]['source_GT7_float32']);ys=many_iou(np.c_[boxes,np.zeros(256)],gt);qs={k:int(probs[k].argmax()) for k in COL if k!='oracle'};qs['oracle']=int(ys.argmax());rgb=np.array(Image.open(r['imagepath']).convert('RGB'));hh,ww=rgb.shape[:2];cam=CameraSpec.from_camera_fields(tgt,read_camera_allowlist(r['metapath']),waymo_view_suffix=int(r['suffix'] or 0),missing_distortion_policy='assume_zero_declared');uv={q:project_model_points(corners(boxes[q]),cam,(ww,hh))['uv'] for q in set(qs.values())}
  raw=(np.load(r['pointpath'],mmap_mode='r') if Path(r['pointpath']).suffix=='.npy' else np.fromfile(r['pointpath'],np.float32).reshape(-1,4));cloud=raw_to_model_points(raw[::max(1,len(raw)//20000),:3],tgt,int(r['suffix'] or 0));fig,axs=plt.subplots(2,2,figsize=(15,11));fig.subplots_adjust(top=.85,hspace=.23);fig.suptitle(c['case']+' '+src+'→'+tgt+' '+c['kind']+' — R112 posthoc diagnosis',fontsize=13);fig.text(.04,.92,'\n'.join(textwrap.wrap(r['caption'],125)),fontsize=10)
  alluv=np.concatenate(list(uv.values()));valid=alluv[np.isfinite(alluv).all(1)];crop=None
  if len(valid):
   lo=np.maximum(valid.min(0),[0,0]);hi=np.minimum(valid.max(0),[ww,hh]);size=np.maximum(hi-lo,[100,100]);crop=np.r_[np.maximum(lo-.15*size,0),np.minimum(hi+.15*size,[ww,hh])]
  for ax,title,zoom in [(axs[0,0],'Complete RGB',False),(axs[0,1],'All selected candidates + context',True)]:
   ax.imshow(rgb)
   for name,q in qs.items():
    for i,j in EDGES:
     if np.isfinite(uv[q][[i,j]]).all():ax.plot(uv[q][[i,j],0],uv[q][[i,j],1],color=COL[name],lw=1.2,ls='--' if name in ['frozen_C','lm_C'] else '-')
   if zoom and crop is not None and (crop[2:]>crop[:2]).all():ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1])
   ax.set_title(title);ax.axis('off')
  ax=axs[1,0];ax.scatter(cloud[:,0],cloud[:,1],s=.25,c='#aaaaaa',alpha=.4);bs=boxes[list(qs.values())];lo=np.min(bs[:,:2]-bs[:,3:5]/2,0)-5;hi=np.max(bs[:,:2]+bs[:,3:5]/2,0)+5
  for name,q in qs.items():
   b=boxes[q];ax.add_patch(Rectangle(b[:2]-b[3:5]/2,b[3],b[4],fill=False,color=COL[name],lw=1.3));ax.text(b[0],b[1],name,fontsize=6,color=COL[name])
  ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal',adjustable='box');ax.set_title('BEV, fixed yaw-zero candidate boxes')
  ax=axs[1,1];ax.axis('off');tab=[]
  for name,q in qs.items():
   rank={s:int(np.flatnonzero(np.argsort(-probs[s],kind='stable')==q)[0])+1 for s in ['real_frozen_fusedABC','native_real_fusedABC','noPA_correct_fusedABC','noEarlyRGB_correct_fusedABC','neither_correct_fusedABC']};tab.append([name,str(q),f'{ys[q]:.4f}']+[str(rank[s]) for s in rank])
  ax.table(cellText=tab,colLabels=['state','q','IoU','full rank','native rank','noPA rank','noRGB rank','neither rank'],loc='center',cellLoc='center',colWidths=[.25,.07,.1,.1,.1,.1,.1,.1]);ax.set_title('Ranks across identical A256; oracle only diagnostic')
  fig.text(.04,.035,'Colors: '+', '.join(COL)+'\nProjection/box overlap alone does not certify instance identity. Caption/media/coordinates PRIVATE; never publish.',fontsize=8);path=O/(c['case']+'.jpg');fig.savefig(path,dpi=100);plt.close(fig);metadata.append({'case':c['case'],'source':src,'target':tgt,'kind':c['kind'],'row_PRIVATE':sid,'caption_PRIVATE':r['caption'],'chosen_queries_PRIVATE':qs,'projection_in_image_corners':{name:int(project_model_points(corners(boxes[q]),cam,(ww,hh))['in_image'].sum()) for name,q in qs.items()},'identity_review':'UNKNOWN_until_local_visual_review','path_PRIVATE':str(path)});print('ATLAS112',c['case'],src,tgt,c['kind'],flush=True)
 (O/'INDEX_PRIVATE.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n');(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>PRIVATE R112 good/failure atlas</h1>'+''.join('<h2>'+m['case']+'</h2><p>'+html.escape(m['caption_PRIVATE'])+'</p><img style="max-width:100%" src="'+m['case']+'.jpg">' for m in metadata));print('ATLAS112_COMPLETE',len(metadata))
if __name__=='__main__':main()
