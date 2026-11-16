"""PRIVATE all Top1 changes: actual views, canonical boxes, never selector feedback."""
from pathlib import Path
import sys,json,itertools,textwrap,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
R=Path(__file__).resolve().parents[2];H=R/'research/round105';O=R/'crossplatform/round105/atlas_PRIVATE';rd=lambda p:json.loads(Path(p).read_text());sys.path.insert(0,str(H));import geometry105 as geo
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
COL={'prepared_frozen':'#00c0ff','frozen_jointABC':'#cccccc','TTA_jointABC':'#ed32da','oracle':'#f1bc00'};EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]
def corners(b):return b[:3]+np.array(list(itertools.product([-1,1],repeat=3)))*b[3:6]/2
def main():
 O.mkdir(exist_ok=False);anchors={a['A']['metapath']:a for a in rd(H/'FULL_ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in rd(H/'FULL_PAIRS_PRIVATE.json')};index=[]
 for name in rd(R/'crossplatform/round105/MATRIX_SEALED.json')['directions']:
  src,_,tgt=name.split('_');lookup={r['id']:r for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')};golds=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json')
  for rr in rd(H/'scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json')):
   if rr['arms']['prepared_frozen']['query_PRIVATE']==rr['arms']['TTA_jointABC']['query_PRIVATE']:continue
   r=lookup[rr['id_PRIVATE']];ar=anchors[str(Path(r['metapath']).resolve())];assert ar['A']['eligible_for_extra_media'];z=np.load(R/'crossplatform/round105'/name/'samples_PRIVATE'/(r['id']+'.npz'));bs=z['source_boxes'];ps=dict(zip(z['arm_names'],z['scores']));gt=np.array(golds[r['anno_index']]['source_GT7_float32']);ys=many_iou(np.c_[bs,np.zeros(256)],gt);qs={k:int(ps[k].argmax()) for k in COL if k!='oracle'};qs['oracle']=int(ys.argmax());fig,axs=plt.subplots(2,3,figsize=(18,11));fig.subplots_adjust(top=.85,hspace=.28);case='CASE'+str(len(index)+1).zfill(4);fig.suptitle(case+' '+name+' — postseal all Top1 switches');fig.text(.04,.92,'\n'.join(textwrap.wrap(r['caption'],160)),fontsize=10);cam=geo.camera(ar['A']);im=np.asarray(Image.open(r['imagepath']).convert('RGB'));wh=(im.shape[1],im.shape[0]);uvs={q:geo.project_model_points(corners(bs[q]),cam,wh)['uv'] for q in set(qs.values())}
   def draw(ax,image,uv,title):
    ax.imshow(image)
    for label,q in qs.items():
     if q not in uv:continue
     for i,j in EDGES:
      if np.isfinite(uv[q][[i,j]]).all():ax.plot(uv[q][[i,j],0],uv[q][[i,j],1],color=COL[label],lw=1.2)
    ax.set_title(title);ax.axis('off')
   draw(axs[0,0],im,uvs,'A full RGB');draw(axs[0,1],im,uvs,'A selected-candidate context');v=np.concatenate(list(uvs.values()));v=v[np.isfinite(v).all(1)]
   if len(v):
    lo=np.maximum(v.min(0)-60,0);hi=np.minimum(v.max(0)+60,wh)
    if (hi>lo).all():axs[0,1].set_xlim(lo[0],hi[0]);axs[0,1].set_ylim(hi[1],lo[1])
   raw=geo.points(ar['A']);cloud=geo.raw_to_model_points(raw[::max(1,len(raw)//16000)],tgt,int(r.get('suffix') or 0));ax=axs[0,2];ax.scatter(cloud[:,0],cloud[:,1],s=.2,c='#aaa');selected=bs[list(qs.values())]
   for label,q in qs.items():
    b=bs[q];ax.add_patch(Rectangle(b[:2]-b[3:5]/2,b[3],b[4],fill=False,color=COL[label],lw=1.5))
   lo=(selected[:,:2]-selected[:,3:5]/2).min(0)-4;hi=(selected[:,:2]+selected[:,3:5]/2).max(0)+4;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title('Canonical A BEV, fixed boxes')
   for ax,key in [(axs[1,0],'B_pair'),(axs[1,1],'C_pair')]:
    pid=ar[key]
    if pid is None:ax.axis('off');ax.set_title(key+' unavailable');continue
    pair=pairs[pid];B=pair['B'];reg=rd(R/'crossplatform/round105/registration_full_PRIVATE'/(pid+'.json'));image=np.asarray(Image.open(B['imagepath']).convert('RGB'));uv={}
    if reg['geometry_cycle_supported']:
     for q in set(qs.values()):
      rawA=geo.transform_points(corners(bs[q]),geo.inverse_se3(geo.raw_to_model_transform(cam)));rawB=geo.transform_points(rawA,geo.inverse_se3(np.asarray(reg['forward']['T_raw_B_to_A'])));modelB=geo.raw_to_model_points(rawB,tgt,B['suffix']);uv[q]=geo.project_model_points(modelB,geo.camera(B),(image.shape[1],image.shape[0]))['uv']
    draw(ax,image,uv,key+' '+reg['status'])
   ax=axs[1,2];ax.axis('off');tab=[]
   for label,q in qs.items():tab.append([label,q,f'{ys[q]:.5f}']+[int(np.flatnonzero(np.argsort(-ps[k],kind='stable')==q)[0])+1 for k in ['prepared_frozen','frozen_jointABC','TTA_jointABC']])
   ax.table(cellText=tab,colLabels=['state','q','IoU','A rank','ABC rank','TTA rank'],loc='center',cellLoc='center',colWidths=[.34,.08,.15,.13,.13,.13]);ax.set_title('Same A256; oracle only posthoc');fig.text(.04,.03,'PRIVATE captions/media/coordinates. Projection and pair overlap are not identity truth. All Top1 changes retained.',fontsize=9);f=O/(case+'.jpg');fig.savefig(f,dpi=105);plt.close(fig);a,b=bs[qs['prepared_frozen']],bs[qs['TTA_jointABC']];inter=np.maximum(0,np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)-np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2)).prod();iou=float(inter/(a[3:].prod()+b[3:].prod()-inter));index.append({'case':case,'direction':name,'row_PRIVATE':r['id'],'caption_PRIVATE':r['caption'],'path_PRIVATE':str(f),'before_IoU':float(ys[qs['prepared_frozen']]),'after_IoU':float(ys[qs['TTA_jointABC']]),'before_after_box_IoU':iou,'identity_review':'UNKNOWN_until_viewed'});print('ATLAS105',case,name,flush=True)
 (O/'INDEX_PRIVATE.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n');print('ATLAS105_COMPLETE',len(index))
if __name__=='__main__':main()
