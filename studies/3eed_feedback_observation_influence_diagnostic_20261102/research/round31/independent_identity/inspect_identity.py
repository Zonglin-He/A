"""CPU-only visible identity and point support; no task GT or IoU loaded."""
import sys,json,hashlib,textwrap,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round27'),str(ROOT/'crossplatform/round17')]
from common30 import read,dump,sha,seal,feature,rows
from observations27 import load
from camera_projection import project_model_points
from association import box_corners
from PIL import Image,ImageDraw,ImageFont
OUT=ROOT/'research/round31/independent_identity';V=OUT/'private_views'
COLORS={0:(240,30,30),1:(20,200,30),5:(40,100,245)}
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
BIG=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',23)
def decorate(im,caption,title):
 panel=Image.new('RGB',(1280,1030),'white');panel.paste(im,(0,45));d=ImageDraw.Draw(panel);d.text((8,8),title,fill='black',font=BIG)
 for j,s in enumerate(textwrap.wrap(caption,width=105)):d.text((8,865+24*j),s,fill='black',font=FONT)
 return panel
r=next(r for r in rows() if r['alias']=='A_drone_034');z=feature(r);q0=int(z['scores'].argmax());pred=read(ROOT/'research/round30/experiment/SELECTED_BETA_LOCK.json');fp=ROOT/'crossplatform/round30/screen_attempt001/drone_b1_A/predictions_PRIVATE'/(r['id']+'.npz')
with np.load(fp) as p:q1=int(p['query']);adapted=p['scores'].copy()
frame,scene,bind=load(r);rgb=frame['rgb'];im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
for m,o in enumerate(scene['objects']):
 bb=list(map(int,o['detector_box_xyxy']));col=COLORS.get(m,(235,0,220));d.rectangle(bb,outline=col,width=3);yy=max(0,bb[1]-28);d.rectangle((bb[0],yy,bb[0]+70,yy+28),fill='white');d.text((bb[0]+3,yy),'M'+str(m),fill='black',font=BIG)
decorate(im,r['caption'],'All visible detector instances | no taskGT / IoU or matching scores').save(V/'IDENTITY_ALL_MASKS.png')
# Predicted boxes and geometrically associated points. No masks or boxes altered.
points=np.asarray(scene['visible_points_model']);owner=np.asarray(scene['point_owner']);uv=np.asarray(scene['visible_uv']);stats=[]
for label,q in [('source',q0),('R30keep',q1)]:
 b=z['boxes'][q];inside=(np.abs(points-b[:3])<=b[3:]/2).all(1);counts={str(m):int(np.sum(inside&(owner==m))) for m in range(len(scene['objects']))};known=int(np.sum(inside&(owner>=0)));alln=int(inside.sum());ims=Image.fromarray(rgb)
 arr=np.array(ims);overlay=arr.copy()
 for m in [0,1,5]:overlay[scene['objects'][m]['mask_bool']]=COLORS[m]
 ims=Image.fromarray((.78*arr+.22*overlay).astype(np.uint8));draw=ImageDraw.Draw(ims)
 for m in [0,1,5]:
  bb=list(map(int,scene['objects'][m]['detector_box_xyxy']));draw.rectangle(bb,outline=COLORS[m],width=3);draw.text((bb[0]+3,max(0,bb[1]-28)),'M'+str(m),fill=COLORS[m],font=BIG,stroke_width=1,stroke_fill='white')
 for (u,v),m in zip(uv[inside],owner[inside]):
  col=COLORS.get(int(m),(250,230,20)) if m>=0 else (220,220,220);u,v=int(round(u)),int(round(v));draw.ellipse((u-2,v-2,u+2,v+2),fill=col)
 proj=project_model_points(box_corners(b),scene['camera_spec'],scene['image_size_wh']);xy=proj['uv'];rect=np.r_[xy.min(0),xy.max(0)];draw.rectangle(list(map(int,rect)),outline=(255,165,0),width=4);draw.text((int(rect[0]),max(0,int(rect[1])-54)),label+' predicted AABB',fill=(255,110,0),font=BIG,stroke_width=1,stroke_fill='black')
 decorate(ims,r['caption'],label+' | RGB masks and visible LiDAR points inside predicted box').save(V/(label+'_POINT_SUPPORT.png'));stats.append({'prediction':label,'query_PRIVATE':q,'box_PRIVATE':b.tolist(),'known_foreground_points':known,'all_visible_points_inside':alln,'mask_point_counts':counts,'foreground_share_by_mask':{m:n/max(known,1) for m,n in counts.items()},'projected_box_xyxy_PRIVATE':rect.tolist(),'unassigned_points':alln-known,'matching_score':float(adapted[q]) if label=='R30keep' else float(z['scores'][q])})
# Shared groundplane overview; masks color only assigned points.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,7))
ax.scatter(points[:,0],points[:,1],s=1,c='lightgray',alpha=.5)
for m,col in COLORS.items():
 p=points[owner==m];ax.scatter(p[:,0],p[:,1],s=8,c=[np.array(col)/255],label='M'+str(m))
for label,q,col in [('source',q0,'green'),('R30keep',q1,'red')]:
 b=z['boxes'][q];xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*b[None,3:5]/2+b[None,:2];ax.plot(xy[:,0],xy[:,1],c=col,lw=2,label=label)
ax.set_xlim(min(z['boxes'][q0,0],z['boxes'][q1,0])-8,max(z['boxes'][q0,0],z['boxes'][q1,0])+8);ax.set_ylim(-9,8);ax.set_aspect('equal');ax.set_xlabel('model x (m)');ax.set_ylabel('model y (m)');ax.legend();ax.set_title('Model-frame visible points and frozen source boxes; no GT');fig.tight_layout();fig.savefig(V/'POINT_SUPPORT_BEV.png',dpi=150);plt.close(fig)
group=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'));fb=[x for x in read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json') if x['id_PRIVATE']==r['id']]
payload={'alias':r['alias'],'caption_PRIVATE':r['caption'],'source_prediction_query_PRIVATE':q0,'R30_prediction_query_PRIVATE':q1,'frozen_boxes':True,'original_groups_PRIVATE':group,'accepted_preferences_PRIVATE':fb,'source_old_pairs_no_GT_PRIVATE':read(ROOT/'crossplatform/round20/pairs144_attempt001/cases_PRIVATE'/(r['id']+'.json')),'prediction_support':stats,'scene_bindings':bind,'mask_names_private':[{'mask':m,'class_id':o['predicted_class_id'],'box_xyxy_PRIVATE':o['detector_box_xyxy'],'exclusive_visible_points':o['n_exclusive']} for m,o in enumerate(scene['objects'])],'no_ground_truth_loaded':True,'no_IoU_or_task_scores_used':True,'limits':['Point ownership depends on shared camera calibration, predicted masks and visibility model.','Support is not independently verified world-instance ID.','Visible surface is not complete object boundary.','No prediction or label has been changed.']}
dump(OUT/'IDENTITY_POINT_SUPPORT_PRIVATE.json',payload);seal(V,'R31_A_drone034_identity_views_without_tasklabels',n=4,no_task_GT_IoU_or_modelscore_displayed=True);print(json.dumps(stats,indent=2))
