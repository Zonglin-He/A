"""Private RGB/BEV/3D figures only after parent explicit media release.
No model, GPU, adaptation or prediction replacement. Synthetic fixture is available
without a data release and never reads a real image/pointcloud/annotation.
"""
from pathlib import Path
import argparse,json,hashlib,sys,math,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import cv2
R=Path(__file__).resolve().parent;B=R.parents[2];E=B/'crossplatform/round5/baseline_engineering'
sys.path.insert(0,str(R));from diagnose import target,sha,read,verify_inputs,PRIMARY,FULL,NOUN
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
COLORS={'GT':'#15a05b','Full official':'#246bd1','Noun text-only':'#ec8627','Full oracle GT-assisted':'#9673b8'}
def corners(box):
 x,y,z,dx,dy,dz,yaw=np.asarray(box,float);sign=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
 pts=sign*np.array([dx,dy,dz])/2;c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
 return pts@rot.T+np.array([x,y,z])
def source_to_camera(meta,s):
 K=np.asarray(meta['image_intrinsic'],float)[:3,:3];Eraw=np.asarray(meta['image_extrinsic'],float);T=np.eye(4)
 if s['platform']=='waymo':
  angle=np.deg2rad(-[0,-45,45,-90,90][s['waymo_view_suffix']]);c,z=np.cos(angle),np.sin(angle);T[:3,:3]=[[c,-z,0],[z,c,0],[0,0,1]];A=np.eye(4);A[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]];M=A@np.linalg.inv(Eraw)@T
 else:
  if s['platform']=='drone':T[2,3]=-1.8
  M=Eraw@T
 D=np.asarray(meta.get('camera_distortion',[0,0,0,0]),float)
 assert K.shape==(3,3) and M.shape==(4,4) and np.isfinite(K).all() and np.isfinite(M).all() and np.isfinite(D).all()
 return K,D,M
def sample_points(s):
 p=Path(s['lidar_path']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);ix=np.random.default_rng(s['sampling_seed']).choice(len(raw),16384,replace=len(raw)<16384);pc=raw[ix,:3].copy()
 if s['platform']=='drone':pc[:,2]+=1.8
 if s['platform']=='waymo':
  a=np.deg2rad([0,-45,45,-90,90][s['waymo_view_suffix']]);c,z=np.cos(a),np.sin(a);pc=pc@np.array([[c,-z,0],[z,c,0],[0,0,1.]]).T
 return pc.astype(np.float32),ix.astype(np.int64)
def project_camera(xyz,K,D):
 return cv2.projectPoints(np.asarray(xyz,np.float64),np.zeros(3),np.zeros(3),K,D)[0][:,0,:]
def near_clip(a,b,near=.1):
 a=np.asarray(a,float).copy();b=np.asarray(b,float).copy()
 if a[2]<near and b[2]<near:return None
 if a[2]<near:a+=(near-a[2])/(b[2]-a[2])*(b-a)
 if b[2]<near:b+=(near-b[2])/(a[2]-b[2])*(a-b)
 return np.stack([a,b])
def draw_overlay(ax,box,color,K,D,M,dashed=False):
 p=(np.c_[corners(box),np.ones(8)]@M.T)[:,:3];n=0
 for i,j in EDGES:
  seg=near_clip(p[i],p[j])
  if seg is None:continue
  uv=project_camera(seg,K,D)
  if not np.isfinite(uv).all():continue
  ax.plot(uv[:,0],uv[:,1],color=color,lw=1.8,ls='--' if dashed else '-');n+=1
 return n
def figure(image,points,boxes,extra,caption,noun,title,table,camera,out,calibration_status='supported'):
 fig=plt.figure(figsize=(18,13));gs=fig.add_gridspec(2,2,left=.055,right=.98,top=.93,bottom=.24,wspace=.16,hspace=.23);ax=fig.add_subplot(gs[0,0]);ax.imshow(image);ax.set_title('RGB for diagnosis only; model input: XYZ + text');w,h=image.shape[1],image.shape[0];segcounts={}
 if camera is not None:
  K,D,M=camera
  for name,box in boxes.items():segcounts[name]=draw_overlay(ax,box,COLORS[name],K,D,M,name.endswith('GT-assisted'))
 ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
 if calibration_status!='supported':ax.text(.02,.02,'Projection unavailable: '+calibration_status,transform=ax.transAxes,color='red')
 bev=fig.add_subplot(gs[0,1]);three=fig.add_subplot(gs[1,0],projection='3d');mainpoints=np.concatenate([corners(b) for n,b in boxes.items() if 'oracle' not in n]);low=mainpoints.min(0)-np.array([5.,5.,2.]);high=mainpoints.max(0)+np.array([5.,5.,2.]);keep=((points>=low)&(points<=high)).all(-1);pc=points[keep];bev.scatter(pc[:,0],pc[:,1],s=.8,c=pc[:,2],cmap='gray',alpha=.4,rasterized=True);three.scatter(pc[:,0],pc[:,1],pc[:,2],s=.6,c='#666666',alpha=.25,rasterized=True)
 for ob in extra:
  q=corners(ob);bev.plot(*q[[0,1,2,3,0],:2].T,color='#aaaaaa',lw=.6,alpha=.6)
 for name,box in boxes.items():
  q=corners(box);color=COLORS[name];ls='--' if name.endswith('GT-assisted') else '-';bev.plot(*q[[0,1,2,3,0],:2].T,color=color,lw=1.8,ls=ls)
  for i,j in EDGES:three.plot(q[[i,j],0],q[[i,j],1],q[[i,j],2],color=color,lw=1.4,ls=ls)
 bev.set_aspect('equal');bev.set_xlim(low[0],high[0]);bev.set_ylim(low[1],high[1]);bev.set_xlabel('Source X (m)');bev.set_ylabel('Source Y (m)');bev.set_title('BEV: fixed sampled point cloud; gray = annotated others');three.set_xlim(low[0],high[0]);three.set_ylim(low[1],high[1]);three.set_zlim(low[2],high[2]);three.set_box_aspect(high-low);three.view_init(elev=25,azim=-60);three.set_xlabel('Source X');three.set_ylabel('Source Y');three.set_zlabel('Source Z');three.set_title('3D, same coordinate frame and physical axis scale')
 txt=fig.add_subplot(gs[1,1]);txt.axis('off');txt.text(0,1,table,va='top',ha='left',family='monospace',fontsize=10.5);handles=[Line2D([0],[0],color=COLORS[n],lw=2,ls='--' if n.endswith('GT-assisted') else '-') for n in boxes];fig.legend(handles,list(boxes),loc='lower left',bbox_to_anchor=(.05,.195),ncol=2,fontsize=10,frameon=False)
 fig.suptitle(title,fontsize=14);captiontext='FULL MODEL TEXT: '+caption+' . not mentioned\nNOUN MODEL TEXT: '+noun+' . not mentioned';wrapped='\n'.join('\n'.join(textwrap.wrap(line,width=150,break_long_words=False)) for line in captiontext.splitlines());fig.text(.055,.175,wrapped,ha='left',va='top',fontsize=10.5,linespacing=1.4);fig.text(.055,.015,'Private diagnostic figure. GT/oracle annotations are for offline analysis only; neither condition receives RGB or GT. Native yaw0 boxes shown.',fontsize=9,color='#555555');fig.savefig(out,dpi=135);plt.close(fig)
 return {'projected_edge_segments':segcounts,'context_points_shown':len(pc),'image_size_wh':[w,h]}
def main(a):
 if a.synthetic:
  rng=np.random.default_rng(25);image=np.full((400,640,3),235,np.uint8);points=rng.uniform([-5,-5,-1],[5,5,3],(1000,3));boxes={'GT':np.array([0,0,1,2,1,2,.2]),'Full official':np.array([.2,0,1,2,1,2,0]),'Noun text-only':np.array([2,1,1,2,1,2,0])};a.out.mkdir(exist_ok=False,parents=True);figure(image,points,boxes,[], 'Synthetic fixture caption only.','car','Synthetic engineering layout, no target data','Synthetic scalars only\nNo inference or GT evaluation',None,a.out/'synthetic_layout.png');fixture={'near_both_behind_rejected':near_clip([0,0,-1],[1,0,-2]) is None,'near_crossing_clipped':np.allclose(near_clip([0,0,-1],[1,0,1])[:,2],[.1,1]),'corner_center_exact':bool(np.max(np.abs(corners(boxes['GT']).mean(0)-boxes['GT'][:3]))<1e-12),'no_target_or_media_files_read':True,'source_code_sha256':sha(Path(__file__))};assert all(v for k,v in fixture.items() if isinstance(v,bool));(a.out/'FIXTURE.json').write_text(json.dumps(fixture,indent=2));return
 rel=read(a.release);assert rel.get('media_rendering_authorized') is True,'No real pixels before explicit parent media release';assert rel.get('qualitative_and_GT_analysis_authorized') is True
 _,result,sp=verify_inputs(a.cache,a.manifest,a.release,a.scored,a.mode);ds=read(a.diagnoses/'SEALED.json')
 for name,h in ds['files'].items():assert sha(a.diagnoses/name)==h
 diag=read(a.diagnoses/'DIAGNOSES_PRIVATE.json');assert diag['prediction_seal_sha256']==sha(sp) and diag['scores_result_sha256']==sha(a.scored/'RESULTS_PRIVATE.json');dd={x['id']:x for x in diag['rows']};ss={x['id']:x for x in result['rows']};inputs={x['id']:x for x in read(a.manifest)};idx={x['id']:x for x in read(E/'SCORING_INDEX_PRIVATE.json')};selection=read(a.diagnoses/'FIGURE_SELECTION_PRIVATE.json')['cases'];a.out.mkdir(exist_ok=False,parents=True);receipts=[]
 for case in selection:
  if case['id'] is None:receipts.append({**case,'state':'empty_stratum_no_substitution'});continue
  sid=case['id'];s=inputs[sid];ix=idx[sid];d=dd[sid];score=ss[sid];mp=Path(ix['meta_path']);assert sha(mp)==ix['metadata_container_sha256'];meta=read(mp);obj=meta['ground_info'][s['object_index']];gt=target(obj['bbox_3d'],s['platform'],s['waymo_view_suffix']);points,indices=sample_points(s);assert hashlib.sha256(points.tobytes()).hexdigest()==score['point_sample_sha256'];assert hashlib.sha256(indices.tobytes()).hexdigest()==score['raw_indices_sha256'];imagepath=mp.with_name('image.jpg')
  with Image.open(imagepath) as im:image=np.asarray(im.convert('RGB'))
  boxes={'GT':gt}
  for condition,name,arm in [('full','Full official',PRIMARY),('noun','Noun text-only',NOUN)]:
   with np.load(a.cache/(sid+'_'+condition+'.npz'),allow_pickle=False) as z:
    h=z['head_names'].tolist().index('last_');q=score['arms'][arm]['query'];boxes[name]=np.r_[z['boxes'][h,q],0].astype(float)
    if condition=='full':boxes['Full oracle GT-assisted']=np.r_[z['boxes'][h,score['arms'][arm]['oracle_query_GT_assisted']],0].astype(float)
  camera=None;state='supported';camera_stats={}
  try:
   camera=source_to_camera(meta,s);K,D,M=camera;cam=(np.c_[points,np.ones(len(points))]@M.T)[:,:3];positive=cam[:,2]>.1;uv=project_camera(cam[positive],K,D) if positive.any() else np.empty((0,2));valid=np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<image.shape[1])&(uv[:,1]>=0)&(uv[:,1]<image.shape[0]);camera_stats={'sample_positive_depth_fraction':float(positive.mean()),'sample_FOV_depth_fraction':float(valid.sum()/len(points)),'matrix_convention':'Waymo A@inv(camera_to_vehicle)@undo_front; Drone E@undo_z; Quad E'}
   if not valid.any():state='no_sample_points_project_into_positive_depth_FOV';camera=None
  except Exception as e:state=type(e).__name__+':'+str(e);camera=None
  extra=[target(o['bbox_3d_other'],s['platform'],s['waymo_view_suffix']) for o in obj.get('others',[]) if 'bbox_3d_other' in o];lines=['Readout                 selected IoU  own-pool oracle']
  for arm,label in [(PRIMARY,'Full official'),(FULL,'Full text-only'),(NOUN,'Noun text-only')]:lines.append(f"{label:22s} {score['arms'][arm]['iou']:8.4f}      {score['arms'][arm]['oracle_iou']:8.4f}")
  p=d['arms'][PRIMARY];g=d['full_noun_geometry_correspondence'];lines+=['','Same-class annotated other maxIoU: '+f"{p['max_sameclass_annotated_other_iou']:.4f}",'Source/caption-conditioned matched pool','mean pair AABB IoU: '+f"{g['mean_matched_AABB_iou']:.4f}",'Full selected ↔ noun selected: '+str(g['full_selected_matches_noun_selected']),'Geometric correspondence ≠ true instance ID.','','GT injection IoUs (offline oracle only):']
  for field,val in score['postseal_geometry_diagnostic']['selected_single_GT_field_iou'].items():lines.append(f'{field:8s}: {val:.4f}')
  lines+=['','Projection: '+state,'Prior research exposure:',d['prior_exposure'],'Official full is class-label-assisted.','Noun is text-only; first noun may differ','from the actual target class.'];filename=f"{case['platform']}_{case['stratum']}_{sid}.png";fr=figure(image,points,boxes,extra,s['caption'],s['noun_caption'],case['platform']+' | '+case['stratum']+' | '+sid,'\n'.join(lines),camera,a.out/filename,state);receipts.append({**case,'state':'rendered','image_sha256':sha(imagepath),'figure':filename,'figure_sha256':sha(a.out/filename),'coordinate_calibration':camera_stats,'calibration_status':state,**fr})
 (a.out/'RECEIPT_PRIVATE.json').write_text(json.dumps({'selection_sha256':sha(a.diagnoses/'FIGURE_SELECTION_PRIVATE.json'),'prediction_seal_sha256':sha(sp),'source_code_sha256':sha(Path(__file__)),'release_sha256':sha(a.release),'all_original_media_private':True,'cases':receipts},indent=2));print('RENDERED',sum(x['state']=='rendered' for x in receipts))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true');p.add_argument('--mode',choices=['smoke','full'],default='full')
 for name in ['cache','manifest','release','scored','diagnoses','out']:p.add_argument('--'+name,type=Path,required=name=='out')
 main(p.parse_args())
