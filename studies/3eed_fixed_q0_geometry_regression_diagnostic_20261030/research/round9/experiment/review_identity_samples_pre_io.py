"""R9 private postseal identity-boundary figures: pure-array preparation stub.
No GT, score, cache, image or annotation file reader is implemented here yet.
Actual I/O adapter must bind the final sealed normal/scorer/private-sidecar schema
and root release before rendering. Never imported by or passed into normal G.
"""
from pathlib import Path
import hashlib,json,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
SALT='R9_identity_overlay_v1'
COLORS={'source_q0':(255,75,65),'direct_G':(35,215,255),'GT7D':(60,240,105)}
EDGES=((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))

def select_diagnostic_rows(rows,declared_scenes):
 """Only already-scored status controls posthoc strata; no new condition fitting.
 Returns selected row references plus aggregate counts. Rows can stay private.
 """
 selected=[];counts=[]
 assert len(declared_scenes)==5
 for platform,scene in sorted(declared_scenes):
  rr=[r for r in rows if (r['platform'],r['scene'])==(platform,scene)];rowcount={'platform':platform,'scene_PRIVATE':scene}
  for confirmed in (True,False):
   pool=[r for r in rr if r['identity_diagnostic']['annotated_spatially_confirmed'] is confirmed]
   key=lambda r:(hashlib.sha256((SALT+'|'+r['id']).encode()).hexdigest(),r['id'])
   chosen=sorted(pool,key=key)[:2];selected.extend(chosen);role='confirmed' if confirmed else 'unconfirmed';rowcount[role+'_available']=len(pool);rowcount[role+'_selected']=len(chosen)
  counts.append(rowcount)
 assert len(selected)<=20 and len({r['id'] for r in selected})==len(selected)
 return selected,counts

def corners7(box7):
 """Model-coordinate yaw radians; follows independent_core_v2.yaw_matrix form."""
 b=np.asarray(box7,dtype=np.float64)
 if b.shape!=(7,) or not np.isfinite(b).all() or (b[3:6]<=0).any():raise ValueError('Finite positive-size model-coordinate box7 required')
 signs=np.array([[-1 if i&4==0 else 1,-1 if i&2==0 else 1,-1 if i&1==0 else 1] for i in range(8)],float)
 c,s=np.cos(b[6]),np.sin(b[6]);rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
 return (signs*b[3:6]/2)@rotation.T+b[:3]

def render_identity_arrays(rgb,*,source_eval6,direct_eval6,GT7D_model,point_projection,project_model_points,full_caption,identity_diagnostic,case_alias='private_case'):
 """No reader/GT matching/model call: caller supplies authorized sealed arrays.
 GT7D is exclusively for this private figure. project_model_points is the frozen
 camera bridge callable with Nx3 modelcoords. Source/direct must be same R9q0.
 """
 if not isinstance(rgb,Image.Image):raise TypeError('Explicitly decoded originalRGB PIL required')
 source=np.asarray(source_eval6,float);direct=np.asarray(direct_eval6,float)
 if source.shape!=(6,) or direct.shape!=(6,) or not np.array_equal(source[3:],direct[3:]):raise ValueError('Source/direct fixed-q0 shape-prior contract required')
 boxes={'source_q0':np.r_[source,0.],'direct_G':np.r_[direct,0.],'GT7D':np.asarray(GT7D_model,float)}
 panel=rgb.convert('RGB').copy();draw=ImageDraw.Draw(panel);uv=np.asarray(point_projection['uv'],float);valid=np.asarray(point_projection['in_image'],bool)
 if uv.ndim!=2 or uv.shape[1]!=2 or valid.shape!=(len(uv),):raise ValueError('Observed-point projection schema')
 ids=np.flatnonzero(valid&np.isfinite(uv).all(1))
 if len(ids)>2500:ids=ids[np.linspace(0,len(ids)-1,2500,dtype=int)]
 for i in ids:
  x,y=uv[i];draw.ellipse((x-1,y-1,x+1,y+1),fill=(220,220,220))
 projection_counts={}
 for name,b in boxes.items():
  projected=project_model_points(corners7(b));pixels=np.asarray(projected['uv'],float);depth=np.asarray(projected['depth'],float);front=(depth>.1)&np.isfinite(pixels).all(1);count=0
  for a,c in EDGES:
   if front[a] and front[c]:draw.line([tuple(pixels[a]),tuple(pixels[c])],fill=COLORS[name],width=3);count+=1
  projection_counts[name]={'front_corners':int(front.sum()),'front_edges_drawn':count,'GTyaw_used':name=='GT7D'}
 # Reflow every caption character into dynamically sized header: no truncation.
 caption=str(full_caption);width=max(1000,panel.width);lines=textwrap.wrap(caption,width=max(30,int((width-24)/8)),replace_whitespace=False,drop_whitespace=False,break_long_words=True) or ['']
 assert ''.join(lines)==caption,'No caption character can be silently dropped'
 header=110+20*len(lines);figure=Image.new('RGB',(width,header+panel.height),(15,15,15));fd=ImageDraw.Draw(figure)
 confirmed=identity_diagnostic['annotated_spatially_confirmed'];label='ANNOTATED SPATIALLY CONFIRMED (not semantic identity proof)' if confirmed else 'UNCONFIRMED (not a wrong-identity label)'
 fd.text((12,6),str(case_alias)+' | '+label,fill=(245,245,245))
 for i,(name,color) in enumerate(COLORS.items()):fd.line((12+i*280,34,42+i*280,34),fill=color,width=4);fd.text((48+i*280,26),name,fill=color)
 fd.text((12,49),'Observed points: gray. GT is offline plotting only; annotation inventory may be incomplete.',fill=(220,220,220))
 fd.text((12,68),'no_other_annotation='+str(identity_diagnostic.get('no_other_annotation'))+' | no per-image method/threshold retuning',fill=(235,205,110))
 for i,line in enumerate(lines):fd.text((12,92+i*20),line,fill=(235,235,235))
 figure.paste(panel,(0,header));return figure,{'state':'private_postseal_identity_overlay_pure_arrays','full_caption_characters_displayed':len(caption),'full_caption_untruncated':True,'annotation_spatial_condition_not_true_identity_label':True,'status_confirmed':bool(confirmed),'observed_point_count':len(uv),'visual_point_count':len(ids),'projection_counts':projection_counts,'GT_not_passed_to_normalG':True,'source_and_direct_native_yaw0_GT_model_yaw_used':True,'raw_media_caption_GT_coordinates_and_IDs_private':True}

def actual_io_adapter(*args,**kwargs):
 raise RuntimeError('Actual rendering disabled in preparation: await root normal seal + GT release and final scorer/cache/privateGTsidecar schema/hash binding. Do not guess or open annotations.')

def synthetic_check():
 box=np.array([0,0,5,4,2,2,np.pi/2]);yaw=corners7(box);zero=corners7(np.r_[box[:6],0.]);assert np.allclose(np.ptp(yaw,axis=0),[2,4,2]) and np.allclose(np.ptp(zero,axis=0),[4,2,2])
 def project(p):
  p=np.asarray(p);z=p[:,2];uv=np.c_[320+100*p[:,0]/z,180+100*p[:,1]/z];return {'uv':uv,'depth':z,'in_image':(z>.1)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<360)}
 caption='A complete synthetic caption with a long relation clause. '*20;image,receipt=render_identity_arrays(Image.new('RGB',(640,360)),source_eval6=box[:6],direct_eval6=box[:6]+np.r_[.05,0,0,0,0,0],GT7D_model=box,point_projection=project(np.array([[0.,0.,5.]])),project_model_points=project,full_caption=caption,identity_diagnostic={'annotated_spatially_confirmed':False,'no_other_annotation':True})
 assert image.height>360 and receipt['full_caption_characters_displayed']==len(caption) and receipt['full_caption_untruncated']
 rows=[]
 for scene in range(5):
  for i in range(5):rows.append({'id':f'synthetic_{scene}_{i}','platform':'synthetic','scene':str(scene),'identity_diagnostic':{'annotated_spatially_confirmed':i<3}})
 chosen,counts=select_diagnostic_rows(rows,{('synthetic',str(i)) for i in range(5)});assert len(chosen)==20 and all(r['confirmed_selected']==r['unconfirmed_selected']==2 for r in counts)
 short,short_counts=select_diagnostic_rows([r for r in rows if r['id'].endswith('_0')],{('synthetic',str(i)) for i in range(5)});assert len(short)==5 and all(r['unconfirmed_selected']==0 for r in short_counts)
 return {'state':'pure_array_yaw_caption_sampling_synthetic_pass_no_real_data','yaw90_preserves_GT_rotation_not_axisaligned':True,'full_caption_no_character_drop':True,'two_per_status_per_scene_hash_and_insufficient_counts_pass':True,'actual_reader_disabled':True,'no_target_GT_scores_cache_media_model_or_GPU_read':True,'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':print(json.dumps(synthetic_check(),indent=2))
