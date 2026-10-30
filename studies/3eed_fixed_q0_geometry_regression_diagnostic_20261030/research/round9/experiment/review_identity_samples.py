"""R9 private postseal identity-boundary figures: pure-array preparation stub.
Actual I/O is disabled without an independently hash-bound root render release.
It reads only the same sealed R9 normal outputs and postrelease scored GT7D sidecar.
No raw annotation object, private scoring index, model or normal-G module is loaded.
Never imported by or passed into normal G.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys,textwrap,time
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

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text())
def bounded(root,name):
 q=(Path(root)/name).resolve()
 if not q.is_relative_to(Path(root).resolve()):raise ValueError('Artifact path escapes registered root')
 return q

def actual_io_adapter(args):
 """Postseal private rendering only; every GT read follows the complete gates.
 The sidecar is a hashed member of the scoring seal, not a fabricated third seal.
 GT is never supplied to the normal G, model, prediction selection or adaptation.
 """
 started=time.monotonic();B=Path(__file__).resolve().parents[3]
 assert file_sha(args.render_release)==args.render_release_sha256
 grant=read_json(args.render_release)
 assert grant['identity_overlay_render_authorized'] is True and grant['no_G_or_method_retuning'] is True
 assert grant['selection_salt']==SALT and grant['per_scene_confirmed_max']==grant['per_scene_unconfirmed_max']==2 and grant['total_max']==20
 planpath=Path(__file__).with_name('IDENTITY_OVERLAY_PLAN_LOCK.json');plan=read_json(planpath)
 assert grant['overlay_plan_sha256']==file_sha(planpath) and grant['renderer_sha256']==file_sha(__file__)
 assert grant['manifest_sha256']==file_sha(args.manifest)==plan['input_manifest_sha256']
 assert plan['GT_is_postseal_plotting_only_never_normalG_feedback'] and plan['no_figures_for_threshold_prompt_G_parameter_or_arm_tuning']
 normal_seal_path=args.normal_predictions/'SEALED.json';scored_seal_path=args.scored/'SEALED.json';sidecar_path=args.scored/'PRIVATE_GT7D_SIDECAR.json'
 assert grant['normal_prediction_seal_sha256']==file_sha(normal_seal_path)
 assert grant['scored_seal_sha256']==file_sha(scored_seal_path)
 assert grant['GT_score_release_sha256']==file_sha(args.score_release)
 assert grant['GT7D_sidecar_sha256']==file_sha(sidecar_path)
 transform=B/'research/round5/audit/independent_core_v2.py';camera_code=B/'crossplatform/round7/camera_projection.py';sampler_code=B/'crossplatform/round7/experiment/runner_helpers.py'
 assert grant['transform_helper_sha256']==plan['GT_helper_sha256']==file_sha(transform)
 assert grant['camera_helper_sha256']==plan['camera_helper_sha256']==file_sha(camera_code)
 normal=read_json(normal_seal_path);scored=read_json(scored_seal_path);GTgrant=read_json(args.score_release)
 assert normal['state']=='sealed_R9_normal_fixedq0_predictions_and_sourcecaches_before_GT' and normal['phase']=='normal'
 assert normal['n_samples']==normal['source_cache_count']==scored['n_samples']==160 and normal['n_arms']==11
 assert normal['all_11arms_and_sourcecache_sealed_before_GT'] and normal['all_normal_arm_queries_fixed_q0'] and normal['grounder_restored_and_CLIP_frozen']
 assert normal['manifest_sha256']==scored['manifest_sha256']==grant['manifest_sha256']
 assert scored['normal_prediction_seal_sha256']==grant['normal_prediction_seal_sha256'] and scored['GT_release_sha256']==grant['GT_score_release_sha256']
 assert GTgrant['CPU_SCORE_RELEASE'] is True and GTgrant['complete_normal_predictions_sealed_before_GT'] is True and GTgrant['stage']=='normal' and GTgrant['source_model_source_only_Waymo'] is True
 assert GTgrant['prediction_seal_sha256']==grant['normal_prediction_seal_sha256'] and GTgrant['manifest_sha256']==grant['manifest_sha256'] and GTgrant['source_checkpoint_sha256']==normal['source_checkpoint_sha256']
 assert normal['program_sha256_base']=='workspace_root'
 for rel,digest in normal['program_sha256'].items():assert file_sha(bounded(B,rel))==digest
 assert normal['program_sha256']['crossplatform/round7/experiment/runner_helpers.py']==file_sha(sampler_code)
 for rel,digest in normal['files'].items():assert file_sha(bounded(args.normal_predictions,rel))==digest
 for name in ('RESULTS_PRIVATE.json','SCORER_LOCK.json','PRIVATE_GT7D_SIDECAR.json'):assert scored['files'][name]==file_sha(args.scored/name)
 assert scored['files']['PRIVATE_GT7D_SIDECAR.json']==grant['GT7D_sidecar_sha256']
 lock=read_json(args.scored/'SCORER_LOCK.json')
 assert lock['state']=='all_normal_source_program_hashes_and_prediction_contracts_checked_before_target_GT'
 assert lock['release_sha256']==grant['GT_score_release_sha256'] and lock['prediction_seal_sha256']==grant['normal_prediction_seal_sha256']
 assert lock['scorer_sha256']==GTgrant['scorer_program_sha256']==file_sha(B/'research/round9/evaluation/score_normal.py')
 assert lock['metrics_sha256']==GTgrant['geometry_metrics_sha256']==file_sha(B/'research/round9/evaluation/geometry_metrics.py')
 # Only now can score-conditioned status and the separately sealed GT7D be read.
 result=read_json(args.scored/'RESULTS_PRIVATE.json');sidecar=read_json(sidecar_path);manifest=read_json(args.manifest)
 assert result['state']=='normal11_fixedq0_scored_postseal' and result['n']==len(result['rows_PRIVATE'])==len(manifest)==160
 assert sidecar['scope']=='labelled_controlled_and_postseal_overlay_only_NOT_normal_G_input'
 for key,expected in [('normal_prediction_seal_sha256',grant['normal_prediction_seal_sha256']),('manifest_sha256',grant['manifest_sha256']),('score_release_sha256',grant['GT_score_release_sha256']),('transform_helper_sha256',grant['transform_helper_sha256'])]:assert sidecar[key]==expected
 rows=result['rows_PRIVATE'];lookup={r['id']:r for r in manifest};targets={r['id']:r for r in sidecar['rows']}
 assert len(lookup)==len(targets)==160 and {r['id'] for r in rows}==set(lookup)==set(targets)
 keys={'id','platform','scene','pointpath','imagepath','metapath','caption','suffix','seed'}
 assert all(set(r)==keys and re.fullmatch(r'[0-9a-f]{16}',r['id']) for r in manifest)
 assert all(isinstance(r['identity_diagnostic']['annotated_spatially_confirmed'],bool) for r in rows)
 scenes={(r['platform'],r['scene']) for r in manifest};assert len(scenes)==5
 selected,counts=select_diagnostic_rows(rows,scenes);assert len(selected)<=20
 trace_path=args.normal_predictions/'trace.json';assert file_sha(trace_path)==normal['files']['trace.json'];traces={r['id']:r for r in read_json(trace_path)}
 registered_plan=read_json(args.normal_predictions/'registered_plan.json');assert file_sha(args.normal_predictions/'registered_plan.json')==normal['plan_sha256']
 assert registered_plan['missing_distortion_policy']=='assume_zero_declared'
 # These imports are input-only helpers, not the model, G, scorer or GT index.
 sys.path.insert(0,str(B/'crossplatform/round7'));sys.path.insert(0,str(B/'crossplatform/round7/experiment'))
 from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
 from runner_helpers import sample_points,array_sha
 assert not args.output.exists(),'Never overwrite previous private figure evidence'
 args.output.mkdir(parents=True);files={};case_receipts=[]
 for i,diagnostic in enumerate(selected,1):
  sid=diagnostic['id'];sample=lookup[sid];target=targets[sid];tr=traces[sid]
  assert (sample['platform'],sample['scene'])==(diagnostic['platform'],diagnostic['scene'])==(target['platform'],target['scene'])
  assert set(target)=={'id','platform','scene','GT7D_model'}
  gt=np.asarray(target['GT7D_model'],float);assert gt.shape==(7,) and np.isfinite(gt).all() and (gt[3:6]>0).all()
  assert np.array_equal(gt,gt.astype(np.float32).astype(float)),'Sidecar must preserve audited float32 GT values'
  pred_path=args.normal_predictions/(sid+'.npz');cache_rel='source_cache/'+sid+'.npz';cache_path=args.normal_predictions/cache_rel
  assert file_sha(pred_path)==normal['files'][pred_path.name] and file_sha(cache_path)==normal['source_cache_files'][cache_rel]==normal['files'][cache_rel]
  with np.load(pred_path,allow_pickle=False) as pred,np.load(cache_path,allow_pickle=False) as cache:
   q=int(pred['q_source']);arm_names=pred['arm_names'].tolist();source_i=arm_names.index('source');direct_i=arm_names.index('rgb_fulltext__direct_G')
   assert len(arm_names)==11 and np.array_equal(pred['arm_query_index'],np.full(11,q)) and q==diagnostic['q_source']==int(cache['query_index'])
   assert str(pred['sample_id'])==str(cache['sample_id'])==sid and str(cache['source_checkpoint_sha256'])==normal['source_checkpoint_sha256']
   assert np.array_equal(cache['boxes_raw6'][0],pred['source_boxes_raw6']) and np.array_equal(pred['arm_boxes_raw6'][source_i],cache['boxes_raw6'][0,q])
   assert np.array_equal(cache['text_scores'][0],pred['source_text_scores256']) and q==int(pred['source_text_scores256'].argmax())
   assert np.array_equal(pred['source_boxes_eval6'][q],pred['arm_boxes_eval6'][source_i])
   source=pred['arm_boxes_eval6'][source_i].copy();direct=pred['arm_boxes_eval6'][direct_i].copy()
   assert np.array_equal(source[3:],direct[3:])
   for key,tkey in [('pointpath','raw_point_file_sha256'),('imagepath','image_file_sha256'),('metapath','camera_metadata_file_sha256')]:assert file_sha(sample[key])==tr[tkey]
   points,index,raw_n=sample_points(sample);assert array_sha(points)==pred['point_sample_sha256'].tobytes().hex()==cache['point_sample_sha256'].tobytes().hex()==tr['point_sample_sha256']
   assert array_sha(index)==pred['raw_indices_sha256'].tobytes().hex()==cache['raw_indices_sha256'].tobytes().hex()==tr['raw_indices_sha256']
   assert pred['image_file_sha256'].tobytes().hex()==tr['image_file_sha256']
   assert cache['actual_model_text_sha256'].tobytes().hex()==hashlib.sha256((sample['caption']+registered_plan['sentinel']).encode()).hexdigest()
   assert cache['manifest_row_sha256'].tobytes().hex()==hashlib.sha256(json.dumps(sample,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   raw_invalid={'source':int(np.count_nonzero(pred['arm_boxes_raw6'][source_i,3:]<=0)),'direct_G':int(np.count_nonzero(pred['arm_boxes_raw6'][direct_i,3:]<=0))}
  with Image.open(sample['imagepath']) as image:rgb=image.convert('RGB')
  fields=read_camera_allowlist(sample['metapath']);camera=CameraSpec.from_camera_fields(sample['platform'],fields,waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared')
  project=lambda p:project_model_points(p,camera,rgb.size);projection=project(points)
  alias='private_identity_case_'+str(i).zfill(2)
  figure,receipt=render_identity_arrays(rgb,source_eval6=source,direct_eval6=direct,GT7D_model=gt,point_projection=projection,project_model_points=project,full_caption=sample['caption'],identity_diagnostic=diagnostic['identity_diagnostic'],case_alias=alias)
  receipt.update({'case_alias':alias,'id_PRIVATE':sid,'platform':sample['platform'],'scene_PRIVATE':sample['scene'],'q_source':q,'identity_diagnostic':diagnostic['identity_diagnostic'],'raw_nonpositive_dimensions_display_clamped':raw_invalid,'source_prediction_npz_sha256':file_sha(pred_path),'source_cache_sha256':file_sha(cache_path),'GT7D_sidecar_sha256':grant['GT7D_sidecar_sha256'],'camera_projection_diagnostics':projection['diagnostics'],'observed_points_original_raw_count':raw_n})
  image_name=alias+'.png';record_name=alias+'.json';figure.save(args.output/image_name);(args.output/record_name).write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
  files[image_name]=file_sha(args.output/image_name);files[record_name]=file_sha(args.output/record_name);case_receipts.append(receipt)
 summary={'state':'sealed_R9_private_postseal_identity_overlays_no_G_retuning','n_selected':len(selected),'n_figures':len(case_receipts),'stratum_counts_PRIVATE':counts,'selection_salt':SALT,'per_scene_confirmed_max':2,'per_scene_unconfirmed_max':2,'total_max':20,'normal_prediction_seal_sha256':grant['normal_prediction_seal_sha256'],'scored_seal_sha256':grant['scored_seal_sha256'],'GT_score_release_sha256':grant['GT_score_release_sha256'],'GT7D_sidecar_sha256':grant['GT7D_sidecar_sha256'],'manifest_sha256':grant['manifest_sha256'],'overlay_plan_sha256':grant['overlay_plan_sha256'],'renderer_sha256':file_sha(__file__),'render_release_sha256':file_sha(args.render_release),'transform_helper_sha256':grant['transform_helper_sha256'],'camera_helper_sha256':grant['camera_helper_sha256'],'no_model_GPU_G_or_new_IoU_scoring':True,'GT7D_used_for_postseal_plotting_only_not_normalG':True,'identity_condition_annotation_spatial_not_semantic_truth':True,'private_media_captions_identifiers_and_coordinates_not_public':True,'source_and_direct_same_sealed_R9_q0_cache':True,'full_caption_untruncated_all_figures':all(r['full_caption_untruncated'] for r in case_receipts),'files':files,'seconds':time.monotonic()-started}
  # No scalar receipt can cause method changes; this is private descriptive evidence.
 (args.output/'SEALED.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'state':summary['state'],'n_figures':summary['n_figures'],'seal_sha256':file_sha(args.output/'SEALED.json'),'seconds':summary['seconds']},indent=2))


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
 return {'state':'pure_array_yaw_caption_sampling_synthetic_pass_no_real_data','yaw90_preserves_GT_rotation_not_axisaligned':True,'full_caption_no_character_drop':True,'two_per_status_per_scene_hash_and_insufficient_counts_pass':True,'actual_reader_requires_hash_bound_root_release_and_sealed_normal_scored_sidecar':True,'no_target_GT_scores_cache_media_model_or_GPU_read':True,'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--synthetic-check',action='store_true')
 for key in ('manifest','normal-predictions','scored','score-release','render-release','output'):parser.add_argument('--'+key,type=Path)
 parser.add_argument('--render-release-sha256');args=parser.parse_args()
 if args.synthetic_check:print(json.dumps(synthetic_check(),indent=2))
 else:
  for key in ('manifest','normal_predictions','scored','score_release','render_release','render_release_sha256','output'):
   if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' required')
  actual_io_adapter(args)

