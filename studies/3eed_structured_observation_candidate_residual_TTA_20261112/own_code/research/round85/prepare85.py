"""Prepare caption-only graph and physical seed/pixel observations; no GT."""
from core85 import *
from PIL import Image
import time
sys.path.insert(0,str(ROOT/'crossplatform/round48'));from parser48 import parse_subject
sys.path.insert(0,str(ROOT/'research/round53'));from support53 import pixel_descriptors
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
start=time.perf_counter();O=B85/'observations366_attempt001';assert not O.exists();(O/'samples_PRIVATE').mkdir(parents=True);(O/'cases_PRIVATE').mkdir();rows=read(H85/'experiment/ALL366_PRIVATE.json');stats=[]
for nr,r in enumerate(rows):
 sid=r['id'];dev=sid in DEV_IDS
 if dev:doc=read(ROOT/'crossplatform/round48/parse336_attempt001/dependencies_PRIVATE'/(sid+'.json'));subject=parse_subject(r['caption'],doc)
 else:
  rr=read(ROOT/'crossplatform/round83/roles2304_attempt001/cases_PRIVATE'/(sid+'.json'));assert rr['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest();doc=rr['document'];subject=parse_subject(r['caption'],doc)
 graph=parse85(r['caption'],doc,subject)
 if dev:z0=prefix(r)
 else:
  with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as f:z0={k:f['A_'+k].copy() for k in EXT_KEYS}
 g=load(casepath(r) if dev else ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz'));boxes=z0['boxes'].astype(float);xyz=z0['xyz'][0].astype(float);labels,reps,support=groups85(boxes,xyz);wh=Image.open(r['imagepath']).size;camera=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');sign=np.array(list(itertools.product([-1,1],repeat=3)));corn=boxes[:,None,:3]+sign[None]*boxes[:,None,3:]/2;projected=project_model_points(corn.reshape(-1,3),camera,wh);uv=projected['uv'].reshape(256,8,2)/np.array(wh);dep=projected['depth'].reshape(256,8);valid=np.isfinite(uv).all((1,2))&(dep>.1).all(1)&(boxes[:,3:]>0).all(1);rect=np.nan_to_num(np.c_[uv[:,:,0].min(1),uv[:,:,1].min(1),uv[:,:,0].max(1),uv[:,:,1].max(1)]);valid&=(rect[:,0]>=0)&(rect[:,1]>=0)&(rect[:,2]<=1)&(rect[:,3]<=1)
 rgb=np.array(Image.fromarray(g['image']).resize((48,32),Image.Resampling.BOX)).reshape(-1,3);desc=pixel_descriptors(rgb);measure=measurements85(boxes,xyz,g,desc,rect,valid);a={'labels':labels,'reps':reps,'support':support,'means':measure[0],'counts':measure[1],'coverage':measure[2],'interval':measure[3],'reliability':measure[4]};np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),**a);dump(O/'cases_PRIVATE'/(sid+'.json'),{'graph':graph,'caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest(),'geometry_sha256':sha(casepath(r) if dev else ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz')),'point_support_is_not_certified_instance':True});stats.append({'row':'OBS'+str(nr+1).zfill(4),'platform':r['platform'],'subject_parsed':graph['usable'],'references':max(0,len(graph['entities'])-1),'relations':len(graph['relations']),'known_camera_relations':sum(e['known_frame'] and e['predicate'] in ['LEFT','RIGHT'] for e in graph['relations']),'target_color':bool(graph['entities'] and graph['entities'][0]['attributes']),'group_count':len(reps),'visible_supported_queries':int((measure[-1]>0).sum())})
 if(nr+1)%64==0:print('PREPARE85',nr+1,flush=True)
dump(O/'COUNTS.json',{'n':len(rows),'seconds':time.perf_counter()-start,'new_GT_reads':0,'vision_expert_outputs_read':False});dump(H85/'OBSERVATION_COVERAGE_ROWS.json',stats);seal(O,'R85_CAPTION_AND_PHYSICAL_OBSERVATIONS_SEALED_BEFORE_ADAPTATION',n=len(rows));print(read(O/'COUNTS.json'))
