"""Prelocked scene-stratified raw geometry plus CLIP reencoding, no labels."""
from common71 import *
import sys,time,torch
from PIL import Image
from prepare71 import prepare
sys.path.insert(0,str(ROOT/'research/round70'));from geometry70 import crops
from clip70 import text_features
sys.path.insert(0,str(ROOT/'crossplatform/round8'));from semantic_encoder import FrozenSemanticEncoder
p=read(H/'experiment/AUDIT_PLAN_LOCK.json');pin(p);out=H/'independent_gpu_audit_attempt001';assert not out.exists();out.mkdir();rr=rows();co=read(ROOT/p['cohort_map']);selected=[]
for cohort,pl,scene in sorted({(co[r['id']],r['platform'],r['scene']) for r in rr}):
 candidates=[r for r in rr if co[r['id']]==cohort and r['platform']==pl and r['scene']==scene];selected+=sorted(candidates,key=lambda r:hashlib.sha256(('R71audit|'+r['id']).encode()).hexdigest())[:2]
dump(out/'SELECTED_PRIVATE.json',[r['id'] for r in selected]);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;enc=FrozenSemanticEncoder(device='cuda',batch_size=32);versions=[x._version for x in enc.model.parameters()];start=time.perf_counter();worst=0.;n=0
for r in selected:
 rel='samples_PRIVATE/'+r['id']+'.npz';a=load(ROOT/p['prepared_root']/rel);z=prefix(r);got,rec=prepare(r,z)
 for k,v in got.items():assert np.array_equal(a[k],v),k
 assert rec['subject_PRIVATE']==read(ROOT/p['prepared_root']/'cases_PRIVATE'/(r['id']+'.json'))['subject_PRIVATE'];c=load(ROOT/p['clip_root']/rel);valid=np.flatnonzero(a['rect_valid']);chosen=valid[::max(1,len(valid)//4)][:4];im=Image.open(r['imagepath']).convert('RGB');v=enc.encode_crops([crops(im,a['rects'][j],s) for j in chosen for s in [1.,1.25]]).numpy().reshape(len(chosen),2,512);err=float(np.max(abs(v-c['visual'][chosen])));assert err<5e-5;worst=max(worst,err);n+=len(chosen)*2;st,_=text_features(enc,rec['subject_PRIVATE']);assert np.max(abs(st-c['subject_text']))<1e-6
assert versions==[x._version for x in enc.model.parameters()];enc._assert_frozen();rec={'state':'PASS_R71_INDEPENDENT_SCENE_STRATIFIED_RAW_PROJECTION_AND_FROZEN_CLIP_AUDIT','n':len(selected),'scene_clusters':len({(co[r['id']],r['platform'],r['scene']) for r in selected}),'CLIP_reencoded_crops':n,'max_image_feature_error':worst,'raw_point_support_groups_and_camera_projection_reconstructed':True,'no_task_labels_or_3EED_updates':True,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);print(rec)
