"""Prelabel independent 20-input GPU replay and two-image CLIP crop audit."""
from common70 import *
from models70 import *
import sys,time
from PIL import Image
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
sys.path.insert(0,str(ROOT/'research/round66'));import interface66 as F
from geometry70 import prepare,crops
p=read(H/'experiment/DESIGN_LOCK.json');out=H/'independent_gpu_audit_attempt001';assert not out.exists();out.mkdir();rr=rows();selected=[]
for pl,sc in sorted({(r['platform'],r['scene']) for r in rr}):
 candidates=[r for r in rr if r['platform']==pl and r['scene']==sc];selected+=sorted(candidates,key=lambda r:hashlib.sha256(('R70audit|'+r['id']).encode()).hexdigest())[:4]
dump(out/'SELECTED_PRIVATE.json',[r['id'] for r in selected]);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;start=time.perf_counter();m=load_source(read(ROOT/p['source_plan']));m.eval().requires_grad_(False);fp=fingerprint(m);worst=0.;calls=0
for r in selected:
 rel='samples_PRIVATE/'+r['id']+'.npz';z=prefix(r);a=load(ROOT/p['prepared_root']/rel);recalc,_=prepare(r,z)
 for k in ['seed_indices','seed_pixels','patches','fit_patch_indices','held_patch_indices','query_group','reliability','rects']:assert np.array_equal(a[k],recalc[k]),k
 values=load(ROOT/p['B_root']/'samples_PRIVATE'/(r['id']+'_held_predictions.npz'));b=load(ROOT/p['B_root']/rel);bm=dict(zip(b['arm_names'],b['scores']))
 for arm,pred in values.items():
  vis=z['vis0'].copy();ii=a['seed_indices'][a['held_patch_indices']];base=vis[0,ii];norm=np.linalg.norm(base,axis=1,keepdims=True);du=pred-base/np.maximum(norm,1e-6);du/=np.maximum(np.linalg.norm(du,axis=1,keepdims=True),1.);vis[0,ii]=base+.2*norm*du;mask=np.ones(1024,bool);mask[ii]=False;assert np.array_equal(vis[0,mask],z['vis0'][0,mask]);assert np.max(np.linalg.norm(vis[0,ii]-base,axis=1)/np.maximum(norm[:,0],1e-6))<=.20001
  with torch.no_grad():got=F.replay(m,z,alternate_vis0=vis)
  err=np.max(abs(got['p']-bm[arm]));assert err<1e-7,(arm,err);worst=max(worst,float(err));calls+=1
 state=load(ROOT/p['B_root']/'samples_PRIVATE'/(r['id']+'_adapter.npz'));net=RGBAdapter().cuda();net.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()});xx=torch.tensor(a['patches'][a['held_patch_indices']],device='cuda',dtype=torch.float32)/127.5-1
 with torch.no_grad():got=net(xx).cpu().numpy()
 assert np.max(abs(got-values['B_step32']))<2e-5
assert fingerprint(m)==fp;del m,net;torch.cuda.empty_cache()
sys.path.insert(0,str(ROOT/'crossplatform/round8'));from semantic_encoder import FrozenSemanticEncoder
from clip70 import text_features
enc=FrozenSemanticEncoder(device='cuda');crops_n=0;cliperr=0.
for r in [next(r for r in selected if r['platform']==pl) for pl in ['drone','quad']]:
 rel='samples_PRIVATE/'+r['id']+'.npz';a=load(ROOT/p['prepared_root']/rel);c=load(ROOT/p['clip_root']/rel);idx=np.flatnonzero(a['rect_valid'])[::max(1,int(a['rect_valid'].sum())//4)][:4];im=Image.open(r['imagepath']).convert('RGB');got=enc.encode_crops([crops(im,a['rects'][k],s) for k in idx for s in [1.,1.25]]).numpy().reshape(len(idx),2,512);err=np.max(abs(got-c['visual'][idx]));assert err<5e-5;cliperr=max(cliperr,float(err));crops_n+=len(idx)*2
 st=read(ROOT/p['prepared_root']/'cases_PRIVATE'/(r['id']+'.json'))['subject_PRIVATE']
 for name,text in [('subject_text',st),('full_text',r['caption'])]:got,_=text_features(enc,text);assert np.max(abs(got-c[name]))<1e-6
rec={'state':'PASS_R70_INDEPENDENT_STRATIFIED_GPU_AND_RAW_CALIBRATED_RGB_AUDIT','n':len(selected),'scene_clusters':5,'B_replays':calls,'B_saved_final_weights_replayed':len(selected),'B_all8arms_probability_maxerror':worst,'raw_patch_projection_group_reconstruction_inputs':len(selected),'CLIP_reencoded_crops':crops_n,'CLIP_image_feature_maxerror':cliperr,'source_unchanged':True,'no_task_labels_used':True,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);print(rec)
