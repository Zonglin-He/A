"""Independent NumPy head replay and calibrated full-box interval proof."""
from common73 import *
from scipy.special import softmax,logsumexp
import itertools
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,model_to_camera_matrix
def native(w,h,t,m):
 x=h
 for k in ['0','2','4']:
  x=x@w[k+'.weight'].T+w[k+'.bias']
  if k!='4':x=np.maximum(x,0)
 x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);l=x@t.T/.07
 return logsumexp(l[:,m],1)-logsumexp(l[:,~m],1)
def main():
 p=read(H/'experiment/RUN_PLAN_LOCK.json');pin(p);pred=ROOT/p['prediction_root'];ad=ROOT/p['adapt_root'];evroot=ROOT/p['evidence_root'];ss={str(root):read(root/'SEALED.json') for root in [pred,ad,evroot]};initial=load(ROOT/p['head_initial']);error=0.;n=accepted=rejected=0;proposals=0
 for r in read(H/'input/MECHANISM64_PRIVATE.json'):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz'
  for root in [pred,evroot]:assert sha(root/rel)==ss[str(root)]['files'][rel]
  z=load(pred/rel);ev=load(evroot/rel);f=prefix(r);assert np.array_equal(z['source_boxes'],f['boxes']) and np.array_equal(z['selected_original_boxes'],f['boxes'][z['queries']]);assert np.array_equal(z['queries'],z['scores'].argmax(1));assert list(z['arm_names'])==ARMS
  case=read(evroot/'cases_PRIVATE'/(sid+'.json'));rec=read(ad/'cases_PRIVATE'/(sid+'.json'));N=ev['main_N'][ev['query_group']];p0=f['scores'].astype(float);p0/=p0.sum();scores=dict(zip(ARMS,z['scores']))
  for arm,key in [('old_R72','old_N'),('parse_fix_direct','parse_N'),('position_direct_opt','main_N'),('position_mismatch_direct_opt','mismatch_N')]:
   q=p0.copy();mask=ev[key][ev['query_group']]
   if mask.any() and not mask.all():
    mass=p0[mask].sum();q[mask]*=.5;q[~mask]*=(1-.5*mass)/(1-mass);q/=q.sum()
   assert np.max(abs(q-scores[arm]))<1e-13
  # Algebraic pinhole reconstruction: no producer projection routine.
  cam=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared');K,E,d,_=cam.validated();assert not np.any(d);undo=np.eye(4)
  if r['platform']=='drone':undo[2,3]=-1.8
  corners=np.array([center+size*np.array(signs)/2 for center,size in zip(f['boxes'][:,:3],f['boxes'][:,3:]) for signs in itertools.product([-1,1],repeat=3)]);pc=(np.c_[corners,np.ones(len(corners))]@(E@undo).T)[:,:3];homo=pc@K.T;uv=homo[:,:2]/homo[:,2:3];uv=uv.reshape(256,8,2);width,height=ev['image_wh'];good=(pc[:,2]>.1).reshape(256,8).all(1)&(uv[:,:,0]>=0).all(1)&(uv[:,:,0]<width).all(1)&(uv[:,:,1]>=0).all(1)&(uv[:,:,1]<height).all(1);rect=np.c_[uv[:,:,0].min(1),uv[:,:,1].min(1),uv[:,:,0].max(1),uv[:,:,1].max(1)];assert np.array_equal(good,ev['query_projection_valid']);assert np.allclose(rect[good],ev['query_rects'][good],atol=1e-8)
  for g in range(len(ev['position_known'])):
   ix=ev['query_group']==g;known=bool(good[ix].all());assert known==ev['position_known'][g];opposite=(rect[ix,0]/width>.55).all() if case['constraint_PRIVATE']['direction']=='left' else (rect[ix,2]/width<.45).all();assert ev['absolute_N'][g]==bool(known and opposite)
  w=load(ad/'actual_weights_PRIVATE'/(sid+'.npz'));expected=softmax(np.log(p0)+native(w,f['h'],f['tokens'],f['token_mask']).astype(float)-native(initial,f['h'],f['tokens'],f['token_mask']).astype(float));err=float(np.max(abs(expected-scores['position_TTA_backtrack1'])));error=max(error,err);assert err<5e-5
  if rec['active']:
   proposals+=1;before=rec['loss_before'];assert len(rec['trials'])<=9
   if rec['accepted']:
    accepted+=1;assert rec['loss_after']<=before-max(1e-12,1e-6*abs(before));assert rec['parameter_change_L2']>0
    first_acceptable=next(x for x in rec['trials'] if x['loss'] is not None and x['loss']<=before-max(1e-12,1e-6*abs(before)));assert first_acceptable['scale']==rec['scale']
   else:rejected+=1;assert rec['before_sha256']==rec['after_sha256'] and rec['optimizer_before_sha256']==rec['optimizer_after_sha256'] and rec['parameter_change_L2']==0
  n+=1
 out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();cert={'state':'PASS_R73_ALL64_PROJECTION_WEIGHT_READOUT_AND_TRANSACTIONAL_STEP_AUDIT','n':n,'task_readouts':n*len(ARMS),'proposals':proposals,'accepted':accepted,'rejected':rejected,'accepted_loss_increases':0,'max_independent_probability_error':error,'no_task_label_lookup':True,'prediction_seal_sha256':sha(pred/'SEALED.json')};dump(out/'AUDIT_RESULT.json',cert);print(cert)
if __name__=='__main__':main()
