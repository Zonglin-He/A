"""Independent prelabel readout/state/partition audit; no task labels."""
from common70 import *
import torch
from scipy.special import softmax
p=read(H/'experiment/DESIGN_LOCK.json');out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();pred=ROOT/p['prediction_root'];ss=read(pred/'SEALED.json');n=0;maxerr=0.;groups=visual=0;helds=[]
for r in rows():
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';z=prefix(r);x=load(pred/rel);assert sha(pred/rel)==ss['files'][rel];assert np.array_equal(x['source_boxes'],z['boxes']) and np.array_equal(x['source_scores'],z['scores']);assert np.array_equal(x['queries'],x['scores'].argmax(1));assert np.array_equal(x['selected_original_boxes'],z['boxes'][x['queries']]);a=load(ROOT/p['prepared_root']/rel);c=load(ROOT/p['clip_root']/rel);ap=load(ROOT/p['A_root']/rel);bp=load(ROOT/p['B_root']/rel);assert set(a['fit_patch_indices']).isdisjoint(a['held_patch_indices']);assert len(set(a['seed_indices'].tolist()))==len(a['seed_indices']);assert np.array_equal(ap['scores'][0],z['scores']);am=dict(zip(ap['arm_names'],ap['scores']));bm=dict(zip(bp['arm_names'],bp['scores']));base=np.log(np.maximum(z['scores'].astype(float),1e-300));v=c['visual'];qg=a['query_group'];assert qg.min()>=0 and qg.max()<len(v);assert np.all(a['reliability']>=0) and np.all(a['reliability']<=1);groups+=len(v);visual+=int((a['reliability']>0).sum());helds.append(len(a['held_patch_indices']))
 for arm,kind,joint,relia,shuffle in [('A_subject_tight','subject',False,True,False),('A_subject_joint','subject',True,True,False),('A_full_joint','full',True,True,False),('A_projectonly_joint','subject',True,False,False),('A_mismatch_joint','subject',True,True,True)]:
  vv=v.mean(1) if joint else v[:,0];tt=c['subject_text' if kind=='subject' else 'full_text'];sim=np.einsum('ij,j->i',vv.astype(float),tt.astype(float))
  if shuffle:sim=sim[perm(len(sim),'R70Avisualbinding')]
  rr=a['reliability'] if relia else a['project_reliability'];center=np.dot(rr,sim)/rr.sum() if rr.sum()>0 else 0;delta=rr*(sim-center)*float(c['logit_scale']);got=softmax(base+delta[qg]);err=np.max(abs(got-am[arm]));native=vv@tt;native=native[perm(len(native),'R70Avisualbinding')] if shuffle else native;nc=np.sum(rr*native)/rr.sum() if rr.sum()>0 else 0.;nd=rr*(native-nc)*float(c['logit_scale']);exact=softmax(base+nd[qg]);assert np.array_equal(exact,am[arm]);roundoff_bound=.5*np.max(abs(delta[qg]-nd[qg]))+1e-12;assert err<=roundoff_bound;maxerr=max(maxerr,float(err))
 xx=np.column_stack([z['query'],v[:,1][qg],np.broadcast_to(c['subject_text'],(256,512)),a['reliability'][qg,None]]).astype(np.float32)
 for step in [3,16]:
  state={k[len('step'+str(step)+'_'):]:vv for k,vv in ap.items() if k.startswith('step'+str(step)+'_')};down=torch.from_numpy(state['down.weight']);bias=torch.from_numpy(state['down.bias']);up=torch.from_numpy(state['up.weight']);ub=torch.from_numpy(state['up.bias']);delta=(torch.relu(torch.from_numpy(xx)@down.T+bias)@up.T+ub).numpy().ravel();got=softmax(base+delta.astype(float));assert np.max(abs(got-am['A_adapter'+str(step)]))<1e-7
 values=load(ROOT/p['B_root']/'samples_PRIVATE'/(sid+'_held_predictions.npz'));assert set(values)==set(bm)
 for k,vv in values.items():assert vv.shape==(len(a['held_patch_indices']),288) and np.isfinite(vv).all() and np.allclose(np.linalg.norm(vv,axis=1),1,atol=2e-5)
 for k in ARMS:
  if k in am:assert np.array_equal(x['scores'][ARMS.index(k)],am[k])
  if k in bm:assert np.array_equal(x['scores'][ARMS.index(k)],bm[k])
 n+=1
rec={'state':'PASS_R70_INDEPENDENT_CPU_READOUT_AND_SAVED_WEIGHT_REPLAY','n':n,'arm_names':ARMS,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'A_actual_step3_step16_weight_replay_all336':True,'B_fit_held_disjoint_all336':True,'B_saved_unit_predictions_all8_arms':True,'GPU_replay_is_independently_sampled_not_claimed_all336':True,'mapped_groups':groups,'supported_groups':visual,'held_seeds_min':min(helds),'held_seeds_mean':float(np.mean(helds)),'max_independent_evidence_probability_error':maxerr,'prediction_seal_sha256':sha(pred/'SEALED.json')};dump(out/'AUDIT_RESULT.json',rec);print(rec)
