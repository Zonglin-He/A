"""Postseal descriptive native replay on fixed selected cases, no optimization or new scores."""
from target90 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
assert read(H90/'independent_native_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');cases=read(H90/'experiment/GOOD_BAD_CASES_PRIVATE.json')+read(H90/'experiment/FIXED_R89_CASES_PRIVATE.json');sel=read(B90/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');S=load(ROOT/sel['models']['candidate_joint']['state_path']);base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion90(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);restore(m,S);O=B90/'case_replay_PRIVATE';O.mkdir(exist_ok=False);P=B90/'prediction336_attempt001';count=0;error=0.
with torch.no_grad():
 for c in cases:
  r=c['row'];sid=r['id'];z0,z,d,rt,route,pm,im,ws=data90(r);m.route=route;a=load(P/'samples_PRIVATE'/(sid+'.npz'));aa=dict(zip(a['arm_names'],a['scores']));out={}
  for arm in ['candidate_joint_frozen','candidate_joint_tta1']:
   restore(m,S);configure90(m,'candidate_joint')
   if arm.endswith('tta1'):put90(m,load(P/'states_PRIVATE'/(sid+'_'+arm+'.npz')));enable_tta90(m)
   else:m.requires_grad_(False)
   weights=[];hk=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(lambda _m,_a,o:weights.append(o[1].cpu().numpy()[0]));p,_,e=m.main(base,z,d,'bidirectional');hk.remove();count+=1;error=max(error,float(abs(p.cpu().numpy()-aa[arm]).max()));out[arm+'_native_attention']=weights[0];out[arm+'_subject_weights']=e['subject_weights'].cpu().numpy();out[arm+'_conditional']=e['conditional'].cpu().numpy()
  np.savez_compressed(O/(c['case']+'.npz'),**out)
assert error<1e-10 and fingerprint(base)==fp;dump(H90/'CASE_REPLAY_AUDIT.json',{'state':'PASS_R90_POSTSEAL_NATIVE_PREPARED_AND_UPDATED_CASE_REPLAY','case_pages':len(cases),'native_replays':count,'max_probability_error':error,'parameters_optimized':False,'new_task_GT_scoring':False});print(read(H90/'CASE_REPLAY_AUDIT.json'))
