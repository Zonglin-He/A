"""Postseal native prepared/readapted visual evidence; no optimization or new task score."""
from targets91 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
assert read(H91/'independent_native_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');cases=read(H91/'experiment/GOOD_BAD_CASES_PRIVATE.json')+read(H91/'experiment/FIXED_R90_CASES_PRIVATE.json');selection=read(B91/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');S=load(ROOT/selection['models']['R76_full_disc']['state_path']);base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);P=B91/'prediction336_attempt001';O=B91/'case_replay_PRIVATE';O.mkdir(exist_ok=False);worst=0.;calls=0
with torch.no_grad():
 for c in cases:
  r=c['row'];sid=r['id'];z0,z,d,rt,route,pm,im,ws=data91(r);a=load(P/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(a['arm_names'],a['scores']));records=read(P/'records_PRIVATE'/(sid+'.json'));out={}
  for arm in ['R76_full_disc_frozen','R76_full_disc_physical_bt','R76_full_disc_conditional_bt']:
   restore(m,S);freeze91(m);m.route=route;m.variant='full';m.use_interface=True
   if not arm.endswith('_frozen'):
    key=records[arm]['patch_key']
    if key!='initial':puttheta91(m,load(P/'states_PRIVATE'/(sid+'_'+key+'.npz')))
   attention=[];hook=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(lambda _m,_a,o:attention.append(o[1].cpu().numpy()[0]));p,_,e=m.main(base,z,d,'bidirectional');hook.remove();calls+=1;worst=max(worst,float(abs(p.cpu().numpy()-scores[arm]).max()));assert int(p.argmax())==int(scores[arm].argmax());out[arm+'_native_attention']=attention[0];out[arm+'_subject_weights']=e['subject_weights'].cpu().numpy()
   lp=li=ld=None
   for p0,i0,w in zip(pm,im,ws):
    v0,v1,v2=m.local_losses(z,d,p0,i0);x=.1*v0*w['P']/2;y=.1*v1*w['I']/2;zz=.01*v2*w['D']/2;lp=x if lp is None else lp+x;li=y if li is None else li+y;ld=zz if ld is None else ld+zz
   out[arm+'_physical_point_contribution']=lp.cpu().numpy();out[arm+'_physical_RGB_contribution']=li.cpu().numpy();out[arm+'_physical_depth_contribution']=ld.cpu().numpy()
  np.savez_compressed(O/(c['case']+'.npz'),**out)
assert worst<1e-6 and fingerprint(base)==fp;v={'state':'PASS_R91_SEALED_NATIVE_CASE_REPLAY_AND_ACTUAL_RECOVERY_CONTRIBUTIONS','pages':len(cases),'unique_observations':len({c['row']['id'] for c in cases}),'native_suffix_calls':calls,'max_probability_error':worst,'top1_changes':0,'parameters_optimized':False,'new_task_GT_scoring':False};dump(H91/'CASE_REPLAY_AUDIT.json',v);print(v)
