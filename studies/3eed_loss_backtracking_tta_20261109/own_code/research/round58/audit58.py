"""Independent all-output audit plus all-first-step and complete online optimizer replay."""
import argparse,time,torch,collections,math
from common58 import *
import independent58 as i

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['audit_root'];assert not out.exists();out.mkdir(parents=True);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');rows=read(ROOT/p['manifest']);pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');assert ps['arm_names']==ARMS and ps['proposal_attempts_total']==22848;seals={k:read(ROOT/p[k]/'SEALED.json') for k in ['AB_root','capture_root','association_root','r57_prediction_root','r56_prediction_root']};initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');source=i.params(initial);start=time.perf_counter();cache={};cases={};counts=collections.Counter();maxerr=0.;maxLerr=0.;bridge={};first=[];probe={min([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256((salt+r['id']).encode()).hexdigest())['id'] for pl in ['drone','quad'] for salt in ['R58auditA','R58auditB']}
 def member(k,rel):assert sha(ROOT/p[k]/rel)==seals[k]['files'][rel];return load(ROOT/p[k]/rel)
 def new(rel):assert sha(pred/rel)==ps['files'][rel];return load(pred/rel)
 def trace_check(tr,rule,kind):
  accepted=0
  for r in tr:
   threshold=r['KL_before']-max(1e-12,abs(r['KL_before'])*1e-6);assert not r['target_requires_grad'];assert 1<=len(r['trials'])<= (1 if rule=='original' else 9);assert r['trial_count']==len(r['trials']);assert all(v['scale']==2.**(-j) for j,v in enumerate(r['trials']))
   if rule=='backtrack':
    for v in r['trials'][:-1]:assert not v['finite'] or v['KL']>threshold
    if r['accepted']:assert r['trials'][-1]['KL']<=threshold and r['scale']==r['trials'][-1]['scale'] and r['KL_after_same_detached_target']==r['trials'][-1]['KL']
    else:assert len(r['trials'])==9 and (not r['trials'][-1]['finite'] or r['trials'][-1]['KL']>threshold)
   if r['accepted']:
    accepted+=1;assert r['optimizer_after_hash']==r['optimizer_proposed_hash']
   else:assert r['student_before_hash']==r['student_after_hash'] and r['teacher_before_hash']==r['teacher_after_hash'] and r['optimizer_before_hash']==r['optimizer_after_hash'] and r['KL_before']==r['KL_after_same_detached_target']
   if kind=='fixed':assert r['teacher_before_hash']==r['teacher_after_hash'] and r['teacher_step_L2']==0
   counts[rule+'_proposals']+=1;counts[rule+'_accepted']+=int(r['accepted']);counts[rule+'_trial_evaluations']+=r['trial_count'];counts[rule+'_loss_increased']+=int(r['KL_after_same_detached_target']>r['KL_before']+1e-12)
  return accepted
 try:
  for row in rows:
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';v=member('AB_root',rel);c=member('capture_root',rel);g=member('association_root',rel);d=i.inputs(v,c,g);cache[sid]=d;r57=member('r57_prediction_root',rel);r56=member('r56_prediction_root',rel);z=new(rel);ep=new('fitted_PRIVATE/'+sid+'_episodic.npz');online={(rule,kind):new('fitted_PRIVATE/'+sid+'_online_'+rule+'_'+kind+'.npz') for rule in ['original','backtrack'] for kind in ['fixed','ema']};cp='cases_PRIVATE/'+sid+'.json';assert sha(pred/cp)==ps['files'][cp];rec=read(pred/cp);cases[sid]=rec
   assert np.array_equal(z['source_boxes'],v['A_boxes']) and np.array_equal(z['source_boxes'],r56['source_boxes']) and np.array_equal(z['source_boxes'],r57['source_boxes']);assert np.array_equal(z['source_scores'],v['A_scores']) and np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);assert list(z['arm_names'])==ARMS and np.isfinite(z['scores']).all() and np.array_equal(z['scores'].argmax(1),z['queries'])
   for k,arm in enumerate(ARMS):
    if arm in ['source','frozen_ABC','r56_one']:
     old=r56 if arm=='r56_one' else r57;name='tta1' if arm=='r56_one' else arm;expected=old['scores'][list(old['arm_names']).index(name)];assert np.array_equal(z['scores'][k],expected);continue
    if arm=='cpu_frozen_ABC':par=source
    else:
     rule,protocol,kind,*_=arm.split('_');f=ep if protocol=='epi' else online[(rule,kind)];par=i.params({n:f[arm+'__'+n] for n in initial});h=i.digest(par)
     if protocol=='online':
      timing=arm.split('_')[-1];who=arm.split('_')[3];assert h==rec['online'][rule+'_'+kind][who+'_'+timing+'_hash']
    with torch.no_grad():rr=i.mean_prediction(par,d).numpy()
    err=float(abs(rr-z['scores'][k]).max());assert err<3e-5 and int(rr.argmax())==int(z['queries'][k]),('output',sid,arm,err);maxerr=max(maxerr,err);counts['readouts']+=1
    if arm.startswith('original_'):
     old=arm[len('original_'):]
     if old.startswith('online_'):old=old.replace('online_fixed_','online_fixed_AB_').replace('online_ema_','online_ema_AB_')
     pp=r57['scores'][list(r57['arm_names']).index(old)];b=bridge.setdefault(arm,{'n':0,'max_absolute_score_error':0.,'Top1_disagreements':0});b['n']+=1;b['max_absolute_score_error']=max(b['max_absolute_score_error'],float(abs(pp-rr).max()));b['Top1_disagreements']+=int(pp.argmax()!=rr.argmax())
   for rule in ['original','backtrack']:
    for kind in ['fixed','ema']:
     tr=rec['episodic'][rule+'_'+kind]['trace'];assert len(tr)==16;trace_check(tr,rule,kind);counter=0
     for r in tr:counter+=int(r['accepted']);assert r['optimizer_step_after']==counter
     if sid in probe:
      s,t,o=i.create(initial)
      for step,r in enumerate(tr,1):
       independent=i.independently_step(s,t,o,d,kind,rule);maxLerr=max(maxLerr,i.compare_step(r,independent));counts['full_ep_replayed_steps']+=1
       if step in [1,3,16]:
        for n in initial:
         assert np.array_equal(s[n].detach().numpy(),ep[f'{rule}_epi_{kind}_student_{step}__'+n])
         if kind=='ema':assert np.array_equal(t[n].numpy(),ep[f'{rule}_epi_ema_teacher_{step}__'+n])
    s,t,o=i.create(initial);firststep=i.independently_step(s,t,o,d,'ema',rule)
    for kind in ['fixed','ema']:maxLerr=max(maxLerr,i.compare_step(rec['episodic'][rule+'_'+kind]['trace'][0],firststep,compare_teacher=kind=='ema'))
    counts['first_step_independent_replays']+=1
    if rule=='backtrack':counts['independent_backtrack_non_descent_directions']+=int(firststep['directional_derivative']>=0);counts['independent_backtrack_descent_but_rejected']+=int(firststep['directional_derivative']<0 and not firststep['accepted'])
   o0=rec['episodic']['original_fixed']['trace'][0];b0=rec['episodic']['backtrack_fixed']['trace'][0];assert o0['KL_before']==b0['KL_before'] and o0['KL_after_same_detached_target']==b0['trials'][0]['KL']
   first.append({'platform':row['platform'],'original_first_loss_increased':rec['episodic']['original_fixed']['trace'][0]['full_step_loss_increased'],'backtrack_first_accepted':rec['episodic']['backtrack_fixed']['trace'][0]['accepted'],'backtrack_first_scale':rec['episodic']['backtrack_fixed']['trace'][0]['scale']})
   if len(cache)%32==0:print('AUDIT58_ENDPOINTS',len(cache),336,counts['readouts'],flush=True)
  orders=read(pred/'ONLINE_ORDERS_PRIVATE.json');expected_orders=read(ROOT/p['r57_prediction_root']/'ONLINE_ORDERS_PRIVATE.json');assert orders==expected_orders
  for rule in ['original','backtrack']:
   for kind in ['fixed','ema']:
    for pl,stream in orders.items():
     s,t,o=i.create(initial);accepted=0
     for position,sid in enumerate(stream):
      r=cases[sid]['online'][rule+'_'+kind];assert r['stream_index']==position and r['student_pre_hash']==i.digest(s) and r['teacher_pre_hash']==i.digest(t) and r['optimizer_pre_hash']==i.optdigest(o);trace_check([r],rule,kind);replay=i.independently_step(s,t,o,cache[sid],kind,rule);maxLerr=max(maxLerr,i.compare_step(r,replay));counts['online_replayed_steps']+=1
      if rule=='backtrack':counts['independent_backtrack_non_descent_directions']+=int(replay['directional_derivative']>=0);counts['independent_backtrack_descent_but_rejected']+=int(replay['directional_derivative']<0 and not replay['accepted'])
      accepted+=int(r['accepted']);assert r['optimizer_step_after']==accepted and r['student_post_hash']==i.digest(s) and r['teacher_post_hash']==i.digest(t) and r['optimizer_post_hash']==i.optdigest(o)
    print('AUDIT58_ONLINE',rule,kind,counts['online_replayed_steps'],flush=True)
  producer=read(pred/'COSTS.json');assert producer['counts']=={k:counts[k] for k in producer['counts']};assert counts['original_proposals']==counts['backtrack_proposals']==11424 and counts['backtrack_loss_increased']==0;oldbridge=read(pred/'CPU_R57_BRIDGE.json')
  for arm,r in bridge.items():assert r['n']==oldbridge[arm]['n'] and r['Top1_disagreements']==oldbridge[arm]['Top1_disagreements'] and abs(r['max_absolute_score_error']-oldbridge[arm]['max_absolute_score_error'])<3e-5
  res={'state':'PASS_R58_INDEPENDENT_ALL_READOUTS_FIRST_STEPS_COMPLETE_ONLINE_TRANSACTIONS_AND_REJECTIONS','n':336,'arm_names':ARMS,'prediction_seal_sha256':sha(pred/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'CPU_max_absolute_probability_error':maxerr,'CPU_argmax_disagreements':0,'independent_trial_KL_max_error':maxLerr,'counts':dict(counts),'CPU_R57_bridge_top1_disagreements':sum(r['Top1_disagreements'] for r in bridge.values()),'R57_supervision_association_and_view_seals_unchanged':True,'new_GT_media_or_full_models':False,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',res);dump(out/'FIRST_STEP_PROCESS_SCALARS_PUBLIC_SAFE.json',first);print(res,flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'type':type(e).__name__,'error':str(e),'completed_inputs':len(cache),'counts':dict(counts),'GT_access':False});raise
if __name__=='__main__':main()
