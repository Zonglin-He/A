"""No labels: same-state factorial readouts plus consistently accepted static-residual TTA."""
from common92 import *
import argparse,time
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();rows=read(H92/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']]
O=B92/('engineering_prediction336_attempt001' if args.engineering else 'prediction336_attempt001')
for f in ['samples_PRIVATE','records_PRIVATE','components_PRIVATE','extra_features_PRIVATE','extra_states_PRIVATE']:(O/f).mkdir(parents=True,exist_ok=False)
seal91=read(P91/'SEALED.json');m,S=model92();base=None;used={};st=[];counts={'n':len(rows),'arms':len(ARMS92),'cached_readout_evaluations':0,'extra_native_features':0,'extra_gradient_proposals':0,'target_GT_read':False,'source_training':False,'architecture_changed':False,'FINAL_access':False};start=time.perf_counter();maxprob=0.;maxfactor=0.
def oldfile(rel,arr=True):
 p=P91/rel;assert sha(p)==seal91['files'][rel];used[rel]=seal91['files'][rel];return load(p) if arr else read(p)
with torch.no_grad():
 for nr,r in enumerate(rows):
  sid=r['id'];route=tensor_route89(load(B89/'target_evidence336_attempt001/samples_PRIVATE'/(sid+'.npz')));a=oldfile('samples_PRIVATE/'+sid+'.npz');old=dict(zip(a['arm_names'],a['scores']));rec91=oldfile('records_PRIVATE/'+sid+'.json',False);features={};components={};meta={};results={k:old[v] for k,v in REF_MAP92.items()};refs={};unknown=len(route['token_roles'])==0
  def getfeature(key):
   if key not in features:features[key]=feature_to92(oldfile('features_PRIVATE/'+sid+'_'+key+'.npz'))
   return features[key]
  def getparts(key):
   if key not in components:
    components[key]=parts92(m,getfeature(key),route,S);counts['cached_readout_evaluations']+=1
   return components[key]
  b0,r0,q0,p0=getparts('view0');maxprob=max(maxprob,float(abs(p0.cpu().numpy()-old[NAME91+'_frozen']).max()));rp=readout92(b0,r0);assert int(rp.argmax())==int(old[NAME91+'_frozen'].argmax())
  for kind,oldkind in [('raw','physical_raw'),('bt','physical_bt'),('wrong_raw','wrong_physical_raw'),('wrong_bt','wrong_physical_bt')]:
   rr=rec91[NAME91+'_'+oldkind];key=rr['feature_key'];b1,r1,q1,p1=getparts(key);maxprob=max(maxprob,float(abs(p1.cpu().numpy()-old[NAME91+'_'+oldkind]).max()));assert int(p1.argmax())==int(old[NAME91+'_'+oldkind].argmax());vals=same_parts92(b0,r0,b1,r1);maxfactor=max(maxfactor,float(((b1+r1)-(b1+r0)-(b0+r1)+(b0+r0)).abs().max()))
   for part in ['static_residual','residual_only']:
    arm='same_state_'+kind+'_'+part;results[arm]=vals[part].cpu().numpy();meta[arm]={'kind':'same_state_factorial','old_kind':oldkind,'feature_key':key,'state_path':str(P91/'states_PRIVATE'/(sid+'_'+rr['patch_key']+'.npz')) if rr['patch_key']!='initial' else 'initial','alpha':rr['alpha'],'parameter_L2':rr['parameter_L2'],'new_optimizer_run':False,'update_applied':rr['update_applied']}
  # Fixed original proposal, but acceptance uses precisely b(theta)+r0 and its own KL.
  extra_info={};lazy=None
  for wrong in [False,True]:
   prefix='wrong' if wrong else 'physical';oldkind='wrong_physical_raw' if wrong else 'physical_raw';before=rec91[NAME91+'_'+oldkind]['before'];trials=[];selected=None
   available={}
   for rr in rec91.values():
    if (rr['kind'].startswith('wrong_physical') if wrong else rr['kind'].startswith('physical')):
     for t in rr['trials']:
      if t['feature_key'] in available:assert available[t['feature_key']]['physical']==t['physical'] and available[t['feature_key']]['bound']==t['bound']
      available[t['feature_key']]=t
   for k,alpha in enumerate(GRID91):
    key=prefix+'_'+str(k)
    if key in available:
     t=available[key];physical=t['physical'];bound=t['bound'];b1,r1,q1,_=getparts(key);statepath=P91/'states_PRIVATE'/(sid+'_'+key+'.npz');oldfile('states_PRIVATE/'+sid+'_'+key+'.npz')
    else:
     # Cache shortage is not a reason to truncate the fixed nine-scale search.
     if base is None:
      sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
      base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'))
     if lazy is None:lazy=data91(r)
     z0,z,d,rt,route,pm,im,ws=lazy
     with torch.enable_grad():g,raw,_=physical_direction91(m,base,z,d,S,pm,im,ws,wrong)
     counts['extra_gradient_proposals']+=1;direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;freeze91(m);theta_step91(m,S,raw if k==0 else alpha*direction)
     f=feature91(m,base,z,d,route,S);counts['extra_native_features']+=1;b1,r1,q1,_=parts92(m,f,route,S);features[key]=f;components[key]=(b1,r1,q1,readout92(b1,r1));statepath=O/'extra_states_PRIVATE'/(sid+'_'+key+'.npz');np.savez_compressed(statepath,**theta_patch91(m));np.savez_compressed(O/'extra_features_PRIVATE'/(sid+'_'+key+'.npz'),**{n:v.cpu().numpy() for n,v in f.items()});physical=float(uniform_aux91(m,z,d,pm,im,ws,wrong));bound=before+1e-4*float(g.double()@(raw if k==0 else alpha*direction).double())
    pp=readout92(b1,r0);keep=float(keep91(pp,q1,rp,q0));total=physical+keep;ok=bool(np.isfinite(total) and total<=bound+1e-10);trials.append({'alpha':alpha,'feature_key':key,'state_path':str(statepath),'physical':physical,'keep_static_readout':keep,'objective':total,'bound':bound,'accepted':ok})
    if ok:selected=(key,pp,alpha,statepath,total);break
   arm='static_residual_formal_'+('wrong_bt' if wrong else 'bt')
   if selected:
    key,pp,alpha,statepath,total=selected;theta=load(statepath);norm=float(np.sqrt(sum(np.sum((theta[n]-S[n])**2) for n in theta)));results[arm]=pp.cpu().numpy();meta[arm]={'kind':'formal_static_residual_acceptance','feature_key':key,'state_path':str(statepath),'alpha':alpha,'parameter_L2':norm,'before':before,'after':total,'accepted_by_loss_rule':True,'update_applied':True,'wrong_binding':wrong,'trials':trials}
   else:results[arm]=rp.cpu().numpy();meta[arm]={'kind':'formal_static_residual_acceptance','feature_key':'view0','state_path':'initial','alpha':0.,'parameter_L2':0.,'before':before,'after':before,'accepted_by_loss_rule':False,'update_applied':False,'wrong_binding':wrong,'trials':trials}
  status={}
  oldstatus=next(x['arm_statuses'] for x in oldfile('READOUT_STATUS_PRIVATE.json',False) if x['id']==sid)
  for arm in ARMS92:
   if arm in REF_MAP92:status[arm]=oldstatus[REF_MAP92[arm]]
   else:status[arm]={'unknown':unknown,'fallback':unknown or not meta[arm]['update_applied'],'no_parameter_update':not meta[arm]['update_applied']}
  v=np.stack([results[x] for x in ARMS92]);assert np.isfinite(v).all() and np.allclose(v.sum(1),1);qq=v.argmax(1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS92),scores=v,queries=qq,source_boxes=a['source_boxes'],selected_original_boxes=a['source_boxes'][qq]);np.savez_compressed(O/'components_PRIVATE'/(sid+'.npz'),**{key+'_'+part:value.cpu().numpy() for key,vals in components.items() for part,value in zip(['base','residual','query','probability'],vals)});dump(O/'records_PRIVATE'/(sid+'.json'),meta);st.append({'id':sid,'arm_statuses':status});restore(m,S)
  if (nr+1)%32==0 or args.engineering:print('RUN92',nr+1,len(rows),flush=True)
assert maxprob<1e-6 and maxfactor<1e-10;counts.update(seconds=time.perf_counter()-start,peak_GPU_bytes=int(torch.cuda.max_memory_allocated()),max_R91_probability_error=maxprob,max_factorial_identity_error=maxfactor);dump(O/'COUNTS.json',counts);dump(O/'READOUT_STATUS_PRIVATE.json',st);dump(O/'USED_R91_FILES_PRIVATE.json',used);seal(O,'R92_FIXED_R91_STATIC_RESIDUAL_FORMAL_ACCEPTANCE_AND_FACTORIAL_READOUTS_SEALED',n=len(rows),arm_names=ARMS92);print(counts)
