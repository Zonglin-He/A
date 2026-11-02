"""R51 conditional proposal-TTA audit draft; no model, optimizer, GT or producer import.

Independently replays all-256 AB group-max consensus, local P/N support gates,
RankNet/keep values at stored states, actual parameter differences and sealed
installed-model forward bridges. Does not reconstruct unstored step2 tensors.
"""
from pathlib import Path
import argparse,hashlib,json,math,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','single_G','sameinfo_consensus0','mlp_s1','mlp_s3','lora_qv_s1','lora_qv_s3']
PREFIX=('prefix_query','prefix_vis','prefix_lang','prefix_pos','prefix_text_padding')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def path(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve();assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p);return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name

def consensus(a,b,aid,bid,cues):
 p=np.asarray(a,np.float64);bp=np.asarray(b,np.float64);out=p.copy()
 for g in np.unique(aid[aid>=0]):
  aa=np.flatnonzero(aid==g);bb=np.flatnonzero(bid==g)
  if len(bb):out[aa]=(p[aa]+bp[bb].max())/2
 return out+2*cues[0]+2*cues[1]+2*cues[2],out

def support_sets(st,view):
 off=st[view+'_support_point_offsets'];idx=st[view+'_support_point_indices']
 assert off.shape==(257,) and off[0]==0 and off[-1]==len(idx) and (np.diff(off)>=0).all()
 sets=[set(map(int,idx[off[i]:off[i+1]])) for i in range(256)]
 assert all(len(v)==off[i+1]-off[i] for i,v in enumerate(sets));return sets

def bags_compatible(box_a,aa,box_b,bb):
 if len(aa)<6 or len(bb)<6 or not (box_a[3:]>0).all() or not (box_b[3:]>0).all():return False
 d=np.abs(box_a[:3].astype(float)-box_b[:3].astype(float))
 if not ((d<=box_a[3:].astype(float)/2).all() and (d<=box_b[3:].astype(float)/2).all()):return False
 common=len(aa&bb);return common/(len(aa)+len(bb)-common)>=.70 and common/len(aa)>=.75 and common/len(bb)>=.75

def local_pair(info,st,A,B):
 aid=st['A_proposal_group_ids'];bid=st['B_A_proposal_group_ids'];assert aid.shape==bid.shape==(256,)
 cues=st['fixed_G_components_A'];F=A['scores'].astype(float)+2*cues[0]+2*cues[1]+2*cues[2]
 exact(F,st['literal_G_scores_A'],'frozen G addition');q0=int(A['scores'].argmax());qg=int(F.argmax());positive=int(aid[qg]);negative=int(aid[q0])
 assert info['source_A_query']==q0 and info['literal_G_A_query']==qg and info['positive_A_proposal_group']==positive and info['negative_A_proposal_group']==negative
 conditions=[q0!=qg,positive>=0,negative>=0,positive!=negative,bool(st['fixed_G_increment_A'][qg]>st['fixed_G_increment_A'][q0])]
 exact(F-A['scores'].astype(float),st['fixed_G_increment_A'],'G diagnostic delta')
 abags=support_sets(st,'A');bbags=support_sets(st,'B')
 groups={g['group_id']:g['members'] for g in info['A_proposal_groups']['groups']}
 for gid,members in groups.items():assert members==np.flatnonzero(aid==gid).tolist() or set(members)==set(np.flatnonzero(aid==gid))
 for gid in {positive,negative}-{-1}:
  members=groups[gid]
  for i in members:
   assert st['A_component_counts'][i]>=1
   for j in members:assert bags_compatible(A['boxes'][i],abags[i],A['boxes'][j],abags[j])
 for q in np.flatnonzero((bid==positive)|(bid==negative)):
  if bid[q]<0:continue
  assert st['B_component_counts'][q]>=1
  matches=[gid for gid,members in groups.items() if all(bags_compatible(B['boxes'][q],bbags[q],A['boxes'][j],abags[j]) for j in members)]
  assert matches==[int(bid[q])],'P/N B mapping must be unique whole-support compatible group'
 for view,ids in (('A',aid),('B',bid)):
  P=np.flatnonzero(ids==positive).tolist() if positive>=0 else [];N=np.flatnonzero(ids==negative).tolist() if negative>=0 else []
  assert info['pairs'][view]=={'P':P,'N':N};ss=support_sets(st,view);mp=mn=0.
  conditions.extend([bool(P),bool(N)])
  for q in P+N:assert len(ss[q])>=6
  for p in P:
   for n in N:
    common=len(ss[p]&ss[n]);mp=max(mp,common/len(ss[p]));mn=max(mn,common/len(ss[n]))
  assert info['A_B_PN_max_directional_shared'][view]==[mp,mn]
  conditions.extend([mp<=.25,mn<=.25])
 assert info['accepted']==all(conditions),'Only local fixed P/N geometry may gate preference'
 assert info['semantic_identity_gold'] is False and info['all_256_A_queries_remain_selectable'] is True
 assert info['dominance_and_global_shared_are_not_acceptance_gates'] is True
 return aid,bid,cues

def loss_at(z,z0,pairs,source):
 gaps=[];keeps=[]
 for view in ('A','B'):
  P=pairs[view]['P'];N=pairs[view]['N'];assert P and N and not set(P)&set(N)
  x=np.asarray(z[view],float);b=np.asarray(z0[view],float);gaps.append(float(x[P].max()-x[N].max()))
  outside=sorted(set(range(256))-set(P)-set(N));anchor=int(source[view]['scores'].argmax())
  if outside:keeps.append(float(np.mean(((x-x[anchor])[outside]-(b-b[anchor])[outside])**2)))
 gaps=np.asarray(gaps);u=-gaps/.1;rank=float(np.mean(np.maximum(u,0)+np.log1p(np.exp(-np.abs(u)))));keep=float(np.mean(keeps)) if keeps else 0.
 return {'total':rank+keep,'ranknet':rank,'keep':keep,'gaps':gaps}

def state_shape(interface,initial):
 if interface=='mlp':
  shapes={'0.weight':(288,288),'0.bias':(288,),'2.weight':(288,288),'2.bias':(288,),'4.weight':(64,288),'4.bias':(64,)};n=184960
 else:
  shapes={'A_q':(4,288),'B_q':(288,4),'A_v':(4,288),'B_v':(288,4)};n=4608
  assert not initial['B_q'].any() and not initial['B_v'].any(),'Bzero initialization'
 assert set(initial)==set(shapes) and sum(x.size for x in initial.values())==n
 for k,x in initial.items():assert x.shape==shapes[k] and x.dtype==np.float32 and np.isfinite(x).all()
 return n

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_INDEPENDENT_CONDITIONAL_TTA_AUDIT_NO_GT'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(args.plan)
 assert plan['n']==336 and plan['arm_names']==ARMS and plan['no_model_optimizer_GT_or_new_feedback'] is True
 for g in ('programs_sha256','inputs_sha256'):
  for f,h in plan[g].items():assert sha(path(f))==h,('pin',f)
 roots={k:path(plan[k]) for k in ('frames_root','capture_A_root','capture_B_root','structure_root','fit_root','closure_root','DEV_prediction_root')};seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 for k,p in roots.items():assert plan['inputs_sha256'][(p/'SEALED.json').relative_to(ROOT).as_posix()]==sha(p/'SEALED.json')
 assert auth['prediction_seal_sha256']==sha(roots['frames_root']/'SEALED.json') and seals['frames_root']['arm_names']==ARMS
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 assert 0<plan['loss_abs_bound']<=2e-4 and 0<plan['parameter_norm_abs_bound']<=1e-6 and 0<plan['parameter_norm_rel_bound']<=1e-5
 assert 0<=plan['probability_max_abs_bound']<=2e-5 and 0<=plan['logodds_max_abs_bound']<=2e-4
 out=path(plan['output_directory']);assert not out.exists();out.mkdir(parents=True);done=[];changed=0;accepted=0;steps=0;episode_index={};source_fp=None
 start=time.perf_counter();max_loss_error=0.;max_zero_vector_error=0.;all_zero_endpoint_queries_equal_AB0=True;all_saved_heads_source_bytes_identical=True
 try:
  rows=read(path(plan['manifest']));assert len(rows)==336 and plan['manifest']=='research/round24/input/ALL336_PRIVATE.json'
  fitplan=read(path(plan['fit_execution_plan']));closureplan=read(path(plan['closure_execution_plan']))
  assert fitplan['ranknet_tau']==.1 and fitplan['keep_beta']==1. and fitplan['lr']==.0005 and fitplan['steps']==[0,1,3]
  assert fitplan['interfaces']==['mlp','lora_qv'] and fitplan['max_full_grounder_forwards']==0 and fitplan['new_G_or_reference_calls']==0
  statusrows=read(bound('frames_root','READOUT_STATUS_PRIVATE.json'));assert len(statusrows)==336 and len({r['id'] for r in statusrows})==336;statuses={r['id']:r['arm_statuses'] for r in statusrows}
  init={interface:load(bound('fit_root','initial_states_PRIVATE/'+interface+'.npz')) for interface in ('mlp','lora_qv')}
  for interface in init:state_shape(interface,init[interface])
  registered_mlp=load(bound('capture_A_root','SOURCE_MATCHING_MLP_PRIVATE.npz'));assert set(registered_mlp)==set(init['mlp'])
  for name in init['mlp']:exact(init['mlp'][name],registered_mlp[name],'actual MLP source-init bytes')
  for row in rows:
   sid=row['id'];ac=read(bound('capture_A_root','cases_PRIVATE/'+sid+'.json'));bc=read(bound('capture_B_root','cases_PRIVATE/'+sid+'.json'))
   A=load(bound('capture_A_root',ac['subject']['file']));B=load(bound('capture_B_root',bc['B_file']));sources={'A':A,'B':B}
   assert ac['A_seed']==row['seed'] and bc['B_seed']!=ac['A_seed'] and bc['same_coordinates_no_rotation_translation_or_flip'] is True
   info=read(bound('structure_root','cases_PRIVATE/'+sid+'.json'));st=load(bound('structure_root','arrays_PRIVATE/'+sid+'.npz'));aid,bid,cues=local_pair(info,st,A,B);accepted+=info['accepted']
   assert info['input_bindings']['A_file_sha256']==sha(roots['capture_A_root']/ac['subject']['file']) and info['input_bindings']['B_file_sha256']==sha(roots['capture_B_root']/bc['B_file'])
   base,_=consensus(A['scores'],B['scores'],aid,bid,cues);bq=int(base.argmax())
   p=load(bound('frames_root','samples_PRIVATE/'+sid+'.npz'));dev=load(bound('DEV_prediction_root','samples_PRIVATE/'+sid+'.npz'));gi=dev['arm_names'].tolist().index(plan['selected_G_arm']);si=dev['arm_names'].tolist().index('source')
   exact(A['boxes'],p['source_boxes'],'canonical frames A boxes');exact(A['scores'],p['source_scores'],'canonical A p');assert p['arm_names'].tolist()==ARMS
   exact(p['scores'][0],dev['scores'][si],'source originalDEV score');exact(p['scores'][1],dev['scores'][gi],'singleG originalDEV score');exact(p['scores'][2],base,'canonical AB0 score')
   all_zero_endpoint_queries_equal_AB0 &= bool((p['queries'][3:]==int(base.argmax())).all())
   assert np.array_equal(p['queries'],p['scores'].argmax(1));exact(p['selected_original_boxes'],A['boxes'][p['queries']],'all output boxes originalA')
   for interface,indices in (('mlp',(3,4)),('lora_qv',(5,6))):
    tag=sid+'__'+interface;z=load(bound('fit_root','episodes_PRIVATE/'+tag+'.npz'));r=read(bound('fit_root','cases_PRIVATE/'+tag+'.json'));call=read(bound('fit_root','calls_PRIVATE/'+tag+'.json'))
    assert call['state']=='COMPLETED' and call['output_sha256']==sha(roots['fit_root']/'episodes_PRIVATE'/(tag+'.npz')) and r['accepted']==info['accepted'] and r['GT_read'] is False
    assert r['source_state_file']=='initial_states_PRIVATE/'+interface+'.npz' and r['source_state_sha256']==sha(roots['fit_root']/r['source_state_file'])
    assert r['nonzero_parameter_updates']['0'] is False
    assert r['optimizer_steps']==(3 if info['accepted'] else 0) and len(r['trace'])==r['optimizer_steps'];steps+=r['optimizer_steps']
    if not info['accepted']:assert not any(k.startswith('state_s') for k in z) and r['rejected_s1_s3_states_are_exact_source_s0_no_optimizer'] is True
    for step in (0,1,3):
     F,C=consensus(z[f's{step}_A_probability'],z[f's{step}_B_probability'],aid,bid,cues)
     exact(F,z[f's{step}_scores'],'episode consensus F');exact(C,z[f's{step}_consensus_probability'],'allA native+Bgroupmax/2')
     q=int(F.argmax());assert q==int(z[f's{step}_canonical_A_query']);exact(A['boxes'][q],z[f's{step}_selected_original_A_box'],'episode originalbox')
     if step==0:
      assert q==bq;max_zero_vector_error=max(max_zero_vector_error,float(np.max(np.abs(F-base))))
      for view in ('A','B'):
       assert int(z[f's0_{view}_probability'].argmax())==int(sources[view]['scores'].argmax())
       assert np.max(np.abs(z[f's0_{view}_probability']-sources[view]['scores']))<=plan['probability_max_abs_bound']
       assert np.max(np.abs(z[f's0_{view}_logodds']-sources[view]['logodds']))<=plan['logodds_max_abs_bound']
     else:
      index=indices[0 if step==1 else 1];exact(F,p['scores'][index],'copied fitted full score');assert p['queries'][index]==q
      state={name:z[f'state_s{step}__{name}'] for name in init[interface]} if info['accepted'] else init[interface]
      report=r['update_reports'][str(step)];assert report['trainable_parameters']==(184960 if interface=='mlp' else 4608)
      rows_byname={v['parameter']:v for v in report['parameters']};assert set(rows_byname)==set(init[interface]);nonzero=False
      for name,original in init[interface].items():
       x=state[name];assert x.shape==original.shape and x.dtype==original.dtype and np.isfinite(x).all()
       all_saved_heads_source_bytes_identical &= x.tobytes()==original.tobytes()
       delta=x-original;count=int(np.count_nonzero(delta));nonzero|=count>0;changed+=count;actual=float(np.linalg.norm(delta.astype(np.float64)))
       assert rows_byname[name]['different_source_elements']==count
       assert abs(rows_byname[name]['source_delta_L2']-actual)<=plan['parameter_norm_abs_bound']+plan['parameter_norm_rel_bound']*actual
      assert r['nonzero_parameter_updates'][str(step)]==nonzero and statuses[sid][interface+f'_s{step}']['no_parameter_update']==(not nonzero)
      if not info['accepted']:
       for view in ('A','B'):
        exact(z[f's{step}_{view}_probability'],z[f's0_{view}_probability'],'rejected unchanged p');exact(z[f's{step}_{view}_logodds'],z[f's0_{view}_logodds'],'rejected unchanged z')
    if interface=='lora_qv':
     for step in ('0','1','3'):
      report=r['update_reports'][step];assert report['rank']==report['alpha']==4 and report['K_bias_outproj_other_modules_frozen'] is True
     if r['trace']:
      first={v['parameter']:v for v in r['trace'][0]['parameter_gradients_before_step']['parameters']}
      for name in ('A_q','A_v'):assert first[name]['gradient_nonzero_elements']==0,'Bzero permits initial A gradient0; do not require all LoRA tensors nonzero'
    if info['accepted']:
     z0={v:z[f's0_{v}_logodds'] for v in ('A','B')}
     for step,where,storedstep in ((1,'before',0),(1,'after',1),(2,'before',1),(3,'after',3)):
      actual=loss_at({v:z[f's{storedstep}_{v}_logodds'] for v in ('A','B')},z0,info['pairs'],sources);trace=r['trace'][step-1]
      for name,key in (('total','loss_'),('ranknet','ranknet_'),('keep','keep_')):
       error=abs(actual[name]-trace[key+where]);max_loss_error=max(max_loss_error,error);assert error<=plan['loss_abs_bound']
      assert np.max(np.abs(actual['gaps']-np.asarray(trace['gaps_'+where])))<=plan['loss_abs_bound']
     for trace in r['trace']:assert trace['step'] in (1,2,3) and np.isfinite(trace['gradient_L2']) and trace['gradient_L2']>=0
    episode_index[(sid,interface)]=(z,r,info,st,A,B);done.append(tag)
   assert set(statuses[sid])==set(ARMS)
   for arm,ss in statuses[sid].items():assert set(ss)=={'unknown','fallback','no_parameter_update'} and all(type(v) is bool for v in ss.values())
  assert len(done)==672;fitcounts=read(bound('fit_root','COUNTS.json'));assert fitcounts['optimizer_steps']==steps and fitcounts['source_restored'] is True
  sourceplan=read(path(plan['source_plan']));assert fitcounts['source_fingerprint']==sourceplan['source_model_state_fingerprint'];source_fp=fitcounts['source_fingerprint']
  cs=seals['closure_root'];zero=changed==0
  if zero:
   assert plan.get('allow_verified_allzero_diagnostic') is True
   assert all_saved_heads_source_bytes_identical
   assert all_zero_endpoint_queries_equal_AB0,'No parameter change cannot support a changed decision through numerical drift'
   assert cs['state']=='R51_PROPOSAL_INSTALLED_CLOSURE_NOT_APPLICABLE_VERIFIED_NO_NONZERO_STATES' and cs['n']==0 and cs['new_full_grounder_forwards']==0
   closure_state='NOT_APPLICABLE_VERIFIED_NO_PARAMETER_UPDATE'
  else:
   assert cs['state']=='R51_PROPOSAL_ACTUAL_INSTALLED_FULL_MODEL_CLOSURE_COMPLETE_NO_GT'
   actual=read(bound('closure_root','RESULT_PRIVATE.json'));assert actual['state']=='PASS_R51_PROPOSAL_ACTUAL_INSTALLED_FULL_MODEL_CLOSURE' and actual['complete_required_interface_platform_coverage'] is True
   assert actual['source_restored'] is True and actual['source_fingerprint']==source_fp and actual['new_full_grounder_forwards']==16
   selected=read(bound('closure_root','SELECTED_BEFORE_FORWARD_PRIVATE.json'));assert len(selected['selected'])==4 and not selected['missing']
   assert closureplan['selection_rule']=='HASH_MIN_ACCEPTED_NONZERO_S1_S3_PER_INTERFACE_PLATFORM'
   expected_selection=[]
   for interface in ('mlp','lora_qv'):
    for platform in ('drone','quad'):
     eligible=[v[1] for (sid,it),v in episode_index.items() if it==interface and v[1]['platform']==platform and v[1]['accepted'] and all(v[1]['nonzero_parameter_updates'][str(s)] for s in (1,3))]
     assert eligible;chosen=min(eligible,key=lambda r:hashlib.sha256((closureplan['selection_salt']+'|'+interface+'|'+r['id_PRIVATE']).encode()).hexdigest())
     expected_selection.append({'id_PRIVATE':chosen['id_PRIVATE'],'interface':interface,'platform':platform})
   assert selected['selected']==expected_selection
   expected_call_paths={x['id_PRIVATE']+'__'+x['interface']+f'__s{s}__'+v+'.json' for x in expected_selection for s in (1,3) for v in ('A','B')}
   assert {p.name for p in (roots['closure_root']/'calls_PRIVATE').glob('*.json')}==expected_call_paths
   for item in selected['selected']:
    sid=item['id_PRIVATE'];interface=item['interface'];tag=sid+'__'+interface;z,r,info,st,A,B=episode_index[(sid,interface)];full={}
    assert r['accepted'] and r['nonzero_parameter_updates']['1'] and r['nonzero_parameter_updates']['3']
    for step in (1,3):
     for view,cache in (('A',A),('B',B)):
      key=tag+f'__s{step}__'+view;call=read(bound('closure_root','calls_PRIVATE/'+key+'.json'));o=load(bound('closure_root','outputs_PRIVATE/'+key+'.npz'))
      assert call['state']=='COMPLETED' and call['model_forward_entered'] is True and call['binding']['fit_episode_sha256']==sha(roots['fit_root']/'episodes_PRIVATE'/(tag+'.npz'))
      assert o['full_prefix_query_mask_is_none'].item() and not o['full_text_attention_mask'].any()
      for name in PREFIX:exact(o['full_'+name],cache[name],'actual installed frozen prefix')
      for name,oldname in (('full_query_anchor_xyz','query_anchor_xyz'),('full_query_points_sample_inds','query_points_sample_inds'),('full_raw_point_indices','point_indices'),('full_proj_tokens','tokens')):exact(o[name],cache[oldname],'actual installed frozen source IDs/tokens')
      if interface=='mlp':exact(o['discarded_new_last_head_geometry'],cache['boxes'],'MLP fullgeometry remains source')
      assert np.max(np.abs(o['probability']-z[f's{step}_{view}_probability']))<=plan['probability_max_abs_bound']
      assert np.max(np.abs(o['logodds']-z[f's{step}_{view}_logodds']))<=plan['logodds_max_abs_bound']
      assert int(o['probability'].argmax())==int(z[f's{step}_{view}_probability'].argmax());full[view]=o
     F,_=consensus(full['A']['probability'],full['B']['probability'],st['A_proposal_group_ids'],st['B_A_proposal_group_ids'],st['fixed_G_components_A'])
     assert int(F.argmax())==int(z[f's{step}_canonical_A_query']);exact(A['boxes'][int(F.argmax())],z[f's{step}_selected_original_A_box'],'actual installed final endpoint originalA box')
   closure_state='PASS_REAL_INSTALLED_16_FULL_FORWARDS'
  result={'state':'PASS_R51_INDEPENDENT_CONDITIONAL_TTA_READOUT_LOSS_STATE_AND_CLOSURE_AUDIT_NO_GT','n':336,'arm_names':ARMS,'prediction_seal_sha256':sha(roots['frames_root']/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'accepted_episodes':accepted,'interface_episodes':672,'optimizer_steps':steps,'parameter_elements_different_from_actual_source_summed_s1_s3':changed,'verified_no_parameter_update_branch':zero,'installed_closure_state':closure_state,'max_loss_replay_error':max_loss_error,'max_fit0_vs_captured_AB0_vector_error':max_zero_vector_error,'both_fit0_queries_boxes_equal_one_canonical_AB0':True,'all_256_A_queries_remain_selectable':True,'local_PN_not_semantic_or_global_label':True,'step2_parameter_arrays_and_full_gradients_not_replayed':True,'new_models_optimizers_GT_or_feedback_run':False,'no_GT_read':True,'source_fingerprint_restoration_receipt_matches_registered_source':True,'program_sha256':sha(__file__),'plan_sha256':sha(args.plan),'seconds':time.perf_counter()-start}
  dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}});print(json.dumps(result),flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_interface_episodes':len(done),'no_automatic_retry_GT_model_or_optimizer':True});raise
if __name__=='__main__':main()
