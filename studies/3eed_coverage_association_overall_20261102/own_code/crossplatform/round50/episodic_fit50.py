"""UNEXECUTED bounded DEV-only tiny-head episodic RankNet 1/3 step runner.

Consumes sealed actual two-view capture plus fixed DEV G evidence. No model
backbone/box update, G regeneration, GT, teacher EMA, new regularizer or sweep.
Every sample restores source MLP and rebuilds AdamW. Same-information 0-step
two-view consensus is the causal adaptation reference.
"""
import sys,time,random,hashlib,argparse,functools,math
from pathlib import Path
import numpy as np
from full_common50 import ROOT,EXP,read,sha,dump,npz,seal
from pool_weights50 import legacy_geometry_from_visible,legacy_weights,legacy_operator,exact
from episodic_algorithm50 import native_head,original_ranknet,frozen_structure,consensus,group_values,pick_group
ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']

@functools.lru_cache(maxsize=16)
def old_seal(folder):return read(Path(folder)/'SEALED.json')
def bound(folder,rel):
 folder=Path(folder);s=old_seal(str(folder));p=folder/rel;assert sha(p)==s['files'][rel];return p

def geometry_context(r):
 """Only previously sealed visible ownership and original cached calibration."""
 sys.path[:0]=[str(ROOT/'crossplatform/round49'),str(ROOT/'crossplatform/round7')]
 from ownership49 import frame_route
 from camera_projection import CameraSpec
 A={x['id']:x for x in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')}
 AF=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')
 fr,mp=frame_route(r,A,AF,RF);ins=npz(mp)
 own=ROOT/'crossplatform/round49/ownership336_attempt001';oc=read(bound(own,'cases_PRIVATE/'+r['id']+'.json'))
 assert sha(mp)==oc['frame_mask_cache_sha256'];frame=npz(bound(own,'frames_PRIVATE/'+oc['frame_key_PRIVATE']+'.npz'))
 fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
 if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
 cam=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared')
 return frame,ins,cam

def B_weights(r,boxes,pool):
 from prepare50 import project_boxes_original
 fr,ins,cam=geometry_context(r);H,W=map(int,ins['original_hw']);rect,pv=project_boxes_original(boxes,cam,(W,H))
 raw=legacy_geometry_from_visible(fr['visible_points_model'],fr['point_owner'],fr['depth_camera'],boxes,rect,pv,ins['boxes_xyxy'])
 ass=legacy_weights(pool,raw['point_support_count'],raw['point_support_fraction'],raw['projection_iou'],boxes)
 return ass,raw,rect,pv

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('plan','authorization'):p.add_argument('--'+k,required=True)
 a=p.parse_args();plan=read(a.plan);grant=read(a.authorization)
 assert grant['state']=='ROOT_AUTHORIZED_R50_DEV_EPISODIC_HEAD_ONLY' and grant['plan_sha256']==sha(a.plan)
 assert plan['arm_names']==ARMS and plan['steps']==[1,3] and plan['lr']==.0005 and plan['weight_decay']==0 and plan['tau']==.1
 assert plan['n'] in (2,336) and plan['no_GT_newG_or_FINAL'] is True
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert 'FINAL384' not in rel and 'round46/final' not in rel and 'full5912_gpu_inputs' not in rel and sha(ROOT/rel)==h,rel
 rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
 if plan['n']==2:
  old={r['id']:r for r in rows};rows=read(ROOT/plan['rows_path']);assert len(rows)==2 and all(r==old[r['id']] for r in rows)
 cap=ROOT/plan['capture_root'];assert cap.is_relative_to(ROOT/'crossplatform/round50')
 assert (Path(plan['capture_root'])/'SEALED.json').as_posix() in plan['inputs_sha256']
 cs=old_seal(str(cap));assert cs['n']==len(rows) and cs['state']=='R50_DEV_EPISODIC_TWO_SOURCE_VIEWS_CAPTURED_NO_OPTIMIZER'
 prep=ROOT/'crossplatform/round50/prepared336_attempt001';pred=ROOT/'crossplatform/round50/predicted336_attempt001'
 out=ROOT/plan['output_root'];assert out.is_relative_to(ROOT/'crossplatform/round50') and not out.exists();out.mkdir(parents=True)
 for d in ('samples_PRIVATE','cases_PRIVATE','states_PRIVATE'):(out/d).mkdir()
 import torch
 from torch import nn
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 source=npz(bound(cap,'SOURCE_MATCHING_MLP_PRIVATE.npz'));torch.manual_seed(0);np.random.seed(0);random.seed(0)
 start=time.perf_counter();recs=[];actual_steps=0
 try:
  for i,r in enumerate(rows):
   sid=r['id'];v=npz(bound(cap,'samples_PRIVATE/'+sid+'.npz'));prepared=npz(bound(prep,'arrays_PRIVATE/'+sid+'.npz'));previous=npz(bound(pred,'samples_PRIVATE/'+sid+'.npz'))
   exact(v['A_boxes'],previous['source_boxes'],'canonical_A_boxes');exact(v['A_scores'],previous['source_scores'],'canonical_A_native_scores')
   Aass={'weights':prepared['wide_old_weights'],'valid':prepared['legacy_valid']};pool=prepared['wide_pool'];e=previous['image_e'];pr=previous['raw_mask_PR']
   bass,rawB,rectB,pvalidB=B_weights(r,v['B_boxes'],pool)
   readA=legacy_operator(v['A_scores'],Aass['weights'],pool,e,pr);readB=legacy_operator(v['B_scores'],bass['weights'],pool,e,pr)
   oldwideidx=previous['arm_names'].tolist().index('wideentry_oldW');exact(readA['scores'],previous['scores'][oldwideidx],'literal_A_G_replay')
   cA=np.stack([readA[k] for k in ('vision','position','relation')]);cB=np.stack([readB[k] for k in ('vision','position','relation')])
   structure=frozen_structure(v['A_boxes'],v['A_scores'],v['B_boxes'],v['B_scores'],Aass['valid'],bass['valid'],readA['scores'],readB['scores'])
   gA=structure['groupsA'];repA=structure['repsA'];fg=group_values(readA['scores'],gA);gq=int(repA[pick_group(fg,repA)])
   assert gq==int(readA['scores'].argmax()),'STOP_group_A_G_must_preserve_literal_G_query'
   exact(v['A_boxes'][gq],v['A_boxes'][int(readA['scores'].argmax())],'group_A_G_literal_box')
   # Per-input optimizer/RNG reset, no shared learned state or buffers.
   torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.cuda.manual_seed_all(0)
   head=nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64)).cuda().eval()
   head.load_state_dict({n:torch.tensor(x,device='cuda') for n,x in source.items()},strict=True)
   assert sum(x.numel() for x in head.parameters())==184960 and not list(head.named_buffers())
   state0={n:x.detach().clone() for n,x in head.state_dict().items()};opt=torch.optim.AdamW(head.parameters(),lr=.0005,weight_decay=0.)
   views={name:{k:torch.tensor(v[name+'_'+k],device='cuda').detach() for k in ('h','tokens','token_mask')} for name in ('A','B')}
   def forward():return [native_head(head,z['h'],z['tokens'],z['token_mask']) for z in views.values()]
   with torch.no_grad():outs=forward()
   p0=[o[0].cpu().numpy() for o in outs];z0=[o[1].cpu().numpy() for o in outs]
   replay=[]
   for j,name in enumerate(('A','B')):
    err=float(np.max(abs(p0[j]-v[name+'_scores'])));assert err<=plan['initial_head_native_max_abs_bound'] and int(p0[j].argmax())==int(v[name+'_scores'].argmax())
    replay.append(err)
   cached_A_G=p0[0].astype(float)+2*cA[0]+2*cA[1]+2*cA[2]
   assert int(cached_A_G.argmax())==int(readA['scores'].argmax()),'STOP_initial_cached_G_endpoint_changed'
   con0=consensus(p0[0],p0[1],cA,cB,structure);literal0=consensus(v['A_scores'],v['B_scores'],cA,cB,structure)
   assert con0['query']==literal0['query'],'STOP_initial_consensus_bridge_changed'
   checkpoints={0:dict(p=p0,z=z0,readout=con0)};trace=[];nonzero_updates={};nonzero_outputs={}
   if structure['accepted']:
    agP=structure['positive_group'];agN=structure['source_group'];bgP=structure['mapping'][agP];bgN=structure['mapping'][agN]
    groups=[structure['groupsA'][agP],structure['groupsA'][agN],structure['groupsB'][bgP],structure['groupsB'][bgN]]
    for step in range(1,4):
     opt.zero_grad();outs=forward();loss,gaps=original_ranknet(outs[0][1],outs[1][1],*groups);assert torch.isfinite(loss);loss.backward()
     grad=float(torch.sqrt(sum(x.grad.square().sum() for x in head.parameters() if x.grad is not None)));assert math.isfinite(grad)
     opt.step();actual_steps+=1;trace.append({'step':step,'loss_before':float(loss.detach()),'gaps_before':gaps.detach().cpu().tolist(),'gradient_L2':grad,'gradient_nonzero':bool(grad>0)})
     if step in (1,3):
      with torch.no_grad():post=forward();loss_post,gaps_post=original_ranknet(post[0][1],post[1][1],*groups)
      pp=[o[0].cpu().numpy() for o in post];zz=[o[1].cpu().numpy() for o in post];ro=consensus(pp[0],pp[1],cA,cB,structure)
      nonzero_updates[str(step)]=any(not torch.equal(x,state0[n]) for n,x in head.state_dict().items())
      nonzero_outputs[str(step)]=any(not np.array_equal(pp[j],p0[j]) for j in (0,1))
      checkpoints[step]=dict(p=pp,z=zz,readout=ro)
      np.savez_compressed(out/'states_PRIVATE'/f'{sid}_s{step}.npz',**{n:x.detach().cpu().numpy() for n,x in head.state_dict().items()})
      trace[-1].update(loss_post=float(loss_post),gaps_post=gaps_post.cpu().tolist(),parameter_delta_L2=float(torch.sqrt(sum((x-state0[n]).square().sum() for n,x in head.state_dict().items()))),nonzero_parameter_update=nonzero_updates[str(step)],nonzero_native_output_change=nonzero_outputs[str(step)])
   else:
    checkpoints[1]=checkpoints[0];checkpoints[3]=checkpoints[0];nonzero_updates={'1':False,'3':False};nonzero_outputs={'1':False,'3':False}
   queries=np.array([int(v['A_scores'].argmax()),int(readA['scores'].argmax()),gq,*[checkpoints[s]['readout']['query'] for s in (0,1,3)]])
   score_rows=[v['A_scores'].astype(float),readA['scores'],fg[structure['membershipA']],*[checkpoints[s]['readout']['scores'] for s in (0,1,3)]]
   payload=dict(arm_names=np.array(ARMS),scores=np.stack(score_rows),queries=queries,selected_original_boxes=v['A_boxes'][queries],
    source_boxes=v['A_boxes'],source_scores=v['A_scores'],query_override=np.array([False,False,True,*[checkpoints[s]['readout']['query_override'] for s in (0,1,3)]]),
    A_fixed_VPR=cA,B_fixed_VPR=cB,B_weights=bass['weights'],B_legacy_support=rawB['point_support_count'],B_legacy_coverage=rawB['point_support_fraction'],B_projection_iou=rawB['projection_iou'],
    A_membership=structure['membershipA'],B_membership=structure['membershipB'],A_reps=repA,geometry_repsA=structure['geometry_repsA'],B_reps=structure['repsB'],match_A=structure['match_A'],match_B=structure['match_B'])
   for step in (0,1,3):
    for j,name in enumerate(('A','B')):payload[f'{name}_p_s{step}']=checkpoints[step]['p'][j];payload[f'{name}_z_s{step}']=checkpoints[step]['z'][j]
   np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),**payload)
   rec={k:structure[k] for k in ('q0','qG','source_group','positive_group','pre_A_condition','accepted','reason','strict_A_group_gap','q0_group_has_B_mapping','mask_source_A','mask_positive_A','matching_diagnostics')}
   rec.update(id_PRIVATE=sid,platform=r['platform'],initial_head_max_abs_errors=replay,initial_source_and_G_queries_exact=True,
    steps=3 if structure['accepted'] else 0,trace=trace,nonzero_parameter_updates=nonzero_updates,nonzero_native_output_changes=nonzero_outputs,groupsA_PRIVATE=gA,groupsB_PRIVATE=structure['groupsB'],
    capture_file_sha256=sha(cap/'samples_PRIVATE'/(sid+'.npz')),prepared_file_sha256=sha(prep/'arrays_PRIVATE'/(sid+'.npz')),
    same_visual_information_new_VLM_calls=0,boxes_from_canonical_A_only=True,zero_step_vs_literal_consensus_query_exact=True,
    group_A_G_literal_query_and_box_exact=True,geometry_representatives_frozen_source=True,output_representatives_frozen_canonical_G=True)
   dump(out/'cases_PRIVATE'/(sid+'.json'),rec);recs.append(rec)
   if (i+1)%8==0:print('R50_EPISODIC_HEADS',i+1,len(rows),actual_steps,flush=True)
 except BaseException as e:
  dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed':len(recs),'actual_head_steps':actual_steps,'no_automatic_retry':True});raise
 dump(out/'RECEIPTS_PRIVATE.json',recs)
 seal(out,'R50_DEV_CURRENT_INPUT_1_3_STEP_MATCHING_ONLY_PREDICTIONS_NO_GT',n=len(rows),arm_names=ARMS,
  actual_optimizer_steps=actual_steps,source_grounder_forwards_in_this_stage=0,new_visual_calls=0,
  seconds=time.perf_counter()-start,plan_sha256=sha(a.plan),episodic_reset=True)
if __name__=='__main__':main()
