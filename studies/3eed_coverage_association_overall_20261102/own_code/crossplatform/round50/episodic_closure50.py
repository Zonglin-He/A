"""UNEXECUTED no-GT installed-head closure for real accepted episodic states.

Select at most one accepted episode/platform by a prelocked hash, never IoU.
Install its actual 1/3-step states and compare fresh full-grounder output with
the sealed cached-feature computation, including all256 boxes and nativep.
"""
import sys,time,hashlib,random,argparse
from pathlib import Path
import numpy as np
from full_common50 import ROOT,read,sha,dump,npz,seal
from episodic_capture50 import bseed
from pool_weights50 import exact
from episodic_algorithm50 import consensus

def outside(model):
 h=hashlib.sha256()
 for n,x in model.state_dict().items():
  if not n.startswith('contrastive_align_projection_image.'):
   h.update(n.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def bound(folder,seal_,rel):
 p=folder/rel;assert sha(p)==seal_['files'][rel];return p
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('plan','authorization'):p.add_argument('--'+k,required=True)
 a=p.parse_args();a.plan=str(Path(a.plan).resolve());a.authorization=str(Path(a.authorization).resolve())
 plan=read(a.plan);grant=read(a.authorization)
 assert grant['state']=='ROOT_AUTHORIZED_R50_DEV_EPISODIC_INSTALLED_HEAD_CLOSURE' and grant['plan_sha256']==sha(a.plan)
 assert plan['n_per_platform']==1 and plan['steps']==[1,3] and plan['max_grounder_forwards']==8
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert 'FINAL384' not in rel and 'round46/final' not in rel and 'full5912_gpu_inputs' not in rel and sha(ROOT/rel)==h,rel
 cap=ROOT/plan['capture_root'];fit=ROOT/plan['fit_root']
 for key,folder in [('capture_root',cap),('fit_root',fit)]:
  assert folder.is_relative_to(ROOT/'crossplatform/round50') and (Path(plan[key])/'SEALED.json').as_posix() in plan['inputs_sha256']
 cs=read(cap/'SEALED.json');fs=read(fit/'SEALED.json')
 records=read(bound(fit,fs,'RECEIPTS_PRIVATE.json'));old={r['id']:r for r in read(ROOT/'research/round24/input/ALL336_PRIVATE.json')}
 selected=[];missing=[];eligible_counts={}
 for platform in ('drone','quad'):
  candidates=[r for r in records if r['platform']==platform and r['accepted'] and all(r['nonzero_parameter_updates'][str(s)] for s in (1,3))]
  eligible_counts[platform]={'accepted':sum(r['platform']==platform and r['accepted'] for r in records),'nonzero_s1_and_s3':len(candidates)}
  if candidates:selected.append(min(candidates,key=lambda r:hashlib.sha256((plan['selection_salt']+r['id_PRIVATE']).encode()).hexdigest()))
  else:missing.append(platform)
 out=ROOT/plan['output_root'];assert out.is_relative_to(ROOT/'crossplatform/round50') and not out.exists();out.mkdir(parents=True)
 dump(out/'SELECTED_BEFORE_FORWARD_PRIVATE.json',{'rows':selected,'no_nonzero_accepted_episode_platforms':missing,'eligible_counts':eligible_counts,'rule':'prelocked_hash_among_pseudofeedback_accepted_with_actual_nonzero_states_not_taskGT'})
 if not selected:
  seal(out,'R50_EPISODIC_CLOSURE_NOT_AVAILABLE_NO_NONZERO_ACCEPTED_STATES',n=0,grounder_forwards=0,no_GT=True);return
 sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
 from run_diagnostic import load_model,mask_for,fingerprint
 from runner_helpers import sample_points
 import torch
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.manual_seed(0);np.random.seed(0);random.seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));model.eval().requires_grad_(False)
 fp=fingerprint(model);outside0=outside(model);projection={n:x.detach().clone() for n,x in model.contrastive_align_projection_image.state_dict().items()}
 calls=0;recs=[];start=time.perf_counter()
 try:
  for chosen in selected:
   sid=chosen['id_PRIVATE'];row=old[sid];capt=npz(bound(cap,cs,'samples_PRIVATE/'+sid+'.npz'));prediction=npz(bound(fit,fs,'samples_PRIVATE/'+sid+'.npz'))
   case=read(bound(fit,fs,'cases_PRIVATE/'+sid+'.json'))
   structure={'groupsA':case['groupsA_PRIVATE'],'groupsB':case['groupsB_PRIVATE'],'repsA':prediction['A_reps'],
    'membershipA':prediction['A_membership'],'mapping':{int(x):int(y) for x,y in zip(prediction['match_A'],prediction['match_B'])},
    'source_group':case['source_group'],'pre_A_condition':case['pre_A_condition'],'q0_group_has_B_mapping':case['q0_group_has_B_mapping']}
   fields={}
   for step in (1,3):
    state=npz(bound(fit,fs,'states_PRIVATE/'+sid+'_s'+str(step)+'.npz'))
    model.contrastive_align_projection_image.load_state_dict({n:torch.tensor(v,device='cuda') for n,v in state.items()},strict=True)
    for name in ('A','B'):
     r=row if name=='A' else dict(row,seed=bseed(row,plan['B_sampling_salt']))
     points,indices,_=sample_points(r);assert np.array_equal(indices,capt[name+'_point_indices']) and hashlib.sha256(points.tobytes()).hexdigest()==str(capt[name+'_point_sha256'])
     text=row['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');assert np.array_equal(np.asarray(mask,bool),capt[name+'_token_mask'])
     tm=torch.as_tensor(mask,device='cuda',dtype=torch.bool);calls+=1;assert calls<=8
     with torch.inference_mode():o=model({'point_clouds':torch.tensor(points,device='cuda')[None],'text':[text]})
     p=((((o['last_proj_queries']@o['proj_tokens'].transpose(-1,-2))/.07).softmax(-1))*tm[None,None]).sum(-1)[0].cpu().numpy()
     b=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy();cached=prediction[name+'_p_s'+str(step)]
     exact(b,capt[name+'_boxes'],'installed_head_frozen_all256_boxes_'+name)
     er=float(np.max(abs(p-cached)));assert er<=plan['cached_full_max_abs_bound'] and int(p.argmax())==int(cached.argmax())
     fields[f'{name}_s{step}_native_p']=p;fields[f'{name}_s{step}_all256_boxes']=b
     recs.append({'id_PRIVATE':sid,'platform':row['platform'],'step':step,'view':name,'max_abs_error':er,'argmax_exact':True,'all256_source_boxes_exact':True})
    endpoint=consensus(fields[f'A_s{step}_native_p'],fields[f'B_s{step}_native_p'],prediction['A_fixed_VPR'],prediction['B_fixed_VPR'],structure)
    arm=prediction['arm_names'].tolist().index('sameinfo_consensus'+str(step));assert endpoint['query']==int(prediction['queries'][arm]),'STOP_installed_head_final_consensus_endpoint_changed'
    exact(capt['A_boxes'][endpoint['query']],prediction['selected_original_boxes'][arm],'installed_head_final_canonical_box')
    fields[f's{step}_final_consensus_scores']=endpoint['scores']
    recs.append({'id_PRIVATE':sid,'platform':row['platform'],'step':step,'view':'FINAL_TWO_VIEW_CONSENSUS',
     'max_abs_error':float(np.max(abs(endpoint['scores']-prediction['scores'][arm]))),'final_query_exact':True,
     'canonical_A_selected_box_exact':True})
   np.savez_compressed(out/(sid+'_FULL_CLOSURE_PRIVATE.npz'),**fields)
   model.contrastive_align_projection_image.load_state_dict(projection,strict=True)
  assert outside(model)==outside0 and fingerprint(model)==fp and calls==4*len(selected)
 except BaseException as e:
  dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'actual_forwards':calls,'completed_checks':len(recs),'no_GT_or_optimizer':True});raise
 dump(out/'RESULT_PRIVATE.json',{'state':'PASS_R50_ACTUAL_EPISODIC_INSTALLED_HEAD_CLOSURE','checks':recs,'no_nonzero_accepted_episode_platforms':missing,'eligible_counts':eligible_counts,'new_grounder_forwards':calls,'outside_projection_unchanged':True,'source_fully_restored':True})
 seal(out,'R50_EPISODIC_INSTALLED_HEAD_CLOSURE_COMPLETE_NO_GT',n=len(selected),grounder_forwards=calls,seconds=time.perf_counter()-start,plan_sha256=sha(a.plan))
if __name__=='__main__':main()
