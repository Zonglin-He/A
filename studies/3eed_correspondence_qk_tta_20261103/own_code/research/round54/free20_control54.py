"""Stronger same-information no-network-update control, fixed20 iterations.

The first3 free steps are reconstructed to continue Adam state; they must
reproduce the already-saved3step variables. No target labels are opened.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from adapt54 import Interface,make_context,prepare_views,source_states,objective,attention_forward,assignment,diagnostics,fingerprint_in_order
from model51 import load_source,fingerprint,PREFIX_KEYS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h,rel
 rows=read(ROOT/p['manifest']);root=ROOT/p['output_root'];assert not root.exists();root.mkdir(parents=True)
 for d in ['samples_PRIVATE','cases_PRIVATE','fitted_PRIVATE']:(root/d).mkdir()
 torch.manual_seed(0);np.random.seed(0);start=time.perf_counter();model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);order=list(model.state_dict());interface=Interface(model);latent=ROOT/p['latent_root'];ls=read(latent/'SEALED.json');obsroot=ROOT/p['observation_root'];os=read(obsroot/'SEALED.json');mainroot=ROOT/p['main_root'];ms=read(mainroot/'SEALED.json');statuses=[];max_reconstruction_error=0.
 try:
  for row in rows:
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(latent/rel)==ls['files'][rel] and sha(obsroot/rel)==os['files'][rel]
   with np.load(latent/rel,allow_pickle=False) as z:arrays={k:z[k].copy() for k in ['boxes','scores','logodds','h','tokens','token_mask',*PREFIX_KEYS] if k in z.files}
   with np.load(obsroot/rel,allow_pickle=False) as z:obs_np={k:z[k].copy() for k in ['allowed','geometry_cost','RGB_cost','RGB_shuffle_cost','edge_pairs','edge_affinity','edge_shuffle_affinity','point_RGB_known','query_prior','view_A','view_B']}
   fr='fitted_PRIVATE/'+sid+'.npz';assert sha(mainroot/fr)==ms['files'][fr]
   with np.load(mainroot/fr,allow_pickle=False) as z:old={k:z[k].copy() for k in ['free_3','free_RGBshuffle_3']}
   interface.reset(False);ctx=make_context(model,arrays);device=ctx['device'];obs={k:torch.as_tensor(v,device=device,dtype=torch.float32 if v.dtype.kind=='f' else None) for k,v in obs_np.items()};views,shared=prepare_views(obs,device);source=source_states(model,ctx,obs,views);scores=[];details={};fitted={};tick=time.perf_counter()
   for shuffle in [False,True]:
    name='free_RGBshuffle_20' if shuffle else 'free_20';oldname='free_RGBshuffle_3' if shuffle else 'free_3';U=nn.Parameter(torch.zeros((256,1024),device=device));optimizer=torch.optim.Adam([U],lr=.01,betas=(.9,.999),eps=1e-8);history=[]
    with torch.no_grad():initial,_,_,_=objective(model,ctx,obs,views,shared,source,U,shuffle,False)
    for step in range(1,21):
     optimizer.zero_grad(set_to_none=True);loss,terms,_,_=objective(model,ctx,obs,views,shared,source,U,shuffle,False);assert torch.isfinite(loss);loss.backward();grad=float(U.grad.detach().norm());torch.nn.utils.clip_grad_norm_([U],1.);optimizer.step();interface.check()
     if step==3:
      error=float(np.max(np.abs(U.detach().cpu().numpy()-old[oldname])));max_reconstruction_error=max(error,max_reconstruction_error);assert error<=1e-6,error
     history.append({'step':step,'loss_before_step':float(loss.detach()),'gradient_norm_before_clipping':grad})
    with torch.no_grad():final,terms,current,S=objective(model,ctx,obs,views,shared,source,U,shuffle,False);diag=diagnostics(terms,S,source[0]['S'],ctx,current,float(U.detach().norm()),history[-1]['gradient_norm_before_clipping']);scores.append(current['scores'].cpu().numpy().copy())
    details[name]={'initial_loss':float(initial),'final_loss':float(final),'history':history,'final_diagnostics':diag,'no_network_parameter_update':True};fitted[name]=U.detach().cpu().numpy().copy();optimizer.zero_grad(set_to_none=True);del optimizer,U
   stack=np.stack(scores);q=stack.argmax(1).astype(np.int64);np.savez_compressed(root/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.asarray(['free_20','free_RGBshuffle_20']),scores=stack,queries=q,source_boxes=arrays['boxes'],source_scores=arrays['scores'],selected_original_boxes=arrays['boxes'][q]);np.savez_compressed(root/'fitted_PRIVATE'/(sid+'.npz'),**fitted);status={'id':sid,'arm_statuses':{a:{'unknown':bool(not obs_np['point_RGB_known'].any() or not obs_np['allowed'].any()),'fallback':False,'no_parameter_update':True} for a in ['free_20','free_RGBshuffle_20']}};statuses.append(status);dump(root/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'platform':row['platform'],'details':details,'seconds':time.perf_counter()-tick,'status_row':status})
   if len(statuses)%16==0:print('FREE20_CONTROL54',len(statuses),336,flush=True)
  interface.reset(False);interface.check();parametrize.remove_parametrizations(interface.module,'in_proj_weight',leave_parametrized=False);model.requires_grad_(False);assert fingerprint_in_order(model,order)==fp
  dump(root/'READOUT_STATUS_PRIVATE.json',statuses);cost={'n':len(statuses),'optimizer_steps':336*40,'first_three_step_reconstruction_max_abs_error':max_reconstruction_error,'new_full_grounder_forwards':0,'source_model_unchanged':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(root/'COSTS.json',cost);dump(root/'SEALED.json',{'state':'R54_FIXED20_STRONG_DIRECT_CONTROL_PREDICTIONS_NO_GT',**cost,'arm_names':['free_20','free_RGBshuffle_20'],'plan_sha256':sha(a.plan),'files':{f.relative_to(root).as_posix():sha(f) for f in sorted(root.rglob('*')) if f.is_file()}});print(json.dumps(cost),flush=True)
 except BaseException as e:dump(root/('FAILED_'+str(time.time_ns())+'.json'),{'type':type(e).__name__,'error':str(e),'n':len(statuses),'no_auto_retry':True,'no_GT_read':True});raise
if __name__=='__main__':main()
