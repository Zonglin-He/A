"""Independent actual decoder FD packets; no fit/optimizer/GT/DINO calls."""
import collections,gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_observation_response_projection_common_v1 import *

def gpu_readback():
 import torch
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old();verify();assert read(BASE/'QUALIFICATION_BARRIER.json')['status']=='sealed'
 from scripts.run_stvg_opd_expanded_after_p8_priority001 import gpu
 handle=gpu()
 from scripts.run_decota_paper_main_v1 import model_for,read_row,frames_for
 from scripts.stvg_opd_paper_inputs_v1 import observation
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from vg_tta.c1_enabling_tricks_v1 import TrickReplay
 from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
 from methods.decota_final_simplified_v1.tensors import state_hash
 d=read(BASE/'DESIGN_LOCK.json');rr=[r for r in d['records'] if r['index'] in d['qualification_indices']]
 cache=collections.OrderedDict();model=None;ds=None;count=0;captures=0;bindings={};begin=time.time()
 def values(rp,obs,unvisited):
  head=[];hook=rp.base.decoder.decoder.bbox_embed.register_forward_hook(lambda module,args,out:head.append(out))
  try:boxes=rp.values()['boxes']
  finally:hook.remove()
  assert len(head)==12
  native=torch.stack([head[5+(i%2)*6][i//2,0] for i in range(len(rp.ids))]).float()
  assert torch.equal(native.sigmoid(),boxes)
  mu,_=proposed_coordinates(boxes[obs],native[obs])
  return torch.cat((mu.flatten(),boxes[unvisited].flatten())),boxes
 def set_parameters(rp,state,vec,scale):
  offset=0
  with torch.no_grad():
   for name,p in rp.named:
    delta=torch.tensor(scale*vec[offset:offset+p.numel()],device=p.device,dtype=p.dtype).reshape_as(p)
    p.copy_(state[name].to(p)+delta);offset+=p.numel()
  assert offset==1792
 try:
  for r in rr:
   standby()
   if ds!=r['dataset']:
    if model is not None:assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(20260920);np.random.seed(20260920);model=model_for(r['source']);mh=model._fixed_full_state_hash;ds=r['dataset']
   row=read_row(ds,r['query_ordinal']);frames,ids,_=frames_for(ds,row,cache);shifted,pixel,spec=observation(ds,row,'clean',frames)
   batch,records,base=frozen_forward(model,shifted,row);captures+=1
   inp=normalized(load(ROOT/r['originals'][0]['input_path']));assert pixel==r['pixel_sha256']==inp['pixel_sha256'] and ids==inp['frame_ids']
   assert mh==r['source_model_sha256'] and spec is None
   A=load(ROOT/r['A_path'])['fit'];del frames,shifted,batch
   for k in [0,r['config']['steps']//2,r['config']['steps']-1]:
    witness=load(BASE/'qualification'/f"{r['index']:05}"/f'P_k{k}_repeat0.pt')['witness']
    if not witness['updated']:continue
    state=A['path'][k]['state'];base.restore(state);rp=TrickReplay(base,ids,row['key'],{})
    obs=A['positions'];unvisited=[i for i in range(len(ids)) if i not in set(r['visited'])]
    J=np.concatenate((witness['J_O'].numpy(),witness['J_U'].numpy()),axis=0)
    y,boxes=values(rp,obs,unvisited);assert torch.equal(boxes.cpu(),A['path'][k]['boxes'])
    gen=np.random.default_rng(SEED+r['index']*100+k)
    cot=gen.normal(size=len(y));cot/=np.linalg.norm(cot)
    grads=torch.autograd.grad(y,[p for n,p in rp.named],grad_outputs=torch.tensor(cot,device=y.device,dtype=y.dtype),retain_graph=True)
    actual_vjp=torch.cat([g.flatten() for g in grads]).detach().cpu()
    directions=[gen.normal(size=1792),witness['d_A'].numpy()];pairs=[]
    for di,vec in enumerate(directions):
     if not np.linalg.norm(vec):continue
     vec=vec/np.linalg.norm(vec)
     for h in [.002,.001]:
      set_parameters(rp,state,vec,h)
      with torch.no_grad():yp,_=values(rp,obs,unvisited)
      set_parameters(rp,state,vec,-h)
      with torch.no_grad():ym,_=values(rp,obs,unvisited)
      pairs.append(dict(direction=di,h=h,vector=torch.tensor(vec),plus=yp.detach().cpu(),minus=ym.detach().cpu()))
    rp.restore(state)
    z=dict(index=r['index'],dataset=ds,boundary=k,output_before=y.detach().cpu(),cotangent=torch.tensor(cot),actual_VJP=actual_vjp,
     finite_differences=pairs,qualification_witness_sha256=sha(BASE/'qualification'/f"{r['index']:05}"/f'P_k{k}_repeat0.pt'),
     pixel_sha256=pixel,source_model_sha256=mh,frame_ids=ids,complete_grid=True,no_optimizer_call=True,
     runtime_sha256=sha(BASE/'RUNTIME.json'),root_runtime_sha256=sha(BASE/'decoder_derivative_root006/RUNTIME.json'),GT_read=False)
    p=BASE/'decoder_derivative_root006'/'packets'/f"{r['index']:05}_{k:02}.pt";save(p,z);bindings[str(p.relative_to(ROOT))]=sha(p);count+=1
    status(BASE/'decoder_derivative_root006/STATUS.json',dict(status='running_actual_independent_decoder_packets',dataset=ds,done=count,total=21,pid=os.getpid(),GT_read=False,time=time.time()))
    print('INDEPENDENT_DECODER_PACKET',count,21,ds,k,flush=True);del rp,y,boxes,grads,z
   del base,A;gc.collect();torch.cuda.empty_cache()
  assert captures==8 and count==21 and state_hash(model.state_dict())==mh
  verify();standby()
  write(BASE/'decoder_derivative_root006/BARRIER.json',dict(status='sealed',files=bindings,count=count,total=21,actual_fresh_captures=8,
   actual_optimizer_steps=0,actual_DINO_calls=0,scope='independent FD plus/minus vectors and VJP contraction at all nonempty fixed contexts',
   root_runtime_sha256=sha(BASE/'decoder_derivative_root006/RUNTIME.json'),GT_read=False,elapsed_seconds=time.time()-begin,time=time.time()))
 finally:
  if model is not None:del model
  cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()

def cpu_readback():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 b=read(BASE/'decoder_derivative_root006/BARRIER.json');assert b['status']=='sealed' and b['count']==21
 checked=0;contexts=0;matrix_coordinates=0;readback=[]
 for rel,h in b['files'].items():
  assert sha(ROOT/rel)==h;z=load(ROOT/rel);w=load(BASE/'qualification'/f"{z['index']:05}"/f"P_k{z['boundary']}_repeat0.pt")['witness']
  assert z['qualification_witness_sha256']==sha(BASE/'qualification'/f"{z['index']:05}"/f"P_k{z['boundary']}_repeat0.pt")
  J=np.concatenate((w['J_O'].numpy(),w['J_U'].numpy()),axis=0);matrix_coordinates+=J.size
  actual=z['actual_VJP'].numpy().astype(float);expected=J.T@z['cotangent'].numpy();vjp=np.linalg.norm(actual-expected)
  assert abs(vjp-w['derivative_checks']['VJP_error'])<1e-10
  for p in z['finite_differences']:
   pred=J@p['vector'].numpy();fd=(p['plus'].numpy().astype(float)-p['minus'].numpy().astype(float))/(2*p['h'])
   error=np.linalg.norm(fd-pred);tolerance=1e-4+.02*np.linalg.norm(pred)
   oldrow=next(t for t in w['derivative_checks']['directions'][p['direction']]['finite_differences'] if t['h']==p['h'])
   assert abs(error-oldrow['absolute_error'])<1e-10 and abs(tolerance-oldrow['tolerance'])<1e-10
   assert (error<=tolerance)==oldrow['pass_check']
   checked+=1;readback.append(dict(parent_id=z['index'],dataset=z['dataset'],boundary=z['boundary'],direction=p['direction'],
    h=p['h'],absolute_error=float(error),relative_error=float(error/max(1e-12,np.linalg.norm(pred))),tolerance=float(tolerance),pass_check=bool(error<=tolerance)))
  contexts+=1;assert sha(ROOT/rel)==h
 assert contexts==21 and checked==84
 write(BASE/'ROOT_INDEPENDENT_DECODER_DERIVATIVE_READBACK.json',dict(status='pass',scope='all actual independently recaptured physical FD vectors/VJP contractions/complete stored AD matrices; failure results exact readback, not numerical qualification pass',
  count=contexts,total=21,FD_checks=checked,FD_pass=sum(x['pass_check'] for x in readback),Jacobian_coordinates=matrix_coordinates,
  actual_fresh_captures=8,new_optimizer_steps=0,new_DINO_calls=0,CPU_only=True,GT_read=False,time=time.time()))
 write(OUT/'INDEPENDENT_FD_READBACK.json',readback)
 status(BASE/'decoder_derivative_root006/STATUS.json',dict(status='complete',done=21,total=21,CPU_only=True,GT_read=False,time=time.time()))

if __name__=='__main__':
 try:
  if sys.argv[1]=='gpu':gpu_readback()
  else:cpu_readback()
 except BaseException:
  p=BASE/'decoder_derivative_root006/failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc());raise
