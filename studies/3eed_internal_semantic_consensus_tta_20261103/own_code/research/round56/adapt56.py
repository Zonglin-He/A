"""Episodic matching-MLP self-distillation from frozen initial internal semantics."""
import argparse,sys,time,torch
from common56 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint,native_matching

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');a=ap.parse_args();p=read(a.plan);pin(p);rows=read(ROOT/p['manifest']);out=ROOT/p['output_root'];binding=sha(a.plan)
 if out.exists():assert a.resume and read(out/'RUN_BINDING.json')['plan_sha256']==binding and not (out/'SEALED.json').exists()
 else:out.mkdir(parents=True);dump(out/'RUN_BINDING.json',{'plan_sha256':binding})
 for d in ['samples_PRIVATE','cases_PRIVATE','fitted_PRIVATE']:(out/d).mkdir(exist_ok=True)
 torch.manual_seed(0);start=time.perf_counter();model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);module=model.contrastive_align_projection_image;module.requires_grad_(True);assert sum(x.numel() for x in module.parameters())==184960;initial={n:x.detach().clone() for n,x in module.named_parameters()};versions={n:x._version for n,x in model.named_parameters() if not x.requires_grad};done=[];statuses=[]
 ls=read(ROOT/p['latent_root']/'SEALED.json');cs=read(ROOT/p['consensus_root']/'SEALED.json')
 def reset():
  model.eval()
  with torch.no_grad():
   for n,x in module.named_parameters():x.copy_(initial[n]);x.grad=None
 def forward(h,tokens,mask):
  q=torch.nn.functional.normalize(module(h),dim=-1);prob,z=native_matching(q,tokens,mask);return prob,z
 def delta():return float(torch.sqrt(sum((x.detach()-initial[n]).square().sum() for n,x in module.named_parameters())))
 def check():
  for n,x in model.named_parameters():
   if n in versions:assert not x.requires_grad and x.grad is None and x._version==versions[n],n
  assert all(not m.training for m in model.modules())
 try:
  for row in rows:
   sid=row['id'];sf=out/'samples_PRIVATE'/(sid+'.npz');cf=out/'cases_PRIVATE'/(sid+'.json')
   if cf.exists():
    rec=read(cf);assert a.resume and sha(sf)==rec['sample_sha256'];done.append(sid);statuses.append(rec['status_row']);continue
   rel='samples_PRIVATE/'+sid+'.npz';assert sha(ROOT/p['latent_root']/rel)==ls['files'][rel] and sha(ROOT/p['consensus_root']/rel)==cs['files'][rel];l=load(ROOT/p['latent_root']/rel,['h','tokens','token_mask','boxes','scores']);c=load(ROOT/p['consensus_root']/rel,['static_arm_names','static_scores','teacher','shuffled_teacher','subject_phrase_unknown_mass','reference_phrase_unknown_masses']);T=lambda x:torch.as_tensor(x,device='cuda');h=T(l['h'])[None];tokens=T(l['tokens'])[None];mask=T(l['token_mask']).bool();reset()
   with torch.no_grad():prob,z=forward(h,tokens,mask);assert torch.equal(prob,T(l['scores']))
   result={str(name):value.copy() for name,value in zip(c['static_arm_names'],c['static_scores'])};result['zero_step']=prob.cpu().numpy();diag={};fitted={}
   for prefix,key in [('', 'teacher'),('shuffle_','shuffled_teacher')]:
    reset();target=T(c[key]).detach().double();assert not target.requires_grad and abs(float(target.sum())-1)<1e-12
    optimizer=torch.optim.Adam(module.parameters(),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0);trace=[]
    def loss(z):return (target*(target.clamp_min(1e-300).log()-torch.log_softmax(z.double(),-1))).sum()
    with torch.no_grad():_,z=forward(h,tokens,mask);initial_loss=float(loss(z))
    for step in [1,2,3]:
     optimizer.zero_grad(set_to_none=True);_,z=forward(h,tokens,mask);L=loss(z);L.backward();grad=float(torch.sqrt(sum(v.grad.square().sum() for v in module.parameters())));torch.nn.utils.clip_grad_norm_(module.parameters(),1);optimizer.step();check()
     with torch.no_grad():
      prob,z=forward(h,tokens,mask);record={'step':step,'KL_before':float(L),'KL_after':float(loss(z)),'gradient_norm':grad,'parameter_delta_L2':delta(),'max_native_probability_change':float((prob-T(l['scores'])).abs().max()),'candidate_distribution_L1_vs_consensus':float((z.double().softmax(-1)-target).abs().sum())};trace.append(record)
      if step in [1,3]:
       arm=prefix+'tta'+str(step);result[arm]=prob.cpu().numpy().copy()
       for name,x in module.named_parameters():fitted[arm+'__'+name]=x.detach().cpu().numpy().copy()
    diag[prefix+'TTA']={'initial_KL':initial_loss,'trace':trace,'stop_gradient_initial_consensus':True};optimizer.zero_grad(set_to_none=True);del optimizer
   assert set(result)==set(ARMS);stack=np.stack([result[x] for x in ARMS]);q=stack.argmax(1).astype(np.int64);np.savez_compressed(sf,arm_names=np.asarray(ARMS),scores=stack,queries=q,source_boxes=l['boxes'],source_scores=l['scores'],selected_original_boxes=l['boxes'][q]);np.savez_compressed(out/'fitted_PRIVATE'/(sid+'.npz'),**fitted)
   status={'id':sid,'arm_statuses':{arm:{'unknown':False,'fallback':False,'no_parameter_update':False if arm in ['tta1','tta3','shuffle_tta1','shuffle_tta3'] else True} for arm in ARMS}};status['arm_statuses']['phrase_subject']['unknown']=bool(float(c['subject_phrase_unknown_mass'])>=1-1e-12);status['arm_statuses']['phrase_role']['unknown']=bool(float(c['subject_phrase_unknown_mass'])>=1-1e-12 and all(c['reference_phrase_unknown_masses']>=1-1e-12));rec={'state':'COMPLETED','sample_sha256':sha(sf),'fitted_sha256':sha(out/'fitted_PRIVATE'/(sid+'.npz')),'platform':row['platform'],'statistics':diag,'status_row':status,'source_bridge_bitexact':True};dump(cf,rec);done.append(sid);statuses.append(status)
   if len(done)%16==0:print('ADAPT56',len(done),336,flush=True)
   if a.limit is not None and len(done)>=a.limit:break
  reset();module.requires_grad_(False);assert fingerprint(model)==fp
  counts={'n':len(done),'optimizer_steps_total_completed':len(done)*6,'matching_trainable_parameters':184960,'source_fingerprint':fp,'source_restored_exact':True,'seconds_this_process':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'new_full_grounder_forwards':0}
  if len(done)==336:dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',counts);seal(out,'R56_FIXED_SEMANTIC_CONSENSUS_AND_MATCHING_TTA_NO_GT',**counts,arm_names=ARMS,plan_sha256=binding)
  else:dump(out/'ENGINEERING_STOP.json',counts)
  print(json.dumps(counts),flush=True)
 except BaseException as e:dump(out/('FAILED_'+str(time.time_ns())+'.json'),{'type':type(e).__name__,'error':str(e),'completed':len(done),'GT_read':False});raise
if __name__=='__main__':main()
