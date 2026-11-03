"""Same-source fixed/EMA self-training, paired sampling controls, no labels."""
import argparse,sys,time,copy,torch
from common57 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import native_matching,strict_runtime

def head(initial,device):
 m=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64)).to(device);m.load_state_dict({k:torch.as_tensor(v,device=device) for k,v in initial.items()});return m.eval()
def parameters_hash(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode()+v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def state(m):return {k:v.detach().cpu().numpy().copy() for k,v in m.state_dict().items()}
def prepared(v,c,g,device):
 T=lambda a:torch.as_tensor(a,device=device)
 d={view:{'h':T(z[pre+'h'])[None],'tokens':T(z[pre+'tokens'])[None],'mask':T(z[pre+'token_mask']).bool()} for view,z,pre in [('A',v,'A_'),('B',v,'B_'),('C',c,'')]}
 d['base']=T(g['base']).double()
 for view in ['B','C']:d[view].update(W=T(g[view+'_W']).double(),mass=T(g[view+'_fixed_mass']).double())
 return d
def distribution(m,d,view):
 x=d[view];q=torch.nn.functional.normalize(m(x['h']),dim=-1);_,z=native_matching(q,x['tokens'],x['mask']);p=z.double().softmax(-1)
 if view=='A':return p
 mapped=x['W']@p;return x['mass']*mapped/mapped.sum().clamp_min(1e-300)+(1-x['mass'])*d['base']
def ensemble(m,d,views):return sum(distribution(m,d,v) for v in views)/len(views)
def loss(student,d,target):
 pred=distribution(student,d,'C');return (target*(target.clamp_min(1e-300).log()-pred.clamp_min(1e-300).log())).sum()
def train_step(s,t,opt,d,aug,kind):
 with torch.no_grad():target=ensemble(t,d,aug).detach();told={k:v.detach().clone() for k,v in t.named_parameters()}
 opt.zero_grad(set_to_none=True);L=loss(s,d,target);L.backward();gn=float(torch.sqrt(sum(v.grad.square().sum() for v in s.parameters())));torch.nn.utils.clip_grad_norm_(s.parameters(),1);opt.step()
 with torch.no_grad():
  if kind=='ema':
   for name,v in t.named_parameters():v.copy_(.99*told[name]+.01*dict(s.named_parameters())[name])
  assert all(v.grad is None and not v.requires_grad for v in t.parameters());after=float(loss(s,d,target));td=float(torch.sqrt(sum((v-told[n]).square().sum() for n,v in t.named_parameters())))
 return {'KL_before':float(L.detach()),'KL_after_same_detached_target':after,'gradient_norm':gn,'teacher_step_L2':td,'target_requires_grad':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','cases_PRIVATE','fitted_PRIVATE']:(out/f).mkdir()
 strict_runtime();torch.manual_seed(0);rows=read(ROOT/p['manifest']);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');assert sum(v.size for v in initial.values())==184960;source=head(initial,'cuda').requires_grad_(False);init_hash=parameters_hash(source);cache={};records={};results={};snapshots={};start=time.perf_counter();steps=0;zero_controls=[]
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';v=load(ROOT/p['AB_root']/rel);c=load(ROOT/p['capture_root']/rel);g=load(ROOT/p['association_root']/rel);cache[sid]=(v,c,g);d=prepared(v,c,g,'cuda')
  with torch.no_grad():pa=distribution(source,d,'A');ab=ensemble(source,d,'AB');abc=ensemble(source,d,'ABC');q=torch.nn.functional.normalize(source(d['A']['h']),dim=-1);prob,_=native_matching(q,d['A']['tokens'],d['A']['mask']);assert torch.equal(prob,torch.as_tensor(v['A_scores'],device='cuda'))
  results[sid]={'source':v['A_scores'],'frozen_AB':ab.cpu().numpy(),'frozen_ABC':abc.cpu().numpy()};records[sid]={'platform':row['platform'],'episodic':{},'online':{},'source_bridge_bitexact':True};snapshots[sid]={}
  for kind in ['fixed','ema']:
   s=head(initial,'cuda').requires_grad_(True);t=head(initial,'cuda').requires_grad_(False);opt=torch.optim.Adam(s.parameters(),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0);trace=[]
   if len(zero_controls)<4:
    ps=distribution(s,d,'A');target=ps.detach();L=(target*(target.log()-ps.log())).sum();L.backward();zero_controls.append({'platform':row['platform'],'gradient_norm':float(torch.sqrt(sum(v.grad.square().sum() for v in s.parameters())))});opt.zero_grad(set_to_none=True)
   for k in range(1,17):
    trace.append(train_step(s,t,opt,d,'AB',kind));steps+=1
    if k in [1,3,16]:
     arm=f'epi_{kind}_student_{k}'
     with torch.no_grad():results[sid][arm]=ensemble(s,d,'ABC').cpu().numpy()
     for n,x in state(s).items():snapshots[sid][arm+'__'+n]=x
     if kind=='ema':
      arm=f'epi_ema_teacher_{k}'
      with torch.no_grad():results[sid][arm]=ensemble(t,d,'ABC').cpu().numpy()
      for n,x in state(t).items():snapshots[sid][arm+'__'+n]=x
   records[sid]['episodic'][kind]={'trace':trace,'source_reset_before_case':True,'student_final_hash':parameters_hash(s),'teacher_final_hash':parameters_hash(t)};del s,t,opt
  if len(results)%32==0:print('EPISODIC57',len(results),336,steps,flush=True)
 # Each platform is an independently initialized deterministic, nonchronological stream.
 orders={pl:sorted([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R57_online_order_v1|'+r['id']).encode()).hexdigest()) for pl in ['drone','quad']}
 for kind in ['fixed','ema']:
  for aug in ['A','AB']:
   for pl,stream in orders.items():
    s=head(initial,'cuda').requires_grad_(True);t=head(initial,'cuda').requires_grad_(False);opt=torch.optim.Adam(s.parameters(),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0)
    for index,row in enumerate(stream):
     sid=row['id'];d=prepared(*cache[sid],'cuda');entry={'stream_index':index,'student_pre_hash':parameters_hash(s),'teacher_pre_hash':parameters_hash(t)}
     for timing in ['pre','post']:
      if timing=='post':entry.update(train_step(s,t,opt,d,aug,kind));steps+=1
      arm=f'online_{kind}_{aug}_student_{timing}'
      with torch.no_grad():results[sid][arm]=ensemble(s,d,'ABC').cpu().numpy()
      for n,x in state(s).items():snapshots[sid][arm+'__'+n]=x
      if kind=='ema':
       arm=f'online_ema_{aug}_teacher_{timing}'
       with torch.no_grad():results[sid][arm]=ensemble(t,d,'ABC').cpu().numpy()
       for n,x in state(t).items():snapshots[sid][arm+'__'+n]=x
     entry.update(student_post_hash=parameters_hash(s),teacher_post_hash=parameters_hash(t),optimizer_step=int(next(iter(opt.state.values()))['step']));records[sid]['online'][kind+'_'+aug]=entry
    del s,t,opt
   print('ONLINE57',kind,aug,steps,flush=True)
 assert steps==12096 and parameters_hash(source)==init_hash;statuses=[]
 for row in rows:
  sid=row['id'];v,c,g=cache[sid];assert set(results[sid])==set(ARMS);scores=np.stack([results[sid][arm] for arm in ARMS]);queries=scores.argmax(1).astype(np.int64);np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS),scores=scores,queries=queries,source_boxes=v['A_boxes'],source_scores=v['A_scores'],selected_original_boxes=v['A_boxes'][queries]);np.savez_compressed(out/'fitted_PRIVATE'/(sid+'.npz'),**snapshots[sid]);dump(out/'cases_PRIVATE'/(sid+'.json'),records[sid]);statuses.append({'id':sid,'arm_statuses':{arm:{'unknown':bool(float(g['C_fixed_mass'])==0),'fallback':bool(float(g['C_fixed_mass'])==0),'no_parameter_update':arm in ['source','frozen_AB','frozen_ABC']} for arm in ARMS}})
 dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'ONLINE_ORDERS_PRIVATE.json',{pl:[r['id'] for r in stream] for pl,stream in orders.items()});dump(out/'IDENTICAL_INPUT_ZERO_GRADIENT_CONTROLS.json',zero_controls);costs={'n':336,'optimizer_steps_total_completed':steps,'matching_trainable_parameters':184960,'new_full_grounder_forwards':0,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'source_head_restored_exact':True};dump(out/'COSTS.json',costs);seal(out,'R57_ALL_EMA_FIXED_AUGMENTATION_PREDICTIONS_SEALED_NO_TASK_LABELS',**costs,arm_names=ARMS,plan_sha256=sha(a.plan));print(costs,flush=True)
if __name__=='__main__':main()
