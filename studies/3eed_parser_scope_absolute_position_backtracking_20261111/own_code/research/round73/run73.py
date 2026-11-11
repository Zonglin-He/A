"""One transactional Adam proposal; fixed evidence, no labels or expert scores."""
from common73 import *
from constraints72 import direct
from adapt72 import logodds,distribution,objective
import torch,copy,time
sys.path.insert(0,str(ROOT/'research/round58'));from numerics58 import head,state,optimizer,phash,ohash
def fit(initial,z,N):
 p0=torch.tensor(z['scores'].astype(float));p0/=p0.sum();N=torch.tensor(N).bool();m=head(initial);opt=optimizer(m);base=phash(m);optbase=ohash(opt);active=bool(N.any() and not N.all() and p0[N].sum()>1e-12)
 if not active:return p0.numpy(),state(m),{'active':False,'accepted':False,'scale':0.,'trials':[],'parameter_change_L2':0.,'gradient_norm':0.,'before_sha256':base,'after_sha256':base,'optimizer_before_sha256':optbase,'optimizer_after_sha256':optbase}
 h=torch.tensor(z['h']);t=torch.tensor(z['tokens']);mask=torch.tensor(z['token_mask']).bool()
 with torch.no_grad():z0=logodds(m,h,t,mask).double().clone();old={k:v.detach().clone() for k,v in m.named_parameters()}
 oldopt=copy.deepcopy(opt.state_dict());L=objective(distribution(m,h,t,mask,z0,p0),p0,N);before=float(L.detach());opt.zero_grad(set_to_none=True);L.backward();gn=float(torch.sqrt(sum(v.grad.square().sum() for v in m.parameters())));torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();proposal={k:v.detach().clone() for k,v in m.named_parameters()};trials=[];accepted=False;scale=0.
 with torch.no_grad():
  for j in range(9):
   alpha=2.**(-j)
   for k,v in m.named_parameters():v.copy_(old[k]+alpha*(proposal[k]-old[k]))
   after=float(objective(distribution(m,h,t,mask,z0,p0),p0,N));trials.append({'scale':alpha,'loss':after if np.isfinite(after) else None})
   if np.isfinite(after) and after<=before-max(1e-12,1e-6*abs(before)):accepted=True;scale=alpha;break
  if not accepted:
   for k,v in m.named_parameters():v.copy_(old[k])
   opt.load_state_dict(oldopt)
  prob=distribution(m,h,t,mask,z0,p0).numpy().copy();after=float(objective(torch.tensor(prob),p0,N));delta=float(np.sqrt(sum(np.square(v-initial[k]).sum() for k,v in state(m).items())))
 if accepted:assert after<=before-max(1e-12,1e-6*abs(before)) and delta>0
 else:assert phash(m)==base and ohash(opt)==optbase and delta==0 and abs(after-before)<1e-12
 opt.zero_grad(set_to_none=True);rec={'active':True,'accepted':accepted,'scale':scale,'trials':trials,'loss_before':before,'loss_after':after,'gradient_norm':gn,'parameter_change_L2':delta,'before_sha256':base,'after_sha256':phash(m),'optimizer_before_sha256':optbase,'optimizer_after_sha256':ohash(opt),'target_negative_set_fixed':True};return prob,state(m),rec
def main():
 p=read(H/'experiment/RUN_PLAN_LOCK.json');pin(p);out=ROOT/p['prediction_root'];ad=ROOT/p['adapt_root'];assert not out.exists() and not ad.exists();out.mkdir(parents=True);ad.mkdir(parents=True)
 for root,folders in [(out,['samples_PRIVATE']),(ad,['actual_weights_PRIVATE','cases_PRIVATE'])]:
  for folder in folders:(root/folder).mkdir()
 evroot=ROOT/p['evidence_root'];es=read(evroot/'SEALED.json');initial=load(ROOT/p['head_initial']);start=time.perf_counter();statuses=[];torch.set_num_threads(4);steps=accept=0
 for r in read(H/'input/MECHANISM64_PRIVATE.json'):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(evroot/rel)==es['files'][rel];ev=load(evroot/rel);z=prefix(r);p0=z['scores'].astype(float);p0/=p0.sum();ng=ev['query_group'];N=ev['main_N'][ng];prob,states,rec=fit(initial,z,N);steps+=int(rec['active']);accept+=int(rec['accepted']);np.savez_compressed(ad/'actual_weights_PRIVATE'/(sid+'.npz'),**states);dump(ad/'cases_PRIVATE'/(sid+'.json'),rec)
  # Baseline scores are attached only after evidence and the per-input fit.
  base=load(ROOT/'crossplatform/round71/predicted2640_attempt001'/rel);b=dict(zip(base['arm_names'],base['scores']));assert np.array_equal(base['source_boxes'],z['boxes']);scores={'source':b['source'],'old_R72':direct(p0,ev['old_N'][ng]),'parse_fix_direct':direct(p0,ev['parse_N'][ng]),'position_direct_opt':direct(p0,N),'position_TTA_backtrack1':prob,'position_mismatch_direct_opt':direct(p0,ev['mismatch_N'][ng]),'R59':b['R59'],'CLIP_reference':b['CLIP_direct']};score=np.stack([scores[k] for k in ARMS]);q=score.argmax(1);np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=score,queries=q,source_boxes=z['boxes'],selected_original_boxes=z['boxes'][q]);st={}
  for k in ARMS:
   up=k=='position_TTA_backtrack1';valid=ev['position_known'].any();key='old_N' if k=='old_R72' else 'parse_N' if k=='parse_fix_direct' else 'mismatch_N' if k=='position_mismatch_direct_opt' else 'main_N';inactive=not bool(ev[key].any())
   st[k]={'unknown':bool(not valid) if k.startswith('position_') else False,'fallback':(not rec['accepted']) if up else inactive if k not in ['source','R59','CLIP_reference'] else False,'no_parameter_update':not(up and rec['accepted'])}
  statuses.append({'id':sid,'arm_statuses':st})
 dump(ad/'COUNTS.json',{'n':64,'optimizer_proposals':steps,'accepted':accept,'rejected':steps-accept,'updated_parameters':184960,'seconds':time.perf_counter()-start,'GT_lookups':0});seal(ad,'R73_FIXED_EVIDENCE_SINGLE_BACKTRACKED_MLP_PROPOSAL',n=64);dump(out/'READOUT_STATUS_PRIVATE.json',statuses);seal(out,'R73_EIGHT_READOUTS_SEALED_BEFORE_TASK_LOOKUP',n=64,arms=ARMS);print(read(ad/'COUNTS.json'))
if __name__=='__main__':main()
