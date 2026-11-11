"""Region contrastive TTA of the existing matching head, no C/expert/GT access."""
from batchcommon74 import *
import torch,copy,time
sys.path.insert(0,str(ROOT/'research/round58'));from numerics58 import head,state,optimizer,phash,ohash
def query_rep(m,h):return torch.nn.functional.normalize(m(h),dim=-1)
def roi(q,pool):return torch.nn.functional.normalize(pool@q,dim=-1)
def loss_parts(m,d,mask,shuffle):
 qa=query_rep(m,d['A_h']);qb=query_rep(m,d['B_h']);ra=roi(qa,d['A_pool']);rb=roi(qb,d['B_pool']);rb=rb[d['shuffle']] if shuffle else rb;sim=ra@rb.T/.07;eye=torch.eye(len(ra),dtype=torch.bool);allowed=eye|mask;v=mask.any(1);u=mask.any(0);ls=torch.logsumexp(sim.masked_fill(~allowed,-torch.inf),1)-sim.diag();rs=torch.logsumexp(sim.T.masked_fill(~allowed.T,-torch.inf),1)-sim.diag();corr=(ls[v].mean()+rs[u].mean())/2
 keep=(((qa-d['A_source']).double().square().sum(-1).mean()+(qb-d['B_source']).double().square().sum(-1).mean())/4);return corr.double()+keep,corr.double(),keep
def fit(initial,z,e,kind):
 m=head(initial);opt=optimizer(m);d={v+'_h':torch.tensor(z[v]['h']) for v in 'AB'};d.update({v+'_pool':torch.tensor(e[v+'_pool'],dtype=torch.float32) for v in 'AB'});d['shuffle']=torch.tensor(e['shuffle']).long();mask=torch.tensor(e['geometry_negatives'] if kind=='corr_noRGB' else e['RGB_negatives']).bool();shuffle=kind=='corr_shuffled';base=phash(m);obase=ohash(opt)
 if not mask.any():return state(m),{'active':False,'accepted':False,'scale':0.,'trials':[],'parameter_step_L2':0.,'before_sha256':base,'after_sha256':base,'C_view_read':False}
 with torch.no_grad():
  for v in 'AB':d[v+'_source']=query_rep(m,d[v+'_h']).detach().clone()
 old={n:v.detach().clone() for n,v in m.named_parameters()};oldopt=copy.deepcopy(opt.state_dict());opt.zero_grad(set_to_none=True);L,lc,lk=loss_parts(m,d,mask,shuffle);before=float(L.detach());cbefore=float(lc.detach());kbefore=float(lk.detach());L.backward();gn=float(torch.sqrt(sum(v.grad.square().sum() for v in m.parameters())));assert np.isfinite(gn);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();proposal={n:v.detach().clone() for n,v in m.named_parameters()};trials=[];accepted=False;scale=0.
 with torch.no_grad():
  for j in range(9):
   alpha=2.**(-j)
   for n,v in m.named_parameters():v.copy_(old[n]+alpha*(proposal[n]-old[n]))
   ll,cc,kk=loss_parts(m,d,mask,shuffle);after=float(ll);trials.append({'scale':alpha,'loss':after if np.isfinite(after) else None,'contrastive':float(cc) if np.isfinite(float(cc)) else None,'keep':float(kk) if np.isfinite(float(kk)) else None})
   if np.isfinite(after) and after<=before-max(1e-12,1e-6*abs(before)):accepted=True;scale=alpha;break
  if not accepted:
   for n,v in m.named_parameters():v.copy_(old[n])
   opt.load_state_dict(oldopt)
  ll,cc,kk=loss_parts(m,d,mask,shuffle);delta=float(np.sqrt(sum(np.square(v-initial[k]).sum() for k,v in state(m).items())))
 if accepted:assert float(ll)<=before-max(1e-12,1e-6*abs(before)) and delta>0
 else:assert phash(m)==base and ohash(opt)==obase and delta==0
 opt.zero_grad(set_to_none=True);return state(m),{'active':True,'accepted':accepted,'scale':scale,'loss_before':before,'loss_after':float(ll),'contrastive_before':cbefore,'contrastive_after':float(cc),'keep_before':kbefore,'keep_after':float(kk),'gradient_norm':gn,'parameter_step_L2':delta,'trials':trials,'before_sha256':base,'after_sha256':phash(m),'optimizer_before_sha256':obase,'optimizer_after_sha256':ohash(opt),'directed_negative_pairs':int(mask.sum()),'valid_anchor_rows':int(mask.any(1).sum()),'C_view_read':False,'GT_expert_feedback_used':False}
def main():
 p=read(H/'experiment/FIT_PLAN_LOCK.json');pin(p);out=ROOT/p['fit_root'];assert not out.exists();out.mkdir(parents=True)
 for f in ['weights_PRIVATE','cases_PRIVATE']:(out/f).mkdir()
 initial=load(ROOT/p['head_initial']);evroot=ROOT/p['evidence_root'];es=read(evroot/'SEALED.json');counts={k:{'proposed':0,'accepted':0,'rejected':0} for k in ADAPT};start=time.perf_counter();torch.set_num_threads(4)
 for i,r in enumerate(rows()):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(evroot/rel)==es['files'][rel];e=load(evroot/rel);z=ab(r)
  for kind in ADAPT:
   states,rec=fit(initial,z,e,kind);rec.setdefault('GT_expert_feedback_used',False);np.savez_compressed(out/'weights_PRIVATE'/(sid+'_'+kind+'.npz'),**states);dump(out/'cases_PRIVATE'/(sid+'_'+kind+'.json'),rec);counts[kind]['proposed']+=int(rec['active']);counts[kind]['accepted']+=int(rec['accepted']);counts[kind]['rejected']+=int(rec['active'] and not rec['accepted'])
  if(i+1)%128==0:print('FIT74',i+1,counts,flush=True)
 dump(out/'COUNTS.json',{'n':len(rows()),'branches':counts,'updated_parameters':sum(v.size for v in initial.values()),'seconds':time.perf_counter()-start,'C_read':False,'expert_or_GT_used':False});seal(out,'R74_ALL_AB_ONLY_WEIGHTS_SEALED_BEFORE_C_OR_TASK_LOOKUP',n=len(rows()));print(read(out/'COUNTS.json'))
if __name__=='__main__':main()
