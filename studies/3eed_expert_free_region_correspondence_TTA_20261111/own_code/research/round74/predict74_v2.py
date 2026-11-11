"""After-fit C retrieval and fixed A256 task readers; no method selection."""
from common74 import *
from region74 import region_pool,sampled_support
from fit74 import query_rep,roi
from groups66 import native,transport
from direct74 import fused_AB
import torch,time
sys.path.insert(0,str(ROOT/'research/round58'));from numerics58 import head
DIAGNOSTIC_ARMS=['heldout_C_source']+['heldout_C_'+k for k in ADAPT]
MAIN_ARMS=ARMS[:2]+['frozen_AB_repr']+ARMS[2:]
TASK_ARMS=MAIN_ARMS+DIAGNOSTIC_ARMS
def logits(q,t,m):
 l=q@t.T/.07;return torch.logsumexp(l[:,m],-1)-torch.logsumexp(l[:,~m],-1)
def viewprob(q,z,z0):
 t=torch.tensor(z['tokens']);mask=torch.tensor(z['token_mask']).bool();delta=logits(q,t,mask).double()-z0;return (torch.tensor(native(z)).log()+delta).softmax(-1).numpy()
def retrieve(a,b,N):
 n=len(a);allowed=N|np.eye(n,dtype=bool);valid=N.any(1)
 if not valid.any():return {'valid':0,'hits':0,'gap_sum':0.,'positive_cosine_sum':0.,'unknown_rows':n}
 sim=a@b.T;chosen=np.where(allowed,sim,-np.inf).argmax(1);neg=np.where(N,sim,-np.inf).max(1);return {'valid':int(valid.sum()),'hits':int((chosen[valid]==np.flatnonzero(valid)).sum()),'gap_sum':float((sim.diagonal()[valid]-neg[valid]).sum()),'positive_cosine_sum':float(sim.diagonal()[valid].sum()),'unknown_rows':int((~valid).sum())}
def main():
 p=read(H/'experiment/PREDICT_PLAN_LOCK_attempt002.json');pin(p);out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','aux_PRIVATE','cases_PRIVATE']:(out/f).mkdir()
 fs=read(ROOT/p['fit_root']/'SEALED.json');assert fs['n']==64;initial=load(ROOT/p['head_initial']);start=time.perf_counter();statuses=[];torch.set_grad_enabled(False);torch.set_num_threads(4)
 for i,r in enumerate(rows()):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';z=ab(r);e=load(ROOT/p['evidence_root']/rel);c=cview(r);raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3].copy()
  if r['platform']=='drone':raw[:,2]+=np.float32(1.8)
  PC,nC=region_pool(c,e['region_boxes']);sa=sampled_support(raw,z['A']['point_indices'],e['region_boxes']);sc=sampled_support(raw,c['point_indices'],e['region_boxes']);common=np.array([len(np.intersect1d(a,b)) for a,b in zip(sa,sc)]);eligible=(nC>0)&(common>=4);N=e['geometry_negatives']&eligible[:,None]&eligible[None,:];source=head(initial);Q0={v:query_rep(source,torch.tensor(zz['h'])) for v,zz in [('A',z['A']),('B',z['B']),('C',c)]};z0={v:logits(Q0[v],torch.tensor(zz['tokens']),torch.tensor(zz['token_mask']).bool()).double() for v,zz in [('A',z['A']),('B',z['B']),('C',c)]};assoc=load(ROOT/'crossplatform/round69/predicted2304_attempt001/association_PRIVATE'/(sid+'.npz'),['C_W','C_fixed_mass','base']);csource=transport(native(c),assoc['C_W'],float(assoc['C_fixed_mass']),assoc['base']);base=e['base'];W=e['B_W'];mass=float(e['B_fixed_mass']);pA=native(z['A']);pB=native(z['B']);frozen=(pA+transport(pB,W,mass,base))/2;probs={'frozen_AB':frozen,'frozen_AB_repr':fused_AB(z,e),'heldout_C_source':csource};aux={};arrays={};stepcases={}
  for kind in ['source']+ADAPT:
   if kind=='source':m=source
   else:
    path=ROOT/p['fit_root']/'weights_PRIVATE'/(sid+'_'+kind+'.npz');relw=path.relative_to(ROOT/p['fit_root']).as_posix();assert sha(path)==fs['files'][relw];m=head(load(path));stepcases[kind]=read(ROOT/p['fit_root']/'cases_PRIVATE'/(sid+'_'+kind+'.json'))
   Q={v:query_rep(m,torch.tensor(zz['h'])) for v,zz in [('A',z['A']),('B',z['B']),('C',c)]};ra=roi(Q['A'],torch.tensor(e['A_pool'],dtype=torch.float32));rb=roi(Q['B'],torch.tensor(e['B_pool'],dtype=torch.float32));rc=roi(Q['C'],torch.tensor(PC,dtype=torch.float32));rab=torch.nn.functional.normalize((ra+rb)/2,dim=-1).numpy();rc=rc.numpy();aux[kind]={'AB_to_C':retrieve(rab,rc,N),'C_to_AB':retrieve(rc,rab,N)};arrays[kind+'_AB']=rab;arrays[kind+'_C']=rc
   if kind!='source':
    pa=viewprob(Q['A'],z['A'],z0['A']);pb=viewprob(Q['B'],z['B'],z0['B']);probs[kind]=(pa+transport(pb,W,mass,base))/2 if stepcases[kind]['accepted'] else frozen.copy();pc=viewprob(Q['C'],c,z0['C']);probs['heldout_C_'+kind]=transport(pc,assoc['C_W'],float(assoc['C_fixed_mass']),assoc['base']) if stepcases[kind]['accepted'] else csource.copy()
  # Historical readers, including CLIP, never participate in any fit/eligibility.
  historic=load(ROOT/'crossplatform/round71/predicted2640_attempt001'/rel);old=dict(zip(historic['arm_names'],historic['scores']));probs.update(source=old['source'],R59=old['R59'],CLIP_reference=old['CLIP_direct']);score=np.stack([probs[k] for k in TASK_ARMS]);q=score.argmax(1);assert np.isfinite(score).all();assert np.array_equal(historic['source_boxes'],z['A']['boxes']);np.savez_compressed(out/rel,arm_names=np.array(TASK_ARMS),scores=score,queries=q,source_boxes=z['A']['boxes'],selected_original_boxes=z['A']['boxes'][q]);np.savez_compressed(out/'aux_PRIVATE'/(sid+'.npz'),**arrays,eval_negatives=N,C_pool=PC,C_shared_raw_counts=common,C_eligible=eligible);dump(out/'cases_PRIVATE'/(sid+'.json'),aux);st={}
  for k in TASK_ARMS:
   kk=k.removeprefix('heldout_C_');adaptive=kk in ADAPT;rec=stepcases[kk] if adaptive else {};st[k]={'unknown':bool(not(e['geometry_negatives'] if kk=='corr_noRGB' else e['RGB_negatives']).any()) if adaptive else False,'fallback':not rec['accepted'] if adaptive else False,'no_parameter_update':not(adaptive and rec['accepted'])}
  statuses.append({'id':sid,'arm_statuses':st})
 dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COUNTS.json',{'n':64,'seconds':time.perf_counter()-start,'C_used_for_fit_or_step_selection':False,'new_full_model_or_expert_calls':0});seal(out,'R74_SEVEN_MAIN_FOUR_HELDOUT_C_AND_RETRIEVAL_BEFORE_TASK_GT',n=64,arms=TASK_ARMS);print(read(out/'COUNTS.json'))
if __name__=='__main__':main()
