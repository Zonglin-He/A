"""A/B description-specific region evidence, no C features, no task labels."""
import sys,time,torch
from common66 import *
import interface66 as f,groups66 as g
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint

def main():
 H=ROOT/'research/round66';p=read(H/'experiment/EVIDENCE_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);cap=ROOT/p['capture_root'];cs=read(cap/'SEALED.json');out=ROOT/p['evidence_root'];assert not out.exists();out.mkdir(parents=True)
 for d in ['samples_PRIVATE','cases_PRIVATE']:(out/d).mkdir()
 model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();nreplays=0;eligible=0;counts={'inputs':0,'both_view_valid_groups':0,'stable_positive_groups':0,'stable_negative_groups':0,'single_C_never_read':True};allprocess=[]
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel];cc=load(cap/rel);z={v:{k[2:]:a for k,a in cc.items() if k.startswith(v+'_')} for v in 'AB'};path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);ids=np.unique(np.r_[z['A']['point_indices'],z['B']['point_indices']]);x=np.array(raw[ids,:3],np.float32,copy=True)
  if r['platform']=='drone':x[:,2]+=1.8
  af,_,_=g.affinity(z['A']['boxes'].astype(float),z['B']['boxes'].astype(float),x.astype(float));W=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));base=g.native(z['A']);mass=float((W@g.native(z['B'])).sum());pAB=(base+g.transport(g.native(z['B']),W,mass,base))/2;groups,pool=g.make_groups(z['A']['boxes'].astype(float),x.astype(float),pAB);K=len(pool);members=np.array([np.isin(np.arange(256),groups[k]) for k in pool]);mm={v:g.masks(z[v]['xyz'][0].astype(float),z['A']['boxes'].astype(float),groups,pool) for v in 'AB'};baseline=np.zeros((2,2,256));pert=np.zeros((2,2,2,K,256));valid=np.zeros((2,K),bool);proof=[];payload={'B_W':W,'B_fixed_mass':np.array(mass),'base':base,'members':members,'group_ids':np.array(pool,np.int64)}
  for vi,v in enumerate('AB'):
   for ti,category in enumerate([False,True]):
    y=f.replay(model,z[v],category=category);nreplays+=1
    if not category:
     for k in ['h','query','tokens']:assert np.array_equal(y[k],z[v][k]),('zero_interface',i,v,k)
    baseline[vi,ti]=y['p'] if v=='A' else g.transport(y['p'],W,mass,base)
   masks=[];controls=[]
   for j,(mask,control,ok) in enumerate(mm[v]):
    masks.append(mask);controls.append(control);valid[vi,j]=ok and bool(cc['category_available'])
    for ti,category in enumerate([False,True]):
     for ki,m in enumerate([mask,control]):
      if valid[vi,j]:
       y=f.replay(model,z[v],m,category);nreplays+=1;pert[vi,ti,ki,j]=y['p'] if v=='A' else g.transport(y['p'],W,mass,base);proof.append({'view':v,'kind':ki,'category':category,'pool_position':j,**y['proof']})
      else:pert[vi,ti,ki,j]=baseline[vi,ti]
   payload[v+'_region_masks']=np.array(masks);payload[v+'_control_masks']=np.array(controls)
  drop=np.zeros((2,2,2,K))
  for vi in range(2):
   for ti in range(2):
    for ki in range(2):
     for j in range(K):drop[vi,ti,ki,j]=np.log(max(float(baseline[vi,ti,members[j]].sum()),1e-300))-np.log(max(float(pert[vi,ti,ki,j,members[j]].sum()),1e-300))
  full=drop[:,0,0]-drop[:,0,1];cat=drop[:,1,0]-drop[:,1,1];E=full-cat;eps=p['sign_epsilon'];available=valid.all(0);positive=available&(E>eps).all(0);negative=available&(E< -eps).all(0);stable=positive|negative;tested=members[stable].any(0) if stable.any() else np.zeros(256,bool);selected=members[positive].any(0) if positive.any() else np.zeros(256,bool);update=bool(positive.any() and negative.any());eligible+=int(update);counts['inputs']+=1;counts['both_view_valid_groups']+=int(available.sum());counts['stable_positive_groups']+=int(positive.sum());counts['stable_negative_groups']+=int(negative.sum());payload.update(baseline=baseline,perturbed=pert,drops=drop,valid=valid,E_views=E,E=E.mean(0),full_only=full.mean(0),category_only=cat.mean(0),control_only=(drop[:,0,1]-drop[:,1,1]).mean(0),stable=stable,positive=positive,negative=negative,tested_query_mask=tested,positive_query_mask=selected,update_allowed=np.array(update));np.savez_compressed(out/rel,**payload);dump(out/'cases_PRIVATE'/(sid+'.json'),{'proof':proof,'category_available':bool(cc['category_available']),'update_allowed':update});allprocess.append({'row':'DEV_'+str(i+1).zfill(5),'platform':r['platform'],'pool_groups':K,'valid_groups':int(available.sum()),'stable_positive_groups':int(positive.sum()),'stable_negative_groups':int(negative.sum()),'UNKNOWN_groups':int((~stable).sum()),'update_allowed':update,'A_region_seed_counts':[int(a[0].sum()) for a in mm['A']],'B_region_seed_counts':[int(a[0].sum()) for a in mm['B']]})
  if (i+1)%16==0:print('EVIDENCE66',i+1,len(rows),nreplays,eligible,flush=True)
 assert fingerprint(model)==fp;cost={'n':len(rows),'encoder_decoder_replays':nreplays,'update_eligible_inputs':eligible,'counts':counts,'source_unchanged':True,'C_used_for_supervision':False,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COSTS.json',cost);dump(out/'PROCESS_PUBLIC_SAFE.json',allprocess);seal(out,'R66_A_B_REGION_EVIDENCE_AND_PARTIAL_SETS_SEALED_NO_GT_NO_C',n=336);print(cost)
if __name__=='__main__':main()
