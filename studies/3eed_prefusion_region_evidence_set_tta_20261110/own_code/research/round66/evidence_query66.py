"""Validity repair v2: control matches BOTH seed count and masked query-key count.
Reuse sealed v1 source/region responses. Change only control construction.
"""
import sys,time,torch
from common66 import *
import interface66 as f
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint

def matched_controls(xyz,regions,query_indices):
 anchor=np.zeros(len(xyz),bool);anchor[np.asarray(query_indices,int)]=True;assert anchor.sum()==len(query_indices);outside=~np.any(regions,axis=0);Q=np.flatnonzero(outside&anchor);N=np.flatnonzero(outside&~anchor);out=[]
 for m in regions:
  n=int(m.sum());q=int((m&anchor).sum());c=np.zeros(len(xyz),bool);ok=4<=n<=len(xyz)-4 and len(Q)>=q and len(N)>=n-q
  if ok:
   available=Q if q else N;center=xyz[available[np.argmax(np.linalg.norm(xyz[available]-xyz[m].mean(0),axis=1))]];qi=Q[np.argsort(np.linalg.norm(xyz[Q]-center,axis=1),kind='stable')[:q]];ni=N[np.argsort(np.linalg.norm(xyz[N]-center,axis=1),kind='stable')[:n-q]];c[np.r_[qi,ni]]=True;assert c.sum()==n and int((c&anchor).sum())==q and not(m&c).any()
  out.append((c,ok))
 return out

def main():
 H=ROOT/'research/round66';p=read(H/'experiment/MATCHED_QUERY_DESIGN_LOCK_v2.json');pin(p);oldroot=ROOT/p['original_evidence_root'];olds=read(oldroot/'SEALED.json');cap=ROOT/p['capture_root'];out=ROOT/p['evidence_root'];assert not out.exists();out.mkdir(parents=True)
 for k in ['samples_PRIVATE','cases_PRIVATE']:(out/k).mkdir()
 model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();calls=eligible=0;counts={'inputs':0,'both_view_valid_groups':0,'stable_positive_groups':0,'stable_negative_groups':0,'single_C_never_read':True};rows=read(ROOT/p['manifest']);allprocess=[]
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(oldroot/rel)==olds['files'][rel];e=load(oldroot/rel);capz=load(cap/rel);zs={v:{k[2:]:val for k,val in capz.items() if k.startswith(v+'_')} for v in 'AB'};baseline=e['baseline'];pert=e['perturbed'].copy();valid=np.zeros_like(e['valid']);proof=[];oldproof=read(oldroot/'cases_PRIVATE'/(sid+'.json'))['proof'];K=len(e['members'])
  for vi,v in enumerate('AB'):
   ctrl=matched_controls(zs[v]['xyz'][0].astype(float),e[v+'_region_masks'],zs[v]['query_points_sample_inds']);newm=[]
   for j,(m,ok) in enumerate(ctrl):
    newm.append(m);valid[vi,j]=ok and bool(capz['category_available']);assert not valid[vi,j] or e['valid'][vi,j]
    if valid[vi,j]:
     proof.extend([a for a in oldproof if a['view']==v and a['pool_position']==j and a['kind']==0])
     for ti,category in enumerate([False,True]):
      y=f.replay(model,zs[v],m,category);calls+=1;pp=y['p']
      if v=='B':
       mapped=e['B_W']@pp;pp=float(e['B_fixed_mass'])*mapped/max(float(mapped.sum()),1e-300)+(1-float(e['B_fixed_mass']))*e['base']
      pert[vi,ti,1,j]=pp;proof.append({'view':v,'kind':1,'category':category,'pool_position':j,**y['proof']})
    else:
     for ti in range(2):
      for ki in range(2):pert[vi,ti,ki,j]=baseline[vi,ti]
   e[v+'_control_masks']=np.array(newm)
  drops=np.zeros_like(e['drops'])
  for vi in range(2):
   for ti in range(2):
    for ki in range(2):
     for j,m in enumerate(e['members']):drops[vi,ti,ki,j]=np.log(max(float(baseline[vi,ti,m].sum()),1e-300))-np.log(max(float(pert[vi,ti,ki,j,m].sum()),1e-300))
  full=drops[:,0,0]-drops[:,0,1];cat=drops[:,1,0]-drops[:,1,1];E=full-cat;pos=valid.all(0)&(E>p['sign_epsilon']).all(0);neg=valid.all(0)&(E< -p['sign_epsilon']).all(0);stable=pos|neg;tested=e['members'][stable].any(0) if stable.any() else np.zeros(256,bool);S=e['members'][pos].any(0) if pos.any() else np.zeros(256,bool);update=bool(pos.any() and neg.any());eligible+=int(update);e.update(perturbed=pert,drops=drops,valid=valid,E_views=E,E=E.mean(0),full_only=full.mean(0),category_only=cat.mean(0),control_only=(drops[:,0,1]-drops[:,1,1]).mean(0),stable=stable,positive=pos,negative=neg,tested_query_mask=tested,positive_query_mask=S,update_allowed=np.array(update));np.savez_compressed(out/rel,**e);dump(out/'cases_PRIVATE'/(sid+'.json'),{'proof':proof,'category_available':bool(capz['category_available']),'update_allowed':update,'region_proof_reused_from_sealed_v1':True});counts['inputs']+=1;counts['both_view_valid_groups']+=int(valid.all(0).sum());counts['stable_positive_groups']+=int(pos.sum());counts['stable_negative_groups']+=int(neg.sum());allprocess.append({'row':'DEV_'+str(i+1).zfill(5),'platform':r['platform'],'pool_groups':K,'valid_groups':int(valid.all(0).sum()),'stable_positive_groups':int(pos.sum()),'stable_negative_groups':int(neg.sum()),'update_allowed':update})
  if (i+1)%32==0:print('MATCHED_QUERY_EVIDENCE66',i+1,len(rows),calls,eligible,flush=True)
 assert fingerprint(model)==fp;cost={'n':336,'encoder_decoder_replays':calls,'new_full_source_forwards':0,'update_eligible_inputs':eligible,'counts':counts,'source_unchanged':True,'C_used_for_supervision':False,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'original_evidence_seal_sha256':sha(oldroot/'SEALED.json'),'only_control_selection_changed':True};dump(out/'COSTS.json',cost);dump(out/'PROCESS_PUBLIC_SAFE.json',allprocess);seal(out,'R66_QUERY_AND_SEED_SIZE_MATCHED_CONTROL_EVIDENCE_NO_C_NO_GT',n=336);print(cost)
if __name__=='__main__':main()
