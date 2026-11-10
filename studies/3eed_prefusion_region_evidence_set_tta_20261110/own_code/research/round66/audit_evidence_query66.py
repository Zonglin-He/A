"""Independent v2 control-only replay. Original region/source proof reused."""
import sys,time,torch
from common66 import *
import independent66 as f
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint

def controls(x,regions,ids):
 anchors=np.zeros(len(x),bool);anchors[np.asarray(ids,int)]=True;assert int(anchors.sum())==len(ids);outside=~regions.any(0);yes=np.where(outside&anchors)[0];no=np.where(outside&~anchors)[0];out=[]
 for region in regions:
  n=int(region.sum());q=int(region[ids].sum());ok=4<=n<=len(x)-4 and len(yes)>=q and len(no)>=n-q;m=np.zeros(len(x),bool)
  if ok:
   options=yes if q else no;point=options[np.argmax(np.linalg.norm(x[options]-x[region].mean(0),axis=1))];a=yes[np.argsort(np.linalg.norm(x[yes]-x[point],axis=1),kind='stable')[:q]];b=no[np.argsort(np.linalg.norm(x[no]-x[point],axis=1),kind='stable')[:n-q]];m[np.r_[a,b]]=True;assert m.sum()==n and m[ids].sum()==q
  out.append((m,ok))
 return out

def main():
 H=ROOT/'research/round66';p=read(H/'experiment/EVIDENCE_QUERY_AUDIT_PLAN_LOCK_v2.json');pin(p);oldroot=ROOT/p['original_evidence_root'];ev=ROOT/p['evidence_root'];cert=read(H/'independent_evidence_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['evidence_seal_sha256']==sha(oldroot/'SEALED.json');ss=read(ev/'SEALED.json');rows=read(ROOT/p['manifest']);model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();calls=0;maxerr=0.;matched=0
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(ev/rel)==ss['files'][rel];e=load(ev/rel);old=load(oldroot/rel);cap=load(ROOT/p['capture_root']/rel);z={v:{k[2:]:val for k,val in cap.items() if k.startswith(v+'_')} for v in 'AB'}
  for k in ['base','B_W','B_fixed_mass','members','group_ids','baseline','A_region_masks','B_region_masks']:assert np.array_equal(e[k],old[k])
  baseline=old['baseline'];pert=old['perturbed'].copy();valid=np.zeros_like(old['valid'])
  for vi,v in enumerate('AB'):
   mm=controls(z[v]['xyz'][0].astype(float),old[v+'_region_masks'],z[v]['query_points_sample_inds'])
   for j,(mask,ok) in enumerate(mm):
    assert np.array_equal(mask,e[v+'_control_masks'][j]);valid[vi,j]=ok and bool(cap['category_available']);assert not valid[vi,j] or old['valid'][vi,j]
    if valid[vi,j]:
     region=old[v+'_region_masks'][j];assert region.sum()==mask.sum() and region[z[v]['query_points_sample_inds']].sum()==mask[z[v]['query_points_sample_inds']].sum();matched+=1
     for ti,category in enumerate([False,True]):
      y=f.replay(model,z[v],mask,category);calls+=1;pp=y['p'] if v=='A' else f.mapped(y['p'],e['B_W'],float(e['B_fixed_mass']),e['base']);pert[vi,ti,1,j]=pp
    else:
     for ti in range(2):
      for ki in range(2):pert[vi,ti,ki,j]=baseline[vi,ti]
  err=float(abs(pert-e['perturbed']).max());maxerr=max(maxerr,err);assert err<1e-11 and np.array_equal(valid,e['valid']);drop=np.zeros_like(e['drops'])
  for vi in range(2):
   for ti in range(2):
    for ki in range(2):
     for j,m in enumerate(e['members']):drop[vi,ti,ki,j]=np.log(max(float(baseline[vi,ti,m].sum()),1e-300))-np.log(max(float(pert[vi,ti,ki,j,m].sum()),1e-300))
  full=drop[:,0,0]-drop[:,0,1];cat=drop[:,1,0]-drop[:,1,1];E=full-cat;pos=valid.all(0)&(E>p['sign_epsilon']).all(0);neg=valid.all(0)&(E< -p['sign_epsilon']).all(0);st=pos|neg
  for k,a in [('drops',drop),('E_views',E),('E',E.mean(0)),('full_only',full.mean(0)),('category_only',cat.mean(0)),('control_only',(drop[:,0,1]-drop[:,1,1]).mean(0))]:assert np.allclose(a,e[k],rtol=0,atol=1e-10)
  for k,a in [('positive',pos),('negative',neg),('stable',st),('tested_query_mask',e['members'][st].any(0) if st.any() else np.zeros(256,bool)),('positive_query_mask',e['members'][pos].any(0) if pos.any() else np.zeros(256,bool))]:assert np.array_equal(a,e[k])
  assert bool(e['update_allowed'])==bool(pos.any() and neg.any())
  if (i+1)%32==0:print('AUDIT_MATCHED_QUERY_EVIDENCE66',i+1,len(rows),calls,flush=True)
 assert fingerprint(model)==fp;out=H/'independent_query_evidence_audit_attempt001';assert not out.exists();out.mkdir();rr={'state':'PASS_R66_INDEPENDENT_SEED_AND_QUERY_KEY_MATCHED_CONTROL_REPAIR','n':336,'source_full_forward_replays':0,'reused_v1_source_full_forward_proof':672,'independent_encoder_decoder_replays':calls,'all_supported_view_group_seed_and_query_counts_matched':matched,'max_probability_error':maxerr,'source_unchanged':True,'C_features_used':False,'task_GT_used':False,'evidence_seal_sha256':sha(ev/'SEALED.json'),'original_evidence_audit_sha256':sha(H/'independent_evidence_audit_attempt001/AUDIT_RESULT.json'),'program_sha256':sha(__file__),'helper_sha256':sha(H/'independent66.py'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'AUDIT_RESULT.json',rr);print(rr)
if __name__=='__main__':main()
